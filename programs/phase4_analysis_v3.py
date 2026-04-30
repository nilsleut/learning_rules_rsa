"""
phase4_analysis_v3.py
=====================
Phase 4 (v3): Filter Visualization + Partial RSA

Changes from v2:
    1. Model RDMs loaded from seed subdirectories (outputs/model_rdms/seed_N/)
       and averaged across all available seeds — consistent with seed_variability.png
    2. Pixel RDM uses 224x224 input — consistent with main 224x224 run
    3. Gabor analysis runs for all rules where model_weights_*.pt exists
       (STDP skipped automatically if missing, no crash)
    4. STDP excluded gracefully from filter visualization if weights missing

Usage:
    py phase4_analysis_v3.py

Prerequisites:
    - outputs/model_rdms/seed_0/ ... seed_N/ must exist (from 224x224 run)
    - outputs/model_weights_*.pt must exist for Gabor (STDP optional)
    - outputs_720/fmri_rdm_*.npy must exist
    - outputs/rsa_results_cnn.csv must exist (for cross-check)
"""

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FMRI_DIR    = BASE_DIR / "outputs_720"
THINGS_DIR  = Path(r"C:\Users\nilsl\Desktop\Projekte\RSA\Datensatz\images_THINGS\object_images")
SUBJECTS    = ["sub-01", "sub-02", "sub-03"]
IMG_SIZE    = 224  # must match main run

COLORS = {
    "Random Weights":     "#999999",
    "Backprop":           "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}

RULES     = ["Random Weights", "Backprop", "Feedback Alignment",
             "Predictive Coding", "STDP"]
LAYER_MAP = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def upper_tri(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]

def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None

def mean_brain_rdm(roi):
    rdms = [load_fmri_rdm(roi, s) for s in SUBJECTS]
    rdms = [r for r in rdms if r is not None]
    if not rdms: return None
    n = min(r.shape[0] for r in rdms)
    return np.mean([r[:n, :n] for r in rdms], axis=0)

def load_stim_order():
    p = FMRI_DIR / "stim_order_sub-01.txt"
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]

def find_img(stimulus):
    name    = stimulus.replace(".jpg", "")
    parts   = name.split("_")
    last    = parts[-1]
    concept = "_".join(parts[:-1]) if (len(parts) > 1 and len(last) <= 4
                                        and any(c.isdigit() for c in last)) else name
    for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
        hits = sorted(THINGS_DIR.glob(pat))
        if hits: return hits[0]
    for folder in THINGS_DIR.iterdir():
        if folder.name.lower() == concept.lower():
            imgs = sorted(folder.glob("*.jpg"))
            if imgs: return imgs[0]
    return None


# ── Seed-averaged model RDM loading ───────────────────────────────────────────

def find_seed_dirs():
    """Return sorted list of seed subdirectories under outputs/model_rdms/."""
    rdm_base = OUTPUTS_DIR / "model_rdms"
    seeds = sorted([d for d in rdm_base.iterdir()
                    if d.is_dir() and d.name.startswith("seed_")])
    return seeds

def load_model_rdm_mean(rule, layer):
    """
    Load model RDM averaged across all available seeds.
    Falls back to flat outputs/model_rdms/ if no seed subdirs found.
    """
    rule_key  = rule.lower().replace(" ", "_")
    seed_dirs = find_seed_dirs()

    if seed_dirs:
        rdms = []
        for sd in seed_dirs:
            p = sd / f"rdm_{rule_key}_{layer}.npy"
            if p.exists():
                rdms.append(np.load(str(p)))
        if not rdms:
            return None
        n = min(r.shape[0] for r in rdms)
        mean_rdm = np.mean([r[:n, :n] for r in rdms], axis=0)
        return mean_rdm
    else:
        # Fallback: flat directory (old structure)
        p = OUTPUTS_DIR / "model_rdms" / f"rdm_{rule_key}_{layer}.npy"
        return np.load(str(p)) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# PART A — FILTER VISUALIZATION + GABOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def load_conv1_filters():
    """
    Load Conv1 weights from outputs/model_weights_*.pt.
    Handles three different key structures:
      - BP / Random Weights: 'conv1.weight'  (standard nn.Conv2d)
      - FA:                  'conv1.0.W'     (custom FAConv layer)
      - PC:                  'W1.weight'     (custom PC layer)
      - STDP: skipped if .pt missing
    """
    CONV1_CANDIDATES = [
        "conv1.weight",   # BP, Random
        "conv1.0.W",      # FA
        "W1.weight",      # PC
    ]

    filters = {}
    for rule in RULES:
        rule_key = rule.lower().replace(" ", "_")
        path     = OUTPUTS_DIR / f"model_weights_{rule_key}.pt"
        if not path.exists():
            print(f"  {rule}: {path.name} not found — skipping")
            continue

        state = torch.load(str(path), map_location="cpu", weights_only=False)

        found = False
        for candidate in CONV1_CANDIDATES:
            if candidate in state:
                val = state[candidate]
                if val.ndim == 4 and val.shape[1] == 3:
                    filters[rule] = val.numpy()
                    print(f"  {rule}: key='{candidate}' shape={val.shape} loaded")
                    found = True
                    break

        if not found:
            # Fallback: scan all keys for correct shape
            for key, val in state.items():
                if val.ndim == 4 and val.shape[1] == 3:
                    filters[rule] = val.numpy()
                    print(f"  {rule}: fallback key='{key}' shape={val.shape} loaded")
                    found = True
                    break

        if not found:
            print(f"  {rule}: no Conv1 weight found in state dict — skipping")

    return filters


def visualize_filters(filters, n_show=16):
    if not filters:
        print("  No filters to visualize.")
        return
    rules = list(filters.keys())
    fig, axes = plt.subplots(len(rules), n_show,
                             figsize=(n_show * 1.1, len(rules) * 1.3))
    if len(rules) == 1:
        axes = axes[np.newaxis, :]

    for ri, rule in enumerate(rules):
        w = filters[rule]
        axes[ri, 0].set_ylabel(rule, fontsize=8, rotation=0,
                                ha="right", va="center", labelpad=60)
        for ci in range(n_show):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci < w.shape[0]:
                f = np.transpose(w[ci], (1, 2, 0))
                f = f - f.min()
                if f.max() > 0: f /= f.max()
                ax.imshow(f, interpolation="nearest")

    plt.suptitle("Conv1 filters per learning rule (first 16)", fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUTPUTS_DIR / "filter_visualization.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def gabor_score(w):
    """FFT peakedness as proxy for Gabor-likeness."""
    scores = []
    for f in w:
        fg  = f.mean(0)
        fft = np.abs(np.fft.fftshift(np.fft.fft2(fg)))
        h, ww = fft.shape
        fft[h // 2, ww // 2] = 0  # remove DC
        scores.append(fft.max() / (fft.mean() + 1e-8))
    return np.array(scores)


def analyze_gabor(filters):
    if not filters:
        print("  No filters for Gabor analysis.")
        return None

    rows = []
    for rule, w in filters.items():
        s = gabor_score(w)
        rows.append({"rule": rule, "mean": s.mean(), "std": s.std()})
        print(f"  {rule:25s}: {s.mean():.2f} ± {s.std():.2f}")

    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    df.to_csv(str(OUTPUTS_DIR / "gabor_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["rule"], df["mean"], yerr=df["std"], capsize=4,
           color=[COLORS.get(r, "gray") for r in df["rule"]], alpha=0.85)
    ax.set_ylabel("Gabor-peakedness (FFT peak/mean ratio)", fontsize=11)
    ax.set_xlabel("Learning rule", fontsize=11)
    ax.set_title("Gabor-likeness of Conv1 filters per learning rule", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "gabor_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: gabor_plot.png, gabor_analysis.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART B — PARTIAL RSA (224×224 pixel RDM, seed-averaged model RDMs)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pixel_rdm(paths, n_max=720):
    """
    Pixel RDM at IMG_SIZE resolution — must match main analysis input size.
    """
    tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])

    class DS(Dataset):
        def __init__(self, ps): self.ps = ps
        def __len__(self): return len(self.ps)
        def __getitem__(self, i):
            return tf(Image.open(self.ps[i]).convert("RGB")), i

    loader = DataLoader(DS(paths[:n_max]), batch_size=32,
                        shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            feats.append(imgs.view(imgs.size(0), -1).numpy())
    X = np.concatenate(feats)
    print(f"  Pixel feature matrix: {X.shape}")
    return squareform(pdist(X, metric="correlation"))


def partial_spearman(x, y, z):
    """Partial Spearman r(x,y | z) via rank residualization."""
    def residualize(a, b):
        ar = rankdata(a).astype(float)
        br = rankdata(b).astype(float)
        bc = br - br.mean()
        beta = np.dot(ar, bc) / (np.dot(bc, bc) + 1e-10)
        return ar - beta * br
    r, p = spearmanr(residualize(x, z), residualize(y, z))
    return float(r), float(p)


def run_partial_rsa(pixel_rdm):
    """
    Partial RSA controlling for pixel similarity.
    Uses seed-averaged model RDMs for consistency with main analysis.
    """
    seed_dirs = find_seed_dirs()
    n_seeds   = len(seed_dirs)
    print(f"  Using mean RDM across {n_seeds} seed(s): "
          f"{[d.name for d in seed_dirs]}")

    rows    = []
    main_csv = OUTPUTS_DIR / "rsa_results_cnn.csv"
    main_df  = pd.read_csv(str(main_csv)) if main_csv.exists() else None

    for roi in ["V1", "V2", "LOC", "IT"]:
        brain = mean_brain_rdm(roi)
        if brain is None:
            print(f"  {roi}: brain RDM not found — skipping")
            continue
        n = min(brain.shape[0], pixel_rdm.shape[0])
        print(f"\n  {roi}:")

        for rule in RULES:
            layer = LAYER_MAP[roi]
            mrdm  = load_model_rdm_mean(rule, layer)
            if mrdm is None:
                print(f"    {rule:25s}: RDM not found — skipping")
                continue
            nm = min(n, mrdm.shape[0])

            mv = upper_tri(mrdm[:nm, :nm])
            bv = upper_tri(brain[:nm, :nm])
            pv = upper_tri(pixel_rdm[:nm, :nm])

            r_std, _      = spearmanr(mv, bv)
            r_par, p_par  = partial_spearman(mv, bv, pv)

            # Cross-check vs main rsa_results_cnn.csv
            xcheck = ""
            if main_df is not None:
                main_row = main_df[(main_df["rule"] == rule) &
                                   (main_df["roi"]  == roi)  &
                                   (main_df["layer"] == layer)]
                if not main_row.empty:
                    main_rho = main_row.iloc[0]["rho"]
                    diff = abs(r_std - main_rho)
                    xcheck = ("  ✓ consistent" if diff <= 0.005
                              else f"  ⚠ MISMATCH vs CSV (Δ={diff:.4f})")

            print(f"    {rule:25s}: ρ_std={r_std:.4f}  "
                  f"ρ_partial={r_par:.4f}  Δ={r_par-r_std:+.4f}  "
                  f"p={p_par:.4f}{xcheck}")

            rows.append({
                "roi":         roi,
                "rule":        rule,
                "layer":       layer,
                "rho_std":     round(r_std, 4),
                "rho_partial": round(r_par, 4),
                "p_partial":   round(p_par, 4),
                "delta":       round(r_par - r_std, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(str(OUTPUTS_DIR / "partial_rsa_results.csv"), index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not df.empty:
        rois_plot  = [r for r in ["V1", "V2", "LOC", "IT"] if r in df["roi"].values]
        rules_plot = [r for r in RULES if r in df["rule"].values]
        x      = np.arange(len(rois_plot))
        w      = 0.08
        n_r    = len(rules_plot)
        offs   = np.linspace(-(n_r - 1) * w / 2, (n_r - 1) * w / 2, n_r)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax_i, (col, title) in enumerate([
            ("rho_std",     "Standard RSA"),
            ("rho_partial", "Partial RSA (pixel similarity controlled)")
        ]):
            ax = axes[ax_i]
            for i, rule in enumerate(rules_plot):
                sub  = df[df["rule"] == rule].set_index("roi")
                vals = [sub.loc[roi, col] if roi in sub.index else np.nan
                        for roi in rois_plot]
                ax.bar(x + offs[i], vals, w * 0.9,
                       color=COLORS.get(rule, "gray"), alpha=0.85,
                       label=rule if ax_i == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels(rois_plot, fontsize=11)
            ax.set_ylabel("Spearman ρ", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.axhline(0, color="black", lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].legend(fontsize=8, ncol=2)
        plt.suptitle("Partial RSA: learning rule effects beyond pixel similarity",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(str(OUTPUTS_DIR / "partial_rsa_plot.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: partial_rsa_results.csv, partial_rsa_plot.png")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(partial_df):
    if partial_df.empty:
        return
    print("\n" + "=" * 60)
    print("SUMMARY — values for paper text")
    print("=" * 60)
    print("Standard ρ should match rsa_results_cnn.csv (main analysis).")
    print("Partial ρ is reported as a control — not in abstract.\n")

    for roi in ["V1", "V2", "LOC", "IT"]:
        sub = partial_df[partial_df["roi"] == roi]
        if sub.empty: continue
        best = sub.loc[sub["rho_std"].idxmax()]
        print(f"  {roi}: best = {best['rule']} "
              f"(ρ_std={best['rho_std']:.4f}, ρ_partial={best['rho_partial']:.4f})")

    print("\n  Δ (partial − standard) per rule at V1:")
    v1 = partial_df[partial_df["roi"] == "V1"].set_index("rule")
    for rule in RULES:
        if rule in v1.index:
            print(f"    {rule:25s}: Δ = {v1.loc[rule,'delta']:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 4 v3: Filter Visualization + Partial RSA")
    print(f"  Pixel RDM resolution: {IMG_SIZE}×{IMG_SIZE}")
    seed_dirs = find_seed_dirs()
    print(f"  Seed directories found: {len(seed_dirs)} "
          f"({[d.name for d in seed_dirs]})\n")

    # ── Part A: Filters + Gabor ───────────────────────────────────────────────
    print("=" * 60 + "\nPART A — Filter Visualization + Gabor Analysis\n" + "=" * 60)
    filters  = load_conv1_filters()
    visualize_filters(filters)
    gabor_df = analyze_gabor(filters)

    if gabor_df is not None:
        print("\n  Gabor scores (sorted):")
        print(gabor_df.to_string(index=False))

    # ── Part B: Partial RSA ───────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nPART B — Partial RSA\n" + "=" * 60)
    stimuli = load_stim_order()
    paths   = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  {len(paths)}/{len(stimuli)} THINGS images found")
    if len(paths) < 100:
        print("  WARNING: very few images found — check THINGS_DIR path")

    print(f"\n  Computing pixel RDM at {IMG_SIZE}×{IMG_SIZE}...")
    pixel_rdm  = compute_pixel_rdm(paths)
    print(f"  Pixel RDM shape: {pixel_rdm.shape}")

    partial_df = run_partial_rsa(pixel_rdm)
    print_summary(partial_df)

    print("\nPhase 4 v3 complete.")
    print(f"Outputs in: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
"""
stats_analysis_v3.py
=====================
Phase 3 (v3): Statistical Analysis + Subject-Level RSA

Changes from v2:
    1. Subject-level RSA analysis (2.1a): computes ρ per subject individually
       using per-subject fMRI RDMs from outputs_720/
    2. Subject-level plot: bar + individual dots per subject
    3. Subject-level summary CSV with mean ± SEM per rule × ROI
    4. All other v2 features retained (FDR, Cohen's d, forest plot)

Usage:
    py stats_analysis_v3.py

Prerequisites:
    - outputs_720/fmri_rdm_{roi}_{sub}.npy  (per-subject fMRI RDMs)
    - outputs/model_rdms/rdm_{rule}_{layer}.npy  (saved model RDMs)
    - outputs/rsa_results_cnn.csv  (group-level RSA results)

Outputs (in outputs/):
    permutation_results_fdr.csv     -- p-values with FDR correction
    permutation_forest_fdr.png      -- forest plot with FDR significance
    stats_summary_v2.csv            -- summary table for paper
    subject_rsa_results.csv         -- per-subject RSA scores
    subject_level_plot.png          -- bar + dots per subject per ROI
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, sem
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FMRI_DIR    = BASE_DIR / "outputs_720"
SUBJECTS    = ["sub-01", "sub-02", "sub-03"]
ROIS        = ["V1", "V2", "LOC", "IT"]
N_PERM      = 1000
SEED        = 42
rng         = np.random.default_rng(SEED)

COLORS = {
    "Random Weights":     "#999999",
    "Backprop":           "#2E86AB",
    "Feedback Alignment": "#E84855",
    "Predictive Coding":  "#3BB273",
    "STDP":               "#F4A261",
}

LAYER_MAP = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def upper_tri(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]

def rsa(rdm_a, rdm_b):
    n   = min(rdm_a.shape[0], rdm_b.shape[0])
    idx = np.triu_indices(n, k=1)
    r, _ = spearmanr(rdm_a[:n,:n][idx], rdm_b[:n,:n][idx])
    return float(r)

def load_fmri_rdm(roi, sub):
    p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
    return np.load(str(p)) if p.exists() else None

def mean_brain_rdm(roi):
    rdms = [load_fmri_rdm(roi, s) for s in SUBJECTS]
    rdms = [r for r in rdms if r is not None]
    if not rdms: return None
    n = min(r.shape[0] for r in rdms)
    return np.mean([r[:n,:n] for r in rdms], axis=0)


# ── FDR Correction ─────────────────────────────────────────────────────────────

def benjamini_hochberg(p_values):
    n = len(p_values)
    if n == 0: return []
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    adjusted   = np.zeros(n)
    cum_min    = 1.0
    for i in range(n-1, -1, -1):
        adj = sorted_p[i] * n / (i + 1)
        cum_min = min(cum_min, adj)
        adjusted[sorted_idx[i]] = min(cum_min, 1.0)
    return adjusted.tolist()


# ── Effect Size ────────────────────────────────────────────────────────────────

def cohens_d_from_subjects(rsa_df, rule_a, rule_b, roi, layer):
    sub_cols = ["rho_sub01", "rho_sub02", "rho_sub03"]

    def get_sub_vals(rule):
        row = rsa_df[(rsa_df["rule"]==rule) & (rsa_df["roi"]==roi)
                     & (rsa_df["layer"]==layer)]
        if row.empty: return []
        row = row.iloc[0]
        return [row[c] for c in sub_cols if c in row.index and not np.isnan(row[c])]

    a_vals = get_sub_vals(rule_a)
    b_vals = get_sub_vals(rule_b)
    if len(a_vals) < 2 or len(b_vals) < 2: return np.nan
    a, b = np.array(a_vals), np.array(b_vals)
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std < 1e-10: return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


# ── Permutation Test ───────────────────────────────────────────────────────────

def permutation_test(rdm_a, rdm_b, brain_rdm, n_perm=N_PERM):
    n   = min(rdm_a.shape[0], rdm_b.shape[0], brain_rdm.shape[0])
    idx = np.triu_indices(n, k=1)
    va  = rdm_a[:n,:n][idx]
    vb  = rdm_b[:n,:n][idx]
    vbr = brain_rdm[:n,:n][idx]

    r_a      = float(spearmanr(va, vbr)[0])
    r_b      = float(spearmanr(vb, vbr)[0])
    observed = r_a - r_b

    null = []
    for _ in range(n_perm):
        perm  = rng.permutation(len(vbr))
        vbr_p = vbr[perm]
        null.append(float(spearmanr(va, vbr_p)[0]) - float(spearmanr(vb, vbr_p)[0]))
    null  = np.array(null)
    p_val = np.mean(np.abs(null) >= np.abs(observed))
    return observed, p_val, null


def run_permutation_tests(model_rdms, rsa_df=None):
    results   = []
    all_p     = []
    rules     = list(model_rdms.keys())

    for roi in ROIS:
        brain = mean_brain_rdm(roi)
        if brain is None: continue
        layer = LAYER_MAP[roi]
        print(f"\n  ROI: {roi} (Layer: {layer})")

        for rule_a, rule_b in combinations(rules, 2):
            if layer not in model_rdms.get(rule_a, {}): continue
            if layer not in model_rdms.get(rule_b, {}): continue

            rdm_a = model_rdms[rule_a][layer]
            rdm_b = model_rdms[rule_b][layer]
            delta, p, _ = permutation_test(rdm_a, rdm_b, brain)

            d = np.nan
            if rsa_df is not None:
                d = cohens_d_from_subjects(rsa_df, rule_a, rule_b, roi, layer)

            results.append({
                "roi":           roi,
                "layer":         layer,
                "rule_a":        rule_a,
                "rule_b":        rule_b,
                "r_a":           rsa(rdm_a[:720,:720], brain),
                "r_b":           rsa(rdm_b[:720,:720], brain),
                "delta":         round(delta, 5),
                "p_uncorrected": round(p, 4),
                "cohens_d":      round(d, 3) if not np.isnan(d) else np.nan,
            })
            all_p.append(p)

            sig   = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            d_str = f"d={d:.2f}" if not np.isnan(d) else "d=N/A"
            print(f"    {rule_a:25s} vs {rule_b:25s}: "
                  f"Δ={delta:+.5f}  p={p:.4f} {sig}  {d_str}")

    fdr_adjusted = benjamini_hochberg(all_p) if all_p else []
    for i, row in enumerate(results):
        if i < len(fdr_adjusted):
            row["p_fdr"] = round(fdr_adjusted[i], 4)
            p_f = fdr_adjusted[i]
            row["sig_fdr"] = "***" if p_f<0.001 else "**" if p_f<0.01 else \
                             "*" if p_f<0.05 else "ns"
        else:
            row["p_fdr"]   = np.nan
            row["sig_fdr"] = "N/A"

    df = pd.DataFrame(results)
    if not df.empty:
        n_sig_raw = sum(1 for p in all_p if p < 0.05)
        n_sig_fdr = (df["sig_fdr"] != "ns").sum()
        print(f"\n  FDR correction: {n_sig_raw} → {n_sig_fdr} significant "
              f"(out of {len(all_p)} tests)")
    return df


# ── Load Model RDMs ───────────────────────────────────────────────────────────

def load_model_rdms():
    rdm_dir = OUTPUTS_DIR / "model_rdms"
    if not rdm_dir.exists():
        print("Model RDMs not found. Run learning_rules_v6.py first.")
        return None

    rules  = ["Random Weights", "Backprop", "Feedback Alignment",
              "Predictive Coding", "STDP"]
    layers = ["Conv1", "Conv2", "Conv3", "FC1"]

    model_rdms = {}
    for rule in rules:
        rule_key = rule.lower().replace(" ", "_")
        found = False
        for layer in layers:
            path = rdm_dir / f"rdm_{rule_key}_{layer}.npy"
            if path.exists():
                if rule not in model_rdms: model_rdms[rule] = {}
                model_rdms[rule][layer] = np.load(str(path))
                found = True
        if not found:
            print(f"  No RDMs found for: {rule}")

    return model_rdms if model_rdms else None


# ── Plots: Forest ──────────────────────────────────────────────────────────────

def plot_forest(df):
    rois = [r for r in ROIS if r in df["roi"].values]
    fig, axes = plt.subplots(1, len(rois), figsize=(4.5*len(rois), 6), sharey=True)
    if len(rois) == 1: axes = [axes]

    for ax, roi in zip(axes, rois):
        sub = df[df["roi"] == roi].copy()
        sub["label"] = sub["rule_a"].str[:4] + " vs " + sub["rule_b"].str[:4]
        sub = sub.sort_values("delta", ascending=True)

        colors = ["#2ecc71" if s != "ns" else "#e74c3c" for s in sub["sig_fdr"]]
        ax.barh(range(len(sub)), sub["delta"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["label"], fontsize=7)
        ax.set_xlabel("Δρ (A − B)", fontsize=10)
        ax.set_title(f"ROI: {roi}", fontsize=11)

        for i, (_, row) in enumerate(sub.iterrows()):
            if row["sig_fdr"] != "ns":
                ax.text(row["delta"], i, f" {row['sig_fdr']}", va="center",
                        fontsize=7, fontweight="bold")

    plt.suptitle("Pairwise Δρ with FDR correction (green = significant)", fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / "permutation_forest_fdr.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Summary Table ──────────────────────────────────────────────────────────────

def make_summary_table(rsa_df, perm_df):
    rules = ["Random Weights", "Backprop", "Feedback Alignment",
             "Predictive Coding", "STDP"]
    rules = [r for r in rules if r in rsa_df["rule"].values]

    rows = []
    for roi in ROIS:
        for rule in rules:
            layer = LAYER_MAP[roi]
            sub = rsa_df[(rsa_df["roi"] == roi) & (rsa_df["rule"] == rule)
                         & (rsa_df["layer"] == layer)]
            if sub.empty:
                sub = rsa_df[(rsa_df["roi"] == roi) & (rsa_df["rule"] == rule)]
            if sub.empty: continue
            best = sub.iloc[0]
            row  = {
                "ROI":   roi,
                "Rule":  rule,
                "Layer": best["layer"],
                "rho":   round(best["rho"], 4),
                "CI_lo": round(best.get("ci_lo", best["rho"]), 4),
                "CI_hi": round(best.get("ci_hi", best["rho"]), 4),
            }

            if rule != "Backprop" and perm_df is not None:
                p_sub = perm_df[
                    (perm_df["roi"] == roi) &
                    (((perm_df["rule_a"]=="Backprop") & (perm_df["rule_b"]==rule)) |
                     ((perm_df["rule_a"]==rule) & (perm_df["rule_b"]=="Backprop")))
                ]
                if not p_sub.empty:
                    row["p_vs_BP_uncorr"] = round(p_sub.iloc[0]["p_uncorrected"], 4)
                    row["p_vs_BP_fdr"]    = round(p_sub.iloc[0]["p_fdr"], 4)
                    row["sig_vs_BP"]      = p_sub.iloc[0]["sig_fdr"]
                    row["d_vs_BP"]        = p_sub.iloc[0].get("cohens_d", np.nan)
            else:
                row["p_vs_BP_uncorr"] = "-"
                row["p_vs_BP_fdr"]    = "-"
                row["sig_vs_BP"]      = "-"
                row["d_vs_BP"]        = "-"

            rows.append(row)

    summary = pd.DataFrame(rows)
    path    = OUTPUTS_DIR / "stats_summary_v2.csv"
    summary.to_csv(str(path), index=False)
    print(f"\nSummary saved: {path.name}")
    print(summary.to_string())
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT-LEVEL ANALYSIS (2.1a)
# ══════════════════════════════════════════════════════════════════════════════

def compute_subject_rsa(model_rdms):
    """
    Compute RSA separately for each subject, using per-subject fMRI RDMs.
    Returns DataFrame with columns: rule, roi, layer, subject, rho
    """
    rules = list(model_rdms.keys())
    rows  = []

    for roi in ROIS:
        layer = LAYER_MAP[roi]
        for rule in rules:
            if layer not in model_rdms.get(rule, {}):
                continue
            model_rdm = model_rdms[rule][layer]

            for sub in SUBJECTS:
                brain_rdm = load_fmri_rdm(roi, sub)
                if brain_rdm is None:
                    continue
                r = rsa(model_rdm, brain_rdm)
                rows.append({
                    "rule":    rule,
                    "roi":     roi,
                    "layer":   layer,
                    "subject": sub,
                    "rho":     round(r, 5),
                })
                print(f"  {rule:25s}  {roi}  {sub}: ρ = {r:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(str(OUTPUTS_DIR / "subject_rsa_results.csv"), index=False)
    print(f"\n  Saved: subject_rsa_results.csv  ({len(df)} rows)")
    return df


def plot_subject_level(sub_df):
    """
    Bar chart (mean across subjects) with individual subject dots overlaid.
    One subplot per ROI, bars grouped by rule.
    """
    rules = ["Random Weights", "Backprop", "Feedback Alignment",
             "Predictive Coding", "STDP"]
    rules = [r for r in rules if r in sub_df["rule"].values]
    rois  = [r for r in ROIS if r in sub_df["roi"].values]

    if not rois or not rules:
        print("  No data for subject-level plot.")
        return

    n_rules = len(rules)
    x       = np.arange(len(rois))
    w       = 0.7 / n_rules
    offsets = np.linspace(-(n_rules-1)*w/2, (n_rules-1)*w/2, n_rules)

    _, ax = plt.subplots(figsize=(10, 5))

    for i, rule in enumerate(rules):
        color = COLORS.get(rule, "gray")
        means, sems = [], []

        for roi in rois:
            vals = sub_df[(sub_df["rule"]==rule) & (sub_df["roi"]==roi)]["rho"].values
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            sems.append(sem(vals) if len(vals) > 1 else 0.0)

        ax.bar(x + offsets[i], means, w * 0.9,
               color=color, alpha=0.75, label=rule, zorder=2)
        ax.errorbar(x + offsets[i], means, yerr=sems,
                    fmt="none", color="black", capsize=3, lw=1, zorder=3)

        # Individual subject dots
        for j, roi in enumerate(rois):
            vals = sub_df[(sub_df["rule"]==rule) & (sub_df["roi"]==roi)]["rho"].values
            jitter = np.linspace(-w*0.25, w*0.25, len(vals)) if len(vals) > 1 else [0]
            for k, v in enumerate(vals):
                ax.scatter(x[j] + offsets[i] + jitter[k], v,
                           color="white", edgecolors=color,
                           s=30, zorder=4, linewidths=1.2)

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(rois, fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title("Subject-level RSA (bars = mean ± SEM, dots = individual subjects)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = OUTPUTS_DIR / "subject_level_plot.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def print_subject_summary(sub_df):
    """Print mean ± SEM per rule × ROI for quick inspection."""
    rules = ["Random Weights", "Backprop", "Feedback Alignment",
             "Predictive Coding", "STDP"]
    rules = [r for r in rules if r in sub_df["rule"].values]

    print("\n  Subject-level RSA summary (mean ± SEM):")
    print(f"  {'Rule':25s}  {'ROI':5s}  mean ± SEM")
    print("  " + "-"*50)
    for roi in ROIS:
        for rule in rules:
            vals = sub_df[(sub_df["rule"]==rule) & (sub_df["roi"]==roi)]["rho"].values
            if len(vals) == 0:
                continue
            m = vals.mean()
            s = sem(vals) if len(vals) > 1 else 0.0
            print(f"  {rule:25s}  {roi:5s}  {m:.4f} ± {s:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER-BY-LAYER HEATMAP (3.2)
# ══════════════════════════════════════════════════════════════════════════════

LAYERS = ["Conv1", "Conv2", "Conv3", "FC1"]
RULES  = ["Random Weights", "Backprop", "Feedback Alignment",
          "Predictive Coding", "STDP"]


def plot_layer_heatmap(model_rdms):
    """
    Heatmap: rows = learning rules, columns = layers, color = Spearman ρ vs brain.
    One subplot per ROI. Annotates each cell with the ρ value.
    Saves layer_heatmap.png and layer_heatmap.csv.
    """
    rules  = [r for r in RULES  if r in model_rdms]
    layers = [l for l in LAYERS if any(l in model_rdms[r] for r in rules)]

    if not rules or not layers:
        print("  No data for layer heatmap.")
        return

    # Build ρ matrix: shape (n_rules, n_layers) per ROI
    data = {}   # roi -> 2-D array
    rows_csv = []
    for roi in ROIS:
        brain = mean_brain_rdm(roi)
        if brain is None:
            continue
        mat = np.full((len(rules), len(layers)), np.nan)
        for ri, rule in enumerate(rules):
            for li, layer in enumerate(layers):
                if layer in model_rdms.get(rule, {}):
                    mat[ri, li] = rsa(model_rdms[rule][layer], brain)
                    rows_csv.append({"roi": roi, "rule": rule,
                                     "layer": layer, "rho": round(mat[ri, li], 4)})
        data[roi] = mat

    if not data:
        print("  No brain RDMs found — heatmap skipped.")
        return

    pd.DataFrame(rows_csv).to_csv(
        str(OUTPUTS_DIR / "layer_heatmap.csv"), index=False)

    # Plot
    rois_plot = list(data.keys())
    vmin = min(np.nanmin(m) for m in data.values())
    vmax = max(np.nanmax(m) for m in data.values())
    # Symmetric colormap around 0
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    _, axes = plt.subplots(1, len(rois_plot),
                           figsize=(3.2 * len(rois_plot), 2.8 * len(rules) / 5 + 1.5))
    if len(rois_plot) == 1:
        axes = [axes]

    for ax, roi in zip(axes, rois_plot):
        mat = data[roi]
        im  = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=vmin, vmax=vmax, interpolation="nearest")

        # Annotate cells
        for ri in range(len(rules)):
            for li in range(len(layers)):
                val = mat[ri, li]
                if not np.isnan(val):
                    txt_color = "white" if abs(val) > abs_max * 0.6 else "black"
                    ax.text(li, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=txt_color)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=9)
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(rules if ax is axes[0] else [""] * len(rules), fontsize=8)
        ax.set_title(f"ROI: {roi}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")

    plt.suptitle("Layer-by-layer RSA: Spearman ρ (model layer vs. brain ROI)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = OUTPUTS_DIR / "layer_heatmap.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: layer_heatmap.png, layer_heatmap.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Stats Analysis v3 (FDR + effect sizes + subject-level)\n")

    # Load RSA results
    rsa_csv = OUTPUTS_DIR / "rsa_results_cnn.csv"
    if not rsa_csv.exists():
        print(f"Not found: {rsa_csv}")
        print("Run learning_rules_v6.py first.")
        return
    rsa_df = pd.read_csv(str(rsa_csv))
    print(f"RSA results loaded: {len(rsa_df)} rows\n")

    # Load model RDMs
    print("Loading model RDMs...")
    model_rdms = load_model_rdms()
    if model_rdms is None:
        print("  Model RDMs missing — skipping permutation tests and subject-level analysis.")
        make_summary_table(rsa_df, None)
        return

    # Permutation tests with FDR
    print(f"\nPermutation tests (N={N_PERM}) with FDR correction...")
    perm_df = run_permutation_tests(model_rdms, rsa_df)
    perm_df.to_csv(str(OUTPUTS_DIR / "permutation_results_fdr.csv"), index=False)

    if not perm_df.empty:
        plot_forest(perm_df)

    make_summary_table(rsa_df, perm_df)

    # Layer-by-layer heatmap (3.2)
    print("\n" + "="*55)
    print("LAYER-BY-LAYER HEATMAP (3.2)")
    print("="*55)
    plot_layer_heatmap(model_rdms)

    # Subject-level analysis (2.1a)
    print("\n" + "="*55)
    print("SUBJECT-LEVEL ANALYSIS (2.1a)")
    print("="*55)
    sub_df = compute_subject_rsa(model_rdms)
    if not sub_df.empty:
        print_subject_summary(sub_df)
        plot_subject_level(sub_df)

    print("\nDone.")


if __name__ == "__main__":
    main()

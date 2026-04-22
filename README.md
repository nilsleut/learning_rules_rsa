# Learning Rules RSA: Comparing Backpropagation, Feedback Alignment, Predictive Coding, and STDP Against Human fMRI

Code and results for the paper:

> **Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI**  
> Nils Leutenegger, Independent Researcher, Switzerland  
---

## Overview

This repository provides a systematic comparison of four learning rules — Backpropagation (BP), Feedback Alignment (FA), Predictive Coding (PC), and Spike-Timing-Dependent Plasticity (STDP) — applied to identical CNN architectures and evaluated against human fMRI data from the [THINGS-fMRI dataset](https://osf.io/jum2f/) using Representational Similarity Analysis (RSA).

**Key findings:**
- V1/V2 alignment is primarily **architecture-driven**: an untrained CNN (ρ = 0.071) is statistically indistinguishable from BP (ρ = 0.072, p = 0.43)
- Learning rules only differentiate at **higher visual areas** (LOC/IT): BP dominates (ρ = 0.018–0.020, d > 2.3 vs. random)
- **PC matches BP at IT** (ρ = 0.017 vs. 0.020, p = 0.18) using only local Hebbian updates
- **FA actively degrades** representations below the random baseline at V1 (d = 1.1)

---
## Paper

📄 **[arXiv:2604.16875](https://arxiv.org/abs/2604.16875)** — *Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI*

## Setup

```bash
git clone https://github.com/nilsleut/learning-rules-rsa
cd learning-rules-rsa
pip install -r requirements.txt
```

**Requirements:** Python 3.11, PyTorch, snnTorch, scipy, numpy, pandas, matplotlib, scikit-learn

---

## Data

This project uses the [THINGS-fMRI dataset](https://osf.io/jum2f/) (Gifford & Cichy, 2022). Download the fMRI RDMs for subjects 1–3 (V1, V2, LOC, IT) and place them in `data/things_fmri/`.

CIFAR-10 is loaded automatically via `torchvision`.

---

---

## Main Results

| ROI | Layer | Random | BP | FA | PC | STDP |
|-----|-------|--------|----|----|----|----|
| V1  | Conv1 | 0.071 | **0.072** | 0.030 | 0.057 | **0.079** |
| V2  | Conv1 | 0.039 | **0.050** | 0.014 | 0.030 | 0.046 |
| LOC | Conv3 | −0.004 | **0.018** | 0.004 | 0.005 | 0.002 |
| IT  | FC1   | 0.004 | **0.020** | 0.008 | 0.017 | 0.009 |

Spearman ρ (model–brain alignment). Bold = best per ROI. Full statistics in `results/`.

---

## Acknowledgements

Thanks to the creators of the THINGS-fMRI dataset for making their data publicly available, and to the Brain-Score team for their evaluation infrastructure.

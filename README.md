# Untrained CNNs Match Backpropagation at V1

Code, figures, and results for the paper **"Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI"**.

[
[
[
[

> This repository tracks the **latest** project state and currently corresponds to **arXiv v2**. Earlier repository states should be preserved via Git tags or GitHub releases, since releases are the standard mechanism for distributing stable project snapshots on GitHub.[1][2]

## Overview

This repository accompanies a study of whether the learning rule used to train a neural network determines how well its internal representations align with the human visual cortex. The paper compares backpropagation (BP), feedback alignment (FA), predictive coding (PC), spike-timing-dependent plasticity (STDP), and an untrained random-weights baseline using representational similarity analysis (RSA) against THINGS-fMRI.

## Main result

The central result is that early visual alignment is driven primarily by **architecture** rather than the specific learning rule. In V1 and V2, the untrained CNN exceeds backpropagation, while only at LOC does BP show a reliable advantage over the random baseline.

## Key findings

- Random weights exceed BP at V1: rho = 0.076 vs. 0.034.
- STDP achieves the highest V1 alignment among trained rules: rho = 0.064.
- BP is the only condition that reliably exceeds the random baseline at LOC.
- At IT, all five conditions converge and no trained-rule comparison survives FDR correction.
- Partial RSA preserves the main ordering after controlling for pixel similarity.

## Repository structure

```text
figures/     Final manuscript figures corresponding to arXiv v2
paper/       Paper source files and compiled manuscript
programs/    Training, RSA, statistics, and plotting code
results/     Processed outputs, summary tables, and saved results
```



A practical release layout is:

- `v1` — repository state corresponding to the first arXiv submission.[2]
- `v2` — repository state corresponding to the revised arXiv version now online.[2]
- `main` — actively maintained latest project state.[1]

## Reproducibility

This repository contains the code, figures, and processed results used for the current manuscript version. Full end-to-end reproduction may require external datasets, computational resources, and local environment configuration beyond what is redistributed here, which is normal for research repositories handling external benchmark data.[3][4]

A good reproduction workflow is:

1. Set up the Python environment and dependencies.
2. Place required datasets into the expected directory structure.
3. Run the training and feature-extraction scripts in `programs/`.
4. Regenerate result tables in `results/`.
5. Regenerate final manuscript figures in `figures/`.
6. Verify that the outputs match the current arXiv v2 manuscript.

## Recommended maintenance after v2

Now that the repository files have been updated from v1 to v2, the most important maintenance step is consistency.[5][3] The figures, programs, paper files, and results should all describe the **same** scientific state, and the README should explicitly state that the repository corresponds to arXiv v2 so visitors do not confuse it with the earlier version.[5][3]

A concise repository note can therefore be kept near the top:

> This repository currently corresponds to the revised arXiv v2 version of the paper. For the original repository state associated with the first submission, see the `v1` release.

## Citation

If this repository is useful in your work, cite the paper as:

```bibtex
@article{leutenegger2026untrained,
  title={Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI},
  author={Leutenegger, Nils},
  journal={arXiv preprint arXiv:2604.16875},
  year={2026}
}
```

## License

Add a repository license file if not already present. GitHub recommends explicit repository metadata such as licensing because it improves clarity for users and reusers of the codebase.[5]

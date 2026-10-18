# Synthetic Atrophy Mapping: MLP Patches vs. LightGBM with Multi-Scale Change Features

Motivation: Longitudinal brain atrophy mapping is usually done by registering scan pairs and computing voxel-wise change, which is slow. "Efficient Atrophy Mapping: A Single-Step U-Net Approach for Rapid Brain Change Estimation" (2024) proposes producing atrophy maps directly from paired scans in a single network pass. This repository asks whether a cheaper, cross-validated gradient-boosting model with smoothed multi-scale difference features can match or beat a single-step learned mapping on a controlled simulation.

Question: Can a cross-validated LightGBM model with smoothed multi-scale difference features estimate voxel-wise atrophy maps from paired scans more accurately than a single-step MLP regressor on patches, in a simulated brain-slice setting?

Dataset: synthetic 64x64 brain-like slice pairs generated in code with numpy, with a known atrophy map from simulated ventricle enlargement and cortical shrinkage plus noise. The data are not the paper's dataset (`same_as_paper` is false), and no public MRI pair was used.

Baseline: an sklearn MLPRegressor on flattened 7x7 patches from both time points, predicting the centre-pixel atrophy value. It stands in for the paper's single-step network and is not the paper's U-Net.

Improved model: a LightGBM regressor with hyperparameters chosen by group cross-validation on the training subjects, using the same patches plus Gaussian-smoothed difference and multi-scale intensity-change features.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (and `figures/pearson_bar.png`).

## Results (latest run)

On a synthetic 2D brain-slice dataset, the standard single-step MLP patch regressor reached a held-out pixel Pearson r of 0.9967, while the cross-validated LightGBM model with smoothed multi-scale difference features reached 0.9986, a small gain of 0.0019 on the same test set.

| Metric | Value |
|---|---|
| baseline_score | 0.9967 |
| improved_score | 0.9986 |
| improvement | 0.001899 |
| baseline_score_std | 0.0001383 |
| improved_score_std | 0.0000237 |
| baseline_mae | 0.009772 |
| improved_mae | 0.004775 |
| baseline_volume_abs_error | 5.344 |
| improved_volume_abs_error | 3.257 |
| lgb_cv_pearson | 0.998 |
| lgb_num_leaves | 63 |
| lgb_min_child_samples | 100 |

Run on 2026-10-18. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic data only, not the paper's dataset; the paper's single-step U-Net is not reproduced, and the MLP is a stand-in. Hyperparameters were selected by CV on training subjects (seed 42) only; the test set was used once per model and seed. Model seeds 42-44 share one simulated dataset and one split, so the std reflects model randomness only. Volume error is in pixel units per scan. Results may not transfer to real MRI.

## Reference

This project is motivated by the following paper:

> R Raciti, A Rondinella, L Puglisi (2024). *Efficient Atrophy Mapping: A Single-Step U-Net Approach for Rapid Brain Change Estimation*. … on Metrology for. https://ieeexplore.ieee.org/abstract/document/10796213/

Paper author: Prof. Sebastiano Battiato, University of Catania (battiato@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

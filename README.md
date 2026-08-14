# CI/CD Pipeline Failure Prediction

Final year research project, BSc Software Engineering, NSBM Green University.
Author: Tharuka H. Dilshan

## Overview
Machine learning system predicting CI/CD build failures before execution,
enabling automated deployment gating.

## Dataset
TravisTorrent (8 Feb 2017 release) — 677,863 builds across 1,283 GitHub projects.
Source: https://travistorrent.testroots.org/
Not included in this repo due to size (184 MB compressed).

## Structure
- `notebooks/01_data_preparation.ipynb` — cleaning, leakage audit, feature engineering
- `notebooks/02_modelling_evaluation.ipynb` — model training and evaluation
- `docs/` — methodology documentation
- `artefact/` — prediction service (in progress)

## Status
Data preparation complete. Modelling in progress.

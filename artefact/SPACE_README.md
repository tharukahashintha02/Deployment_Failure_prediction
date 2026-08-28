---
title: CI/CD Failure Prediction Service
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CI/CD Pipeline Failure Prediction Service

Predicts the probability that a CI/CD build will fail **before it runs**, returns a
SHAP explanation of the prediction, and enforces a configurable gating policy.

Research artefact for a BSc Software Engineering final year project,
NSBM Green University.

## Model

XGBoost trained on TravisTorrent: 677,863 builds across 1,283 open-source projects.

| Metric | Value |
|---|---|
| ROC-AUC | 0.881 |
| PR-AUC | 0.787 |
| Precision @ threshold 0.80 | 0.855 |
| Features | 46 |

## API

Interactive documentation is at [`/docs`](./docs).

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predict` | Score a build before execution |
| POST | `/outcome` | Report a completed build result |
| GET | `/health` | Model and database status |
| GET | `/model-info` | Version, threshold, training metrics |

## Why it is stateful

The model's strongest features describe a project's recent build history, and
together they carry roughly 86% of its predictive power. A CI runner calling
`/predict` knows only about the current commit, so the service maintains the
history itself and the pipeline reports outcomes back via `/outcome`.

Without a persistent database every project resets to cold start on restart, and
predictions silently fall back to code features alone (~0.64 AUC instead of ~0.88).
Set `DATABASE_URL` to a Postgres connection string in the Space secrets.

## Configuration

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string. **Set this as a Space secret.** |
| `POLICY_MODE` | `shadow` (log only), `warn`, or `block` |

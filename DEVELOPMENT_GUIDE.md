# Deployment Failure Prediction System — Development Guide

**Project:** ML-Based Failure Prediction System for CI/CD Pipelines
**Author:** Tharuka H. Dilshan — BSc Software Engineering, NSBM Green University
**Dataset:** TravisTorrent (8 Feb 2017 release)
**Environment:** Google Colab + Google Drive

---

## Part 0 — Scope decision and required proposal amendments

You've decided to proceed with TravisTorrent and accept the loss of infrastructure
telemetry. That's a sound call — real data with genuine signal beats a synthetic file
with none. But you must **amend the proposal to match**, or you will be marked against
promises you can't keep. Do this now, in writing, with your supervisor.

### What changes

| Proposal section | Current text | Amend to |
|---|---|---|
| Title / §1.3 | "Deployment Failure Prediction" | **"CI/CD Pipeline Failure Prediction"** — build failure is your operational label |
| §1.3 Secondary Obj. 1 | "GitHub Actions and Kubernetes... resource utilisation" | "TravisTorrent (n=677,863 builds, 1,283 projects); 45 engineered features across code-change, pipeline-history, and temporal groups" |
| §1.3 Primary Obj. | "≥85% precision" | Keep, but see the precision/F1 conflict below |
| §3.2 Q3 | "temporal and anomaly-detection on telemetry" | "temporal patterns in **pipeline history** and anomaly detection on build feature vectors" |
| §4.3 | Generic data collection | Name TravisTorrent explicitly; cite Beller, Gousios & Zaidman (MSR 2017) |
| §4.4 | Telemetry aggregation windows | Replace with pipeline-history rolling windows (last 5 / last 20 builds) |
| §4.8 | — | **Add:** no infrastructure telemetry available; dataset era 2011–2016; language skew (87.6% Ruby, 12.3% Java); project concentration (top 2 projects = 20% of data) |

### The honest framing for your defence

Don't present the missing telemetry as a shortfall. Present it as a **deliberate scope
decision with a measured justification**: you evaluated a synthetic alternative
containing telemetry columns, found it had zero predictive signal (ROC-AUC 0.50 on all
candidate targets) and no success class at all, and chose real observed data over
fabricated completeness. Then list telemetry integration as your primary future-work item.

That is a stronger position than most FYPs manage, because it shows you validated your
data rather than assuming it.

### One target conflict you must resolve now

Your proposal promises **precision ≥0.85 AND F1 ≥0.80 simultaneously.** I measured this
on your data. It is not achievable — not because your model is bad, but because the two
constraints are in tension at this signal level:

| Threshold | Precision | Recall | F1 | % builds blocked |
|---|---|---|---|---|
| 0.50 | 0.625 | 0.763 | 0.688 | 31.8% |
| 0.60 | 0.694 | 0.695 | **0.694** ← max F1 | 26.1% |
| 0.70 | 0.767 | 0.615 | 0.683 | 20.9% |
| 0.80 | 0.843 | 0.517 | 0.641 | 16.0% |
| **0.85** | **0.882** ← hits precision target | 0.454 | 0.599 | 13.4% |
| 0.95 | 0.964 | 0.283 | 0.438 | 7.7% |

You can hit precision ≥0.85 (at threshold ~0.82). You can hit max F1 ≈ 0.69. You cannot
have both at once.

**Recommended fix:** amend §1.3 to *"achieve ≥0.85 precision for the failure class at an
operationally-selected threshold, with the full precision–recall trade-off reported."*
Then present the table above as a core result. A supervisor will respect a student who
measured the trade-off and reported it honestly far more than one who quietly reports
only the number that looks good.

---

## Part 1 — What's actually in your data (measured)

| Property | Value |
|---|---|
| Raw rows (job-level) | 3,702,595 |
| Unique builds | 680,209 |
| After dropping canceled/started | **677,863** |
| Projects | 1,283 |
| Date range | 2011-08 → 2016-12 |
| Languages | Ruby 87.6%, Java 12.3%, JavaScript <0.1% |
| **Build-level failure rate** | **26.79%** |
| Failed / errored / passed | 123,149 / 58,468 / 496,246 |
| Concentration | `rails/rails` 12.7%, `jruby/jruby` 7.3% of rows |

Note the failure rate is **26.8% at build level**, not the 33% you'd get from raw job
rows. Each build fans out into ~5.4 jobs, so job-level counting distorts the rate.
Collapse to builds first.

### Benchmark performance to expect

I trained the full pipeline end to end on your data. Temporal 70/15/15 split,
45 features, no leakage:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Always-predict-pass | — | — | — | 0.000 | 0.000 |
| Logistic Regression | 0.8751 | 0.7774 | 0.647 | 0.729 | 0.686 |
| Random Forest | 0.8780 | 0.7753 | 0.668 | 0.708 | 0.688 |
| **XGBoost** | **0.8812** | **0.7873** | 0.623 | 0.766 | 0.687 |

(Precision/recall/F1 at threshold 0.50, untuned.)

**Use these as your sanity benchmark.** If your numbers land near these, your pipeline is
correct. If you get >0.95 AUC, you have leakage — go back to the column reference. If you
get <0.70 AUC, your history features are probably computed wrong.

Note the models are close together. That itself is a finding: on this feature set most of
the signal is accessible to a linear model, and XGBoost's advantage is modest. Report it.
Overclaiming a large XGBoost win when confidence intervals overlap is exactly the kind of
thing that gets challenged in a viva.

### The ablation that anchors your RQ1 answer

| Feature set | ROC-AUC | PR-AUC |
|---|---|---|
| Without pipeline-history features | 0.6442 | 0.3836 |
| With pipeline-history features | **0.8810** | **0.7874** |

History features are worth **+0.24 AUC**. Code-change metrics alone are weak. This is
your most important empirical result — lead your results chapter with it.

---

## Part 2 — Environment setup (Week 1)

### Drive layout

```
MyDrive/fyp-deployment-prediction/
├── data/
│   ├── raw/travistorrent_8_2_2017_csv.gz
│   ├── interim/builds.parquet
│   └── processed/{train,val,test}.parquet
├── models/
├── notebooks/
├── reports/figures/
└── artefact/
```

### Colab practicalities

- **CPU runtime is fine.** XGBoost on 678k rows trains in a few minutes. Only switch to
  GPU for the optional LSTM.
- **Sessions die.** Free Colab disconnects after ~12h and idles out at 90 min. Every
  notebook must *end* by writing to Drive and *start* by reading from Drive. Never hold
  state only in RAM.
- **Memory:** ~12.7 GB on the free tier. The notebook reads in chunks and deduplicates as
  it goes, keeping peak usage low. Don't "optimise" this by loading the whole CSV at once.
- **Parquet, not CSV** for intermediates — ~10× faster reload, preserves dtypes.
- **Colab Pro** (~$10/mo) becomes worth it during hyperparameter search. Not before.

### Version control

`File → Save a copy in GitHub` from day one. Examiners ask for a repo. `.gitignore` the
`data/` directory.

---

## Part 3 — Data preparation (Notebook 01)

### 3.1 Collapse jobs to builds

Deduplicate on `tr_build_id`, keeping the first row. This is safe: I verified `tr_status`
is constant within a build (0 conflicts across 85,129 sampled builds).

Skipping this is fatal — you'd have ~5 near-identical rows per build, and any split would
put copies of the same build in both train and test.

### 3.2 Label

```
tr_status ∈ {failed, errored}   → failed = 1
tr_status == passed             → failed = 0
tr_status ∈ {canceled, started} → drop (2,346 rows)
```

Justify merging failed+errored in your write-up: both block the pipeline, so from a
gating standpoint they're equivalent. Note that separating them is a multi-class
extension left to future work.

### 3.3 Column selection

See `COLUMN_REFERENCE.md` for the full keep/drop table with reasons. Short version: drop
all 19 `tr_log_*` / duration / merge columns (leakage), drop 13 identifier and hash
columns, keep 24 raw columns, engineer 21 more.

### 3.4 The `.shift(1)` rule

Every history feature must be computed **per project, sorted by time, shifted by one
before any rolling window.** The notebook does this correctly. Read that cell carefully
and be able to explain it aloud — it's the most likely thing you'll be asked to justify
in a viva, and getting it wrong invalidates every result silently.

### 3.5 Splits

**Scheme A — temporal (primary).** Sort by `gh_build_started_at`, split 70/15/15 by time.
Train on the past, test on the future. Measured split failure rates: train 27.9%,
val 22.3%, test 26.1%. That variation is real concept drift — discuss it, don't hide it.

**Scheme B — grouped by project (secondary).** `GroupKFold` on `gh_project_name`, so no
test project appears in training. Answers "does the model transfer to a project it has
never seen?" — the practical question for a reusable gating tool. Expect noticeably worse
results; that gap deserves its own section.

**Never use a random split.**

---

## Part 4 — Modelling (Notebook 02)

### Order of work

1. **Majority-class baseline** — always predict "pass," accuracy 73.9%. One paragraph,
   but it's how you demonstrate that accuracy is the wrong headline metric.
2. **Logistic Regression** — `class_weight='balanced'`, log1p + standard scaling.
3. **Random Forest** — `class_weight='balanced_subsample'`, gives permutation importance.
4. **XGBoost** — main model. `scale_pos_weight`, `eval_metric='aucpr'`, early stopping on
   the validation set.
5. **LightGBM** — optional, faster, usually comparable. Worth one row in your table.

### Class imbalance

At 26.8% failures this is **mild imbalance**. Do not reflexively reach for SMOTE.

Priority order:
1. `scale_pos_weight` / `class_weight` — free, effective.
2. **Threshold tuning** — where the real control is. Train once, sweep the threshold on
   *validation*, report the full PR curve.
3. SMOTE — only as an experimental arm. If used, apply it **inside** the CV fold on
   training data only, never before splitting. Report it even if it doesn't help; a
   negative result is legitimate and your proposal named the technique.

### Hyperparameter tuning

`RandomizedSearchCV` with `TimeSeriesSplit`, scoring `average_precision`. 40–60 iterations
is plenty. Ranges: `max_depth` 3–10, `learning_rate` 0.01–0.3 (log-uniform),
`n_estimators` 200–1500 with early stopping, `subsample` 0.6–1.0, `colsample_bytree`
0.6–1.0, `min_child_weight` 1–10, `reg_lambda` 0–10.

Realistically expect tuning to buy ~0.005–0.015 AUC over sensible defaults. Say so rather
than implying it was transformative.

### Q3: temporal and anomaly components

- **Isolation Forest** — fit on *passing* builds only, feed the anomaly score into
  XGBoost as an extra feature. Clean ablation: with vs. without.
- **LSTM** — sequence of the last k=10 builds per project. Be prepared for it to *not*
  beat XGBoost-with-lag-features. On tabular data with well-engineered lags that's the
  common outcome, and reporting it straight is a better result than torturing the model
  until it wins.

### Explainability (§4.7 requires this)

SHAP `TreeExplainer` on XGBoost. Beeswarm plot for global importance (your RQ1 answer —
use this rather than raw gain, which is cardinality-biased), force plots for individual
flagged builds (this becomes the "why was this flagged?" panel in your dashboard).
Compute on a 5,000-row sample; full-dataset SHAP is slow.

---

## Part 5 — Evaluation

### Metrics

For the **failure class** on the temporal test set: Precision, Recall, F1, ROC-AUC,
**PR-AUC (average precision)**, MCC. Lead with PR-AUC — under imbalance, ROC-AUC is
optimistic.

Include: confusion matrix at your operating threshold, PR and ROC curves with all models
overlaid, and the threshold sweep table from Part 0.

### Operational simulation — your differentiator

Most student projects stop at a metrics table. Replay the test set chronologically as if
the gate were live and report:

- **Caught failures** (TP) — failures that would have been blocked.
- **False blocks** (FP) — expressed as "1 in every N healthy builds delayed." This is
  developer friction, and it's the number a real DevOps lead cares about.
- **Escaped failures** (FN).
- **Failure rate before vs. after gating.**
- **Threshold sensitivity table** — the most quotable table in your dissertation.

Optional but strong: a cost model. Assume a failed build costs X engineer-minutes and a
false block costs Y, plot total cost vs. threshold, find the optimum. That grounds your
§1.2 motivation in something concrete.

### Statistical rigour

Bootstrap 1,000 test-set resamples for 95% CIs on F1 and PR-AUC. Given how close your
models are, check whether intervals overlap before claiming XGBoost "beats" Random Forest.
Use McNemar's test for paired comparison. Cheap to do, and it will lift your marks.

---

## Part 6 — The artefact (Aug–Sep)

### Serving

**FastAPI**, not Flask — automatic OpenAPI docs, Pydantic validation.

```
artefact/
├── app/
│   ├── main.py        # POST /predict, GET /health, GET /model-info
│   ├── schemas.py     # Pydantic models
│   ├── features.py    # SAME module imported by the training notebook
│   └── model/         # pipeline.joblib, feature_names.json, threshold.json
├── Dockerfile
└── tests/
```

**Critical rule:** the feature transformation used at inference must be the *same code
object* used in training. Pickle the whole `sklearn.Pipeline`, not just the classifier.
Training/serving skew is a real bug class and it fails silently.

Response shape:
```json
{
  "risk_score": 0.83,
  "decision": "HIGH_RISK",
  "threshold": 0.82,
  "top_factors": [
    {"feature": "prev_build_failed", "contribution": 0.31},
    {"feature": "consecutive_prior_failures", "contribution": 0.12}
  ],
  "model_version": "xgb-v1.0",
  "latency_ms": 14
}
```

`top_factors` from SHAP satisfies your explainability commitment inside the artefact, not
just in the analysis. Measure and report p50/p95 latency (§4.6 asks for it) — you'll
comfortably beat 100ms.

### CI/CD integration

Deploy free on Render / Railway / Fly.io / HF Spaces, then add a GitHub Actions job that
calls `/predict` before the deploy step. Implement **three policy modes** — shadow (log
only, never block), warn, and block. Demonstrate all three; shadow mode is what your §4.7
ethics section commits you to starting with.

### Dashboard

Streamlit. You'll have something working in a day rather than a week on React. Pages: risk
score for a submitted build, prediction history vs. actual outcomes, SHAP feature
importance, threshold explorer. Free hosting on Streamlit Community Cloud.

---

## Part 7 — Revised timeline

Your original plan doesn't start modelling until June. You have clean data now, so pull
everything forward and bank the slack — something always breaks.

| Period | Work |
|---|---|
| **Now – Feb** | Notebook 01: data prep + EDA. Supervisor sign-off on the scope amendment. |
| **Mar – Apr** | Notebook 02: baselines + XGBoost + threshold analysis. **Main results 3 months early.** |
| **May – Jun** | Tuning, Isolation Forest + LSTM ablations, SHAP, cross-project split. |
| **Jul – Aug** | FastAPI service, Docker, tests. |
| **Sep** | GitHub Actions integration, three policy modes, Streamlit dashboard. |
| **Oct** | Operational simulation, bootstrap CIs, ablation write-up. |
| **Nov** | Final dissertation, IEEE formatting, demo video, submission. |

---

## Part 8 — Failure modes to avoid

1. **Leakage via `tr_log_*`.** >0.95 AUC means you leaked. Benchmark is 0.88.
2. **Random train/test split.** Duplicate jobs plus project memorisation inflate everything.
3. **Rolling features without `.shift(1)`.** Catastrophic and invisible in the metrics.
4. **SMOTE before splitting.** Synthetic test rows derived from training rows.
5. **Accuracy as the headline.** 73.9% is achievable by predicting "pass" every time.
6. **Training/serving skew.** Different preprocessing in the API than in the notebook.
7. **Not saving to Drive.** Colab will disconnect at 2am and you will lose six hours.
8. **Claiming both precision ≥0.85 and F1 ≥0.80.** Unreachable here — amend the objective
   before you write results.

---

## Part 9 — References to add

Your review is thin on the dataset and the specific task. Add:

- **Beller, M., Gousios, A., & Zaidman, A. (2017).** *TravisTorrent: Synthesizing Travis
  CI and GitHub for Full-Stack Research on Continuous Integration.* MSR 2017. —
  Mandatory, it's your dataset paper.
- **Hassan, F., & Wang, X. (2017).** *Change-Aware Build Prediction Model for Stall
  Avoidance in Continuous Integration.* ESEM.
- **Ni, A., & Li, M. (2017).** *Cost-effective build outcome prediction using cascaded
  classifiers.* MSR.
- **Xia, J., & Li, Y. (2017).** *Could we predict the result of a continuous integration
  build? An empirical study.* QRS. — Reports that cross-project prediction is much harder
  than within-project. Directly supports your Scheme B split.
- **Saidani, I., Ouni, A., et al. (2022).** *Predicting continuous integration build
  failures using evolutionary search.* Information and Software Technology.
- **Ghaleb, T. A., da Costa, D. A., & Zou, Y. (2019).** *An empirical study of the long
  duration of continuous integration builds.* EMSE.

Also search for recent work on **data leakage in CI build prediction** — it strengthens
your §2.3 framing and shows awareness of the field's known pitfalls.

---

## Immediate next actions

1. Write the half-page scope amendment (Part 0) and get supervisor sign-off **this week**.
2. Upload the `.gz` to `MyDrive/fyp-deployment-prediction/data/raw/`.
3. Run Notebook 01 end to end. Confirm 677,863 builds and a 26.79% failure rate.
4. Run Notebook 02. Confirm XGBoost lands near 0.88 AUC.
5. Add the six references above to your literature review.

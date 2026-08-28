# CI/CD Failure Prediction Service

REST service that scores a build's probability of failure **before it runs**, returns a
SHAP explanation, and enforces a configurable gating policy. This is the artefact
component (RQ4) of the research project.

Model: XGBoost, 46 features, trained on TravisTorrent (677,863 builds, 1,283 projects).
Test-set performance: ROC-AUC 0.881, PR-AUC 0.787, precision 0.88 at threshold 0.80.

---

## The design problem this solves

The model's four strongest features — previous build outcome, consecutive failures,
failure rate over the last 5 and last 20 builds — account for roughly **86% of its
predictive power**. Without them, performance collapses from 0.88 to 0.64 ROC-AUC.

But a CI runner calling `/predict` only knows about the current commit. It has no idea
how the last twenty builds went.

So the service is **stateful**. It maintains a rolling per-project history in SQLite and
derives those features itself. The pipeline reports each build's result back via
`POST /outcome` once it finishes.

This mirrors training exactly: features are computed only from builds *strictly before*
the current one — the serving-time equivalent of the `.shift(1)` used in the notebook.

**Consequence:** a pipeline that calls `/predict` but never `/outcome` will silently
degrade to code-metrics-only accuracy while still returning plausible-looking scores.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. add your trained model
mkdir -p app/model
cp /path/to/xgb_model.joblib       app/model/
cp /path/to/model_metadata.json    app/model/

# 3. run
uvicorn app.main:app --reload

# 4. open the interactive docs
open http://localhost:8000/docs
```

Both files come from Notebook 02 section 14, in your Drive `models/` folder.

### Try it

```bash
# cold start — no history yet
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "project": "myorg/myapp",
  "git_diff_src_churn": 340,
  "git_diff_test_churn": 12,
  "gh_diff_files_modified": 8,
  "gh_sloc": 45000,
  "is_pr": true,
  "branch": "feature/new-auth"
}'

# record some failures
for i in 1 2 3 4 5; do
  curl -X POST localhost:8000/outcome -H 'Content-Type: application/json' \
    -d '{"project":"myorg/myapp","failed":true}'
done

# same request, much higher risk
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "project": "myorg/myapp", "git_diff_src_churn": 340,
  "git_diff_test_churn": 12, "gh_diff_files_modified": 8,
  "gh_sloc": 45000, "is_pr": true, "branch": "feature/new-auth"
}'
```

Measured behaviour for an identical request:

| History | Risk score | Decision |
|---|---|---|
| Cold start | 0.711 | LOW_RISK |
| 5 passing builds | 0.139 | LOW_RISK |
| 5 further failing builds | 0.955 | HIGH_RISK |

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Score a build before it runs |
| POST | `/outcome` | Report a finished build's result |
| GET | `/health` | Liveness and model status |
| GET | `/model-info` | Model version, threshold, training metrics |
| GET | `/projects` | Projects with recorded history |
| GET | `/docs` | Interactive OpenAPI documentation |

### Response shape

```json
{
  "risk_score": 0.9549,
  "decision": "HIGH_RISK",
  "threshold": 0.8,
  "policy": "block",
  "should_block": true,
  "top_factors": [
    {"feature": "prev_build_failed", "contribution": 1.3405, "value": 1.0},
    {"feature": "failure_rate_last_5", "contribution": 0.5477, "value": 1.0},
    {"feature": "consecutive_prior_failures", "contribution": 0.485, "value": 4.0}
  ],
  "history_available": true,
  "builds_in_history": 10,
  "model_version": "xgb-v1.0",
  "latency_ms": 4.1
}
```

`top_factors` are SHAP values. Positive contributions push towards failure. This is the
explainability requirement (proposal §4.7) satisfied inside the artefact, not just in the
offline analysis.

---

## Gating policies

Set via the `POLICY_MODE` environment variable. Staged rollout, as committed to in the
ethics section of the proposal.

| Mode | Behaviour | When to use |
|---|---|---|
| `shadow` | Always allows. Logs predictions only. | **Start here.** Collect predictions and compare against actual outcomes before acting on them. |
| `warn` | Allows, but annotates high-risk builds. | Once shadow-mode accuracy is confirmed. |
| `block` | Fails the pipeline step on high risk. | Only after the operating threshold is validated on your own data. |

`should_block` is `true` only under `block` policy. Shadow and warn are inert by
construction, and there is a test asserting this.

---

## Deployment

See `DEPLOYMENT.md` for the full deployment procedure (Neon Postgres + Render).
In short: set `DATABASE_URL` to a Postgres connection string so the build history
survives restarts, and set `POLICY_MODE=shadow` to begin.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `app/model` | Where the model and metadata live |
| `DATABASE_URL` | unset | Postgres connection string. When set, the history store uses Postgres instead of SQLite. Required for deployment on an ephemeral filesystem. |
| `DB_PATH` | `data/history.db` | SQLite file, used only when `DATABASE_URL` is unset |
| `POLICY_MODE` | `shadow` | `shadow` \| `warn` \| `block` |
| `RISK_THRESHOLD` | from metadata | Override the decision threshold |
| `PORT` | `8000` | Listen port |

---

## Docker

```bash
docker build -t cicd-risk-gate .
docker run -p 8000:8000 \
  -e POLICY_MODE=warn \
  -v $(pwd)/data:/code/data \
  cicd-risk-gate
```

Mount `/code/data` as a volume — otherwise the history store is lost on restart, and
every project resets to cold start.

---

## Deploying free

**Render** — connect the repo, choose Docker, add `POLICY_MODE` as an environment
variable, add a persistent disk mounted at `/code/data`.

**Railway / Fly.io** — same pattern; both detect the Dockerfile automatically.

**Hugging Face Spaces** — works, but the filesystem is ephemeral on the free tier, so
history resets on restart. Acceptable for a demo, not for collecting evidence.

Whichever you pick, verify the persistent volume actually persists. Losing the history
store means losing the features that carry 86% of the model's power.

---

## GitHub Actions integration

`.github/workflows/risk-gate.yml` is a working example. It extracts diff statistics,
requests a score, applies the policy, runs the build, and reports the outcome back.

Set `PREDICT_URL` as a repository secret pointing at your deployed service.

The workflow **fails open**: if the service is unreachable, the build proceeds with a
warning. A risk gate that blocks all deploys when it goes down is worse than no gate.

---

## Tests

```bash
pytest tests/ -v
```

14 tests. The important ones are `test_failing_history_raises_risk` and
`test_history_is_per_project` — if those pass, the history mechanism is wired correctly.
If they fail, predictions will look reasonable while being far less accurate than the
offline evaluation suggests.

---

## Training/serving skew

`app/features.py` is the single source of truth for feature assembly and must match
Notebook 01 exactly. If the API computes a derived ratio even slightly differently, no
error is raised — accuracy just quietly drops.

Two safeguards:

- `validate_feature_order()` runs at startup and refuses to serve if the model's expected
  feature order differs from `FEATURE_ORDER`.
- `test_derived_ratios_match_notebook` asserts the derived-ratio formulas numerically.

When you retrain, re-export `model_metadata.json` with the current `feature_order`.

---

## Known limitations

- **Cold start.** A project with no recorded history gets a prediction based only on code
  features, which is materially weaker. `history_available: false` flags this — treat
  those scores with caution and prefer shadow mode until history accumulates.
- **SQLite is single-writer.** Fine at CI volumes. Swap in Postgres behind the same
  interface if you outgrow it.
- **No authentication.** Add an API key before exposing this publicly.
- **Training data is 2011–2016 Travis CI**, predominantly Ruby and Java. Transfer to
  modern GitHub Actions pipelines in other languages is untested.

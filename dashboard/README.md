# Monitoring dashboard

Web interface for the CI/CD failure prediction service. Four views: score a
build, explore gate settings, review the model evidence, and see recorded
activity.

## Running it

Start the prediction service first, from the `artefact` folder:

    python -m uvicorn app.main:app --reload

Then, from this folder, in a second terminal:

    pip install -r requirements.txt
    streamlit run dashboard.py

It opens at http://localhost:8501.

If the service runs somewhere else — behind an ngrok tunnel, for instance —
change the address in the sidebar. No restart needed.

## The four views

**Score a build.** Enter the characteristics of a change and get a risk score
with the factors behind it. Orange bars push towards failure, teal towards
success. Below the form you can report what actually happened, which is what
keeps the history features current.

**Gate settings.** A stricter gate catches more failures but delays more
healthy builds. The slider moves across the nine thresholds that were actually
measured on the held-out test set — nothing between them is interpolated. Each
setting shows precision, recall, how many builds would be stopped, and what the
failure rate would drop to.

**Model evidence.** The offline evaluation: three models compared, the ablation
showing where the signal comes from, SHAP feature importance, and cross-project
results. Read from the CSVs in `data/`.

**Activity.** Projects, builds and predictions the service has recorded. Sparse
until the gate has been running against a pipeline for a while.

## Data

`data/` holds the results exported by the modelling notebook:

| File | Contents |
|---|---|
| `model_comparison.csv` | Logistic Regression, Random Forest, XGBoost on the test set |
| `ablation.csv` | Performance by feature group |
| `gate_simulation.csv` | Gate outcomes at nine thresholds over 101,680 builds |
| `cross_project.csv` | Five-fold project-grouped cross-validation |
| `shap_importance.csv` | Mean absolute SHAP value per feature |

Precision, recall and F1 in the gate settings view are derived from the counts
in `gate_simulation.csv` rather than stored separately, so they always agree
with the confusion figures shown alongside them.

To refresh after retraining, re-export these five files from the notebook and
replace them here.

## Notes

The three views that read from `data/` work whether or not the service is
running. Only "Score a build" and "Activity" need a live connection, and both
say so plainly when they cannot reach it.

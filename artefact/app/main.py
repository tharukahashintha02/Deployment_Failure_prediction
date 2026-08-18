"""
Deployment / CI failure risk prediction service.

Endpoints
    POST /predict     score a build before it runs
    POST /outcome     report the result afterwards (feeds the history store)
    GET  /health      liveness + model status
    GET  /model-info  model metadata and current policy
    GET  /projects    projects with recorded history
    GET  /docs        interactive OpenAPI docs (FastAPI provides this free)

Run locally:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from app.features import FEATURE_ORDER, build_feature_vector, to_row, validate_feature_order
from app.schemas import (Decision, Factor, HealthResponse, ModelInfoResponse,
                         OutcomeRequest, OutcomeResponse, PolicyMode,
                         PredictRequest, PredictResponse)
from app.store import HistoryStore

# ---------------------------------------------------------------- config
MODEL_DIR = Path(os.getenv("MODEL_DIR", "app/model"))
DB_PATH = os.getenv("DB_PATH", "data/history.db")
POLICY = PolicyMode(os.getenv("POLICY_MODE", "shadow"))
THRESHOLD_OVERRIDE = os.getenv("RISK_THRESHOLD")

state: dict = {"model": None, "explainer": None, "metadata": {}, "threshold": 0.8}
store = HistoryStore(DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


def _load_model() -> None:
    """Load model + metadata at startup. Fails loudly on feature-order mismatch."""
    model_path = MODEL_DIR / "xgb_model.joblib"
    meta_path = MODEL_DIR / "model_metadata.json"

    if not model_path.exists():
        print(f"WARNING: no model at {model_path} — /predict will return 503")
        return

    state["model"] = joblib.load(model_path)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        state["metadata"] = meta
        state["threshold"] = float(meta.get("chosen_threshold", 0.8))
        if meta.get("feature_order"):
            validate_feature_order(meta["feature_order"])
    else:
        print("WARNING: model_metadata.json missing — using default threshold 0.8")

    if THRESHOLD_OVERRIDE:
        state["threshold"] = float(THRESHOLD_OVERRIDE)

    # SHAP is optional; the service still works without explanations
    try:
        import shap
        state["explainer"] = shap.TreeExplainer(state["model"])
    except Exception as exc:  # noqa: BLE001
        print(f"SHAP unavailable ({exc}) — predictions will omit top_factors")

    print(f"Model loaded: {state['metadata'].get('model_version', 'unknown')} | "
          f"threshold={state['threshold']:.3f} | policy={POLICY.value}")


app = FastAPI(
    title="CI/CD Failure Prediction Service",
    description=(
        "Predicts the probability that a CI/CD build will fail, before it runs. "
        "Returns a risk score, a gating decision, and a SHAP explanation.\n\n"
        "The strongest features describe the project's recent build history, so "
        "call POST /outcome after each build to keep the history current."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- endpoints
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Score a build. Call this BEFORE running it — never after recording its outcome."""
    if state["model"] is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()

    history = store.summarise(req.project)
    features = build_feature_vector(req.model_dump(), history)
    row = to_row(features)

    X = np.array([row], dtype=np.float32)
    risk = float(state["model"].predict_proba(X)[0, 1])

    threshold = state["threshold"]
    decision = Decision.HIGH_RISK if risk >= threshold else Decision.LOW_RISK
    should_block = (POLICY == PolicyMode.BLOCK and decision == Decision.HIGH_RISK)

    top_factors = _explain(X, features)

    store.record_prediction(req.project, req.build_ref, risk,
                            decision.value, threshold)

    return PredictResponse(
        risk_score=round(risk, 4),
        decision=decision,
        threshold=threshold,
        policy=POLICY,
        should_block=should_block,
        top_factors=top_factors,
        history_available=history["builds_so_far_in_project"] > 0,
        builds_in_history=int(history["builds_so_far_in_project"]),
        model_version=state["metadata"].get("model_version", "unknown"),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


def _explain(X: np.ndarray, features: dict, top_n: int = 5) -> list[Factor]:
    """SHAP values for this single prediction. Returns [] if SHAP is unavailable."""
    if state["explainer"] is None:
        return []
    try:
        vals = state["explainer"].shap_values(X)
        vals = np.asarray(vals)
        if vals.ndim == 3:          # some SHAP versions return (n, features, classes)
            vals = vals[0, :, -1]
        else:
            vals = vals[0]
        order = np.argsort(np.abs(vals))[::-1][:top_n]
        return [
            Factor(feature=FEATURE_ORDER[i],
                   contribution=round(float(vals[i]), 4),
                   value=round(float(features.get(FEATURE_ORDER[i], 0.0)), 4))
            for i in order
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"SHAP explanation failed: {exc}")
        return []


@app.post("/outcome", response_model=OutcomeResponse, tags=["prediction"])
def record_outcome(req: OutcomeRequest) -> OutcomeResponse:
    """
    Report a finished build's result.

    This is what keeps the history features current, and they carry ~86% of the
    model's predictive power. A pipeline that calls /predict but never /outcome
    will degrade to code-metrics-only performance (roughly 0.64 AUC instead of 0.88).
    """
    store.record_outcome(req.project, req.failed, req.build_ref, req.finished_at)
    stats = store.project_stats(req.project)
    return OutcomeResponse(
        recorded=True,
        project=req.project,
        builds_in_history=stats["builds_recorded"],
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state["model"] is not None else "degraded",
        model_loaded=state["model"] is not None,
        model_version=state["metadata"].get("model_version", "unknown"),
        n_features=len(FEATURE_ORDER),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    meta = state["metadata"]
    return ModelInfoResponse(
        model_version=meta.get("model_version", "unknown"),
        trained_at=meta.get("trained_at"),
        n_features=len(FEATURE_ORDER),
        threshold=state["threshold"],
        policy=POLICY,
        dataset=meta.get("dataset"),
        label_definition=meta.get("label_definition"),
        test_metrics=meta.get("test_metrics"),
    )


@app.get("/projects", tags=["ops"])
def projects() -> dict:
    names = store.list_projects()
    return {"count": len(names),
            "projects": [store.project_stats(n) for n in names]}

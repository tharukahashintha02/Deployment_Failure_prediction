"""Request / response models. Pydantic validates these automatically and
FastAPI turns them into the OpenAPI docs at /docs."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    LOW_RISK = "LOW_RISK"
    HIGH_RISK = "HIGH_RISK"


class PolicyMode(str, Enum):
    """Staged rollout, as committed to in the ethics section of the proposal.

    shadow — always allow, log the prediction. Start here.
    warn   — allow, but flag high-risk builds to the developer.
    block  — fail the pipeline step on high risk.
    """
    SHADOW = "shadow"
    WARN = "warn"
    BLOCK = "block"


class PredictRequest(BaseModel):
    # --- identity (required) ---
    project: str = Field(..., description="Repository identifier, e.g. 'org/repo'",
                         examples=["rails/rails"])
    build_ref: str | None = Field(
        None, description="Commit SHA or build number, for later outcome matching")

    # --- code change (from git diff) ---
    git_diff_src_churn: float = Field(0, ge=0, description="Lines changed in source files")
    git_diff_test_churn: float = Field(0, ge=0, description="Lines changed in test files")
    gh_diff_files_added: float = Field(0, ge=0)
    gh_diff_files_deleted: float = Field(0, ge=0)
    gh_diff_files_modified: float = Field(0, ge=0)
    gh_diff_tests_added: float = Field(0, ge=0)
    gh_diff_tests_deleted: float = Field(0, ge=0)
    gh_diff_src_files: float = Field(0, ge=0)
    gh_diff_doc_files: float = Field(0, ge=0)
    gh_diff_other_files: float = Field(0, ge=0)
    gh_num_commits_on_files_touched: float = Field(0, ge=0)
    git_num_all_built_commits: float = Field(1, ge=0)
    gh_num_commits_in_push: float | None = Field(
        None, description="Null for PR builds — handled by a missing-value indicator")

    # --- project health ---
    gh_sloc: float = Field(0, ge=0, description="Project source lines of code")
    gh_test_lines_per_kloc: float = Field(0, ge=0)
    gh_test_cases_per_kloc: float = Field(0, ge=0)
    gh_asserts_cases_per_kloc: float = Field(0, ge=0)
    gh_team_size: float = Field(1, ge=0, description="Contributors active recently")

    # --- context ---
    is_pr: bool = Field(False, description="Pull request build vs direct push")
    by_core_team_member: bool = Field(False)
    branch: str = Field("main")
    language: str = Field("ruby", description="'java' is modelled explicitly")
    build_started_at: datetime | None = Field(
        None, description="ISO timestamp; defaults to now")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "project": "myorg/myapp",
                "build_ref": "a1b2c3d",
                "git_diff_src_churn": 340,
                "git_diff_test_churn": 12,
                "gh_diff_files_modified": 8,
                "gh_diff_src_files": 6,
                "gh_sloc": 45000,
                "gh_test_lines_per_kloc": 180,
                "gh_team_size": 7,
                "is_pr": True,
                "branch": "feature/new-auth",
                "language": "ruby",
            }]
        }
    }


class Factor(BaseModel):
    feature: str
    contribution: float = Field(
        ..., description="SHAP value. Positive pushes towards failure.")
    value: float = Field(..., description="This build's value for the feature")


class PredictResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    decision: Decision
    threshold: float
    policy: PolicyMode
    should_block: bool = Field(
        ..., description="True only when policy is 'block' AND risk exceeds threshold")
    top_factors: list[Factor] = Field(
        default_factory=list, description="SHAP explanation of this prediction")
    history_available: bool = Field(
        ..., description="False for a cold-start project — treat the score with caution")
    builds_in_history: int
    model_version: str
    latency_ms: float


class OutcomeRequest(BaseModel):
    project: str
    failed: bool = Field(..., description="True if the build failed or errored")
    build_ref: str | None = None
    finished_at: datetime | None = None


class OutcomeResponse(BaseModel):
    recorded: bool
    project: str
    builds_in_history: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    n_features: int


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str | None
    n_features: int
    threshold: float
    policy: PolicyMode
    dataset: str | None
    label_definition: str | None
    test_metrics: dict | None

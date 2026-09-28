"""
CI/CD Pipeline Failure Prediction — Monitoring Dashboard

Runs alongside the prediction service. Four views:

  Score a build     live prediction against the running API, with the factors
                    behind the score
  Gate settings     explore the safety/friction trade-off across the measured
                    decision thresholds
  Model evidence    the offline evaluation results from the dissertation
  Activity          predictions and outcomes recorded by the service so far

Start the API first:
    python -m uvicorn app.main:app --reload

Then, from the dashboard folder:
    streamlit run dashboard.py

Author: Tharuka H. Dilshan, NSBM Green University
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------- constants
DATA = Path(__file__).parent / "data"
DEFAULT_API = "http://localhost:8000"

# Palette matches the dissertation figures so the dashboard and the written
# results read as one piece of work.
SLATE = "#264653"
TEAL = "#2A9D8F"
CLAY = "#E76F51"
SAND = "#E9C46A"
MUTED = "#7A8B91"
PAPER = "#FBFAF8"

st.set_page_config(page_title="Pipeline risk gate",
                   page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background: {PAPER}; }}
  h1, h2, h3 {{ color: {SLATE}; letter-spacing: -0.01em; }}
  .verdict {{
      border-left: 6px solid {SLATE};
      padding: 1.1rem 1.4rem;
      background: #fff;
      border-radius: 2px;
  }}
  .verdict .score {{
      font-size: 3.4rem; font-weight: 700; line-height: 1;
      color: {SLATE}; font-variant-numeric: tabular-nums;
  }}
  .verdict .label {{ color: {MUTED}; font-size: .85rem; }}
  .risky {{ border-left-color: {CLAY}; }}
  .risky .score {{ color: {CLAY}; }}
  .safe {{ border-left-color: {TEAL}; }}
  .safe .score {{ color: {TEAL}; }}
  [data-testid="stMetricValue"] {{ font-size: 1.6rem; color: {SLATE}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_results() -> dict:
    """Offline evaluation results produced by the modelling notebook."""
    out = {}
    for name in ("model_comparison", "ablation", "gate_simulation",
                 "cross_project", "shap_importance"):
        path = DATA / f"{name}.csv"
        if path.exists():
            idx = 0 if name in ("model_comparison", "shap_importance") else None
            out[name] = pd.read_csv(path, index_col=idx)
    return out


def api(base: str, path: str, method: str = "GET", payload: dict | None = None,
        timeout: int = 20):
    url = f"{base.rstrip('/')}{path}"
    headers = {"ngrok-skip-browser-warning": "true"}
    # a corporate/system proxy must not intercept calls to a local service
    proxies = {"http": None, "https": None} if "localhost" in base or "127.0.0.1" in base else None
    try:
        if method == "POST":
            r = requests.post(url, json=payload, headers=headers, timeout=timeout,
                              proxies=proxies)
        else:
            r = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the service. Start it with: uvicorn app.main:app --reload"
    except requests.exceptions.Timeout:
        return None, "The service did not respond in time."
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


R = load_results()


def derive_threshold_table() -> pd.DataFrame:
    """
    Precision and recall at each measured threshold, derived from the gate
    simulation counts. Only thresholds that were actually evaluated appear —
    nothing is interpolated.
    """
    g = R["gate_simulation"].copy()
    tp = g["failures_caught"]
    fp = g["false_blocks"]
    fn = g["escaped_failures"]
    g["precision"] = tp / (tp + fp)
    g["recall"] = tp / (tp + fn)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
    g["true_negatives"] = (tp + fp + fn) * 0 + (
        101680 - tp - fp - fn)  # test set size
    return g


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"### <span style='color:{SLATE}'>Pipeline risk gate</span>",
                unsafe_allow_html=True)
    st.caption("Predicts CI/CD build failure before execution")

    api_base = st.text_input("Service address", DEFAULT_API,
                             help="Change this if the service runs elsewhere, "
                                  "for example behind an ngrok tunnel.")

    health, err = api(api_base, "/health", timeout=8)
    if health:
        store = health.get("storage", {})
        st.success("Service reachable")
        st.caption(
            f"Model {health.get('model_version')} · "
            f"{health.get('n_features')} features\n\n"
            f"History: {store.get('backend', 'unknown')}"
            f"{' (persistent)' if store.get('persistent') else ' (not persistent)'}"
        )
    else:
        st.error("Service unreachable")
        st.caption(err or "")
        st.caption("The gate settings and model evidence views still work — "
                   "they read saved results rather than the live service.")

st.title("Pipeline risk gate")
st.caption("Machine learning failure prediction for CI/CD pipelines · "
           "Tharuka H. Dilshan · NSBM Green University")

tab_score, tab_gate, tab_model, tab_activity = st.tabs(
    ["Score a build", "Gate settings", "Model evidence", "Activity"])


# ============================================================ SCORE A BUILD
with tab_score:
    st.subheader("Score a build before it runs")
    st.write("Enter the characteristics of a change and the project it belongs "
             "to. The score reflects both the change itself and how that "
             "project's recent builds have gone.")

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        project = st.text_input("Project", "demo/app",
                                help="Build history is kept separately for each project.")
        build_ref = st.text_input("Commit reference", "", placeholder="optional")

        st.markdown("**The change**")
        c1, c2 = st.columns(2)
        src_churn = c1.number_input("Source lines changed", 0, 100000, 340, 10)
        test_churn = c2.number_input("Test lines changed", 0, 100000, 12, 10)
        files_mod = c1.number_input("Files modified", 0, 5000, 8)
        files_add = c2.number_input("Files added", 0, 5000, 0)
        src_files = c1.number_input("Source files touched", 0, 5000, 6)
        doc_files = c2.number_input("Doc files touched", 0, 5000, 0)

        st.markdown("**The project**")
        c3, c4 = st.columns(2)
        sloc = c3.number_input("Project size (lines)", 0, 10_000_000, 45000, 1000)
        team = c4.number_input("Active contributors", 0, 1000, 7)
        test_density = c3.number_input("Test lines per KLOC", 0, 5000, 180)
        branch = c4.text_input("Branch", "feature/new-auth")

        is_pr = st.checkbox("Pull request build", True)
        core = st.checkbox("Author is a core contributor", False)

        go_btn = st.button("Score this build", type="primary", width="stretch")

    with right:
        if go_btn:
            payload = {
                "project": project, "build_ref": build_ref or None,
                "git_diff_src_churn": src_churn, "git_diff_test_churn": test_churn,
                "gh_diff_files_modified": files_mod, "gh_diff_files_added": files_add,
                "gh_diff_src_files": src_files, "gh_diff_doc_files": doc_files,
                "gh_sloc": sloc, "gh_team_size": team,
                "gh_test_lines_per_kloc": test_density,
                "is_pr": is_pr, "by_core_team_member": core,
                "branch": branch, "language": "ruby",
            }
            result, err = api(api_base, "/predict", "POST", payload)

            if err:
                st.error(err)
            else:
                st.session_state["last_result"] = result

        result = st.session_state.get("last_result")
        if result:
            risky = result["decision"] == "HIGH_RISK"
            css = "risky" if risky else "safe"
            verdict = "Flagged as high risk" if risky else "Within acceptable risk"

            st.markdown(f"""
            <div class="verdict {css}">
              <div class="label">Failure probability</div>
              <div class="score">{result['risk_score']:.3f}</div>
              <div style="margin-top:.5rem;font-weight:600;color:{SLATE}">{verdict}</div>
              <div class="label" style="margin-top:.35rem">
                Threshold {result['threshold']:.2f} ·
                {result['policy']} policy ·
                {'build would be stopped' if result['should_block'] else 'build proceeds'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Builds on record", result["builds_in_history"])
            m2.metric("History available", "Yes" if result["history_available"] else "No")
            m3.metric("Response time", f"{result['latency_ms']:.0f} ms")

            if not result["history_available"]:
                st.info("This project has no recorded builds yet, so the score "
                        "rests on the change alone. It will sharpen as outcomes "
                        "are reported.")

            factors = result.get("top_factors", [])
            if factors:
                st.markdown("**What drove this score**")
                f = pd.DataFrame(factors).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=f["contribution"], y=f["feature"], orientation="h",
                    marker_color=[CLAY if c > 0 else TEAL for c in f["contribution"]],
                    text=[f"{v:g}" for v in f["value"]],
                    textposition="outside",
                    hovertemplate="%{y}<br>value %{text}<br>"
                                  "contribution %{x:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    height=40 * len(f) + 90,
                    margin=dict(l=0, r=20, t=10, b=30),
                    xaxis_title="Push towards failure  →",
                    yaxis_title=None, plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color=SLATE, size=12), showlegend=False,
                )
                fig.add_vline(x=0, line_width=1, line_color=MUTED)
                st.plotly_chart(fig, width="stretch")
                st.caption("Orange pushes the prediction towards failure, "
                           "teal towards success. Values are SHAP contributions.")
        else:
            st.info("Fill in the details and select **Score this build** to see "
                    "a prediction and the reasoning behind it.")

    st.divider()
    st.markdown("**Report what happened**")
    st.write("Recording outcomes is what keeps predictions accurate — the "
             "strongest signals come from a project's recent build history.")
    o1, o2, o3 = st.columns([2, 1, 1])
    out_project = o1.text_input("Project", project, key="outcome_project",
                                label_visibility="collapsed")
    def _report(failed: bool) -> None:
        res, err = api(api_base, "/outcome", "POST",
                       {"project": out_project, "failed": failed})
        if res:
            st.success(f"Recorded. {res['builds_in_history']} builds now on "
                       f"record for {out_project}.")
        else:
            st.error(err)

    if o2.button("Build passed", width="stretch"):
        _report(False)
    if o3.button("Build failed", width="stretch"):
        _report(True)


# ============================================================ GATE SETTINGS
with tab_gate:
    st.subheader("Choosing where to set the gate")
    st.write("A stricter gate catches more failures but delays more healthy "
             "builds. This is a policy decision rather than a property of the "
             "model, so the whole range is shown. Figures come from replaying "
             "101,680 held-out builds.")

    if "gate_simulation" not in R:
        st.warning("gate_simulation.csv not found in the data folder.")
    else:
        gt = derive_threshold_table()
        options = [round(t, 2) for t in gt["threshold"].tolist()]
        chosen = st.select_slider("Decision threshold", options=options,
                                  value=0.8 if 0.8 in options else options[len(options) // 2])
        row = gt[gt.threshold.round(2) == chosen].iloc[0]

        k = st.columns(5)
        k[0].metric("Precision", f"{row.precision:.3f}",
                    help="Of the builds it stops, how many really would have failed")
        k[1].metric("Recall", f"{row.recall:.3f}",
                    help="Of the builds that would have failed, how many it stops")
        k[2].metric("F1", f"{row.f1:.3f}")
        k[3].metric("Builds stopped", f"{row.pct_builds_blocked:.1f}%")
        k[4].metric("Failure rate after gating",
                    f"{row.residual_failure_rate:.1f}%",
                    delta=f"{row.residual_failure_rate - row.original_failure_rate:.1f} pts",
                    delta_color="inverse")

        st.markdown(
            f"At this setting the gate stops **{int(row.failures_caught):,}** of the "
            f"**{int(row.failures_caught + row.escaped_failures):,}** builds that would "
            f"have failed, while delaying **{int(row.false_blocks):,}** healthy ones — "
            f"about one in {row.one_in_n_healthy_blocked:.0f}. The proportion of builds "
            f"that fail drops from {row.original_failure_rate:.1f}% to "
            f"{row.residual_failure_rate:.1f}%."
        )

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            for col, colour, name in [("precision", SLATE, "Precision"),
                                      ("recall", TEAL, "Recall"),
                                      ("f1", CLAY, "F1")]:
                fig.add_trace(go.Scatter(x=gt.threshold, y=gt[col], name=name,
                                         mode="lines+markers",
                                         line=dict(color=colour, width=2)))
            fig.add_vline(x=chosen, line_dash="dot", line_color=MUTED)
            fig.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=10, t=30, b=0),
                              title="Accuracy across the threshold range",
                              xaxis_title="Threshold", yaxis_title="Score",
                              font=dict(color=SLATE, size=12),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gt.threshold, y=gt.pct_failures_caught,
                                     name="Failures stopped", mode="lines+markers",
                                     line=dict(color=TEAL, width=2)))
            fig.add_trace(go.Scatter(x=gt.threshold, y=gt.pct_builds_blocked,
                                     name="All builds stopped", mode="lines+markers",
                                     line=dict(color=CLAY, width=2, dash="dash")))
            fig.add_vline(x=chosen, line_dash="dot", line_color=MUTED)
            fig.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=10, t=30, b=0),
                              title="Safety against friction",
                              xaxis_title="Threshold", yaxis_title="Percent",
                              font=dict(color=SLATE, size=12),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Outcomes at this threshold**")
        tn = 101680 - row.failures_caught - row.false_blocks - row.escaped_failures
        grid = pd.DataFrame(
            [[f"{int(tn):,}", f"{int(row.false_blocks):,}"],
             [f"{int(row.escaped_failures):,}", f"{int(row.failures_caught):,}"]],
            index=["Would have passed", "Would have failed"],
            columns=["Allowed through", "Stopped by the gate"])
        st.table(grid)
        st.caption("Top right: healthy builds delayed unnecessarily. "
                   "Bottom left: failures the gate let through.")

        with st.expander("All measured thresholds"):
            show = gt[["threshold", "precision", "recall", "f1",
                       "pct_builds_blocked", "pct_failures_caught",
                       "residual_failure_rate"]].round(3)
            show.columns = ["Threshold", "Precision", "Recall", "F1",
                            "% builds stopped", "% failures caught",
                            "Failure rate after"]
            st.dataframe(show, width="stretch", hide_index=True)


# =========================================================== MODEL EVIDENCE
with tab_model:
    st.subheader("How the model was evaluated")
    st.write("Results from 677,863 builds across 1,283 open-source projects "
             "(TravisTorrent). Training used the earliest builds and testing "
             "the most recent, so no future information reaches the model.")

    if "model_comparison" in R:
        mc = R["model_comparison"]
        st.markdown("**Comparing the three models**")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="ROC-AUC", x=mc.index, y=mc.roc_auc,
                                 marker_color=SLATE,
                                 text=mc.roc_auc.round(4), textposition="outside"))
            fig.add_trace(go.Bar(name="PR-AUC", x=mc.index, y=mc.pr_auc,
                                 marker_color=CLAY,
                                 text=mc.pr_auc.round(4), textposition="outside"))
            fig.update_layout(barmode="group", height=330, yaxis_range=[0, 1],
                              plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=10, t=10, b=0),
                              font=dict(color=SLATE, size=12),
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
        with c2:
            show = mc[["roc_auc", "pr_auc", "precision", "recall", "f1"]].round(4)
            show.columns = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
            st.dataframe(show, width="stretch")
            st.caption("The three perform closely. Confidence intervals for "
                       "Random Forest and XGBoost overlap, so neither can be "
                       "claimed superior; XGBoost trains roughly eight times "
                       "faster at equal accuracy.")

    st.divider()

    if "ablation" in R:
        ab = R["ablation"]
        st.markdown("**Where the predictive signal comes from**")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ab.config, x=ab.roc_auc, orientation="h",
                             name="ROC-AUC", marker_color=SLATE,
                             text=ab.roc_auc.round(3), textposition="outside"))
        fig.add_trace(go.Bar(y=ab.config, x=ab.pr_auc, orientation="h",
                             name="PR-AUC", marker_color=CLAY,
                             text=ab.pr_auc.round(3), textposition="outside"))
        fig.update_layout(barmode="group", height=330, xaxis_range=[0, 1.05],
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=30, t=10, b=0),
                          font=dict(color=SLATE, size=12),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        gain = ab.loc[ab.config == "Code + history", "roc_auc"].iloc[0] - \
            ab.loc[ab.config == "Code/change features only", "roc_auc"].iloc[0]
        st.markdown(
            f"Characteristics of the change alone predict poorly. Adding a "
            f"project's recent build history raises discrimination by "
            f"**{gain:.2f} ROC-AUC**. Pipeline state, not change size, carries "
            f"the signal — which is why the service keeps a build history rather "
            f"than scoring each commit in isolation."
        )

    st.divider()

    if "shap_importance" in R:
        si = R["shap_importance"].iloc[:, 0].head(12).iloc[::-1]
        st.markdown("**Which inputs matter most**")
        history_feats = {"prev_build_failed", "consecutive_prior_failures",
                         "failure_rate_last_5", "failure_rate_last_20",
                         "project_cum_failure_rate", "builds_so_far_in_project",
                         "hours_since_last_build", "builds_in_last_24h"}
        fig = go.Figure(go.Bar(
            x=si.values, y=si.index, orientation="h",
            marker_color=[CLAY if f in history_feats else SLATE for f in si.index],
            text=si.values.round(3), textposition="outside"))
        fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=40, t=10, b=30),
                          xaxis_title="Average influence on the prediction",
                          font=dict(color=SLATE, size=12), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Orange marks features derived from build history. "
                   "The five most influential are all of that kind.")

    st.divider()

    if "cross_project" in R:
        cp = R["cross_project"]
        st.markdown("**Does it work on a project it has never seen?**")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"Fold {int(f)}" for f in cp.fold],
                                 y=cp.roc_auc, marker_color=TEAL,
                                 text=cp.roc_auc.round(3), textposition="outside",
                                 name="Unseen projects"))
            fig.add_hline(y=0.8807, line_dash="dash", line_color=CLAY,
                          annotation_text="Same projects, later builds (0.881)",
                          annotation_position="bottom right")
            fig.update_layout(height=320, yaxis_range=[0.8, 0.92],
                              plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=10, t=10, b=0),
                              yaxis_title="ROC-AUC",
                              font=dict(color=SLATE, size=12), showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.metric("Average on unseen projects", f"{cp.roc_auc.mean():.4f}",
                      delta=f"{cp.roc_auc.mean() - 0.8807:+.4f} against held-out builds")
            st.write("Performance holds up on projects absent from training. "
                     "The history features describe each project relative to its "
                     "own recent behaviour rather than in absolute terms, so the "
                     "pattern transfers. A team can adopt the gate without "
                     "retraining on their own repository.")


# ================================================================= ACTIVITY
with tab_activity:
    st.subheader("What the service has seen")

    projects, err = api(api_base, "/projects", timeout=10)
    if err:
        st.error(err)
        st.caption("This view reads from the running service.")
    elif not projects or projects.get("count", 0) == 0:
        st.info("No builds recorded yet. Score a build and report its outcome, "
                "or connect the GitHub Actions workflow, and activity will "
                "appear here.")
    else:
        rows = pd.DataFrame(projects["projects"])
        rows["failure_rate"] = (rows.failures_recorded /
                                rows.builds_recorded.replace(0, pd.NA))

        m = st.columns(4)
        m[0].metric("Projects", projects["count"])
        m[1].metric("Builds recorded", int(rows.builds_recorded.sum()))
        m[2].metric("Failures", int(rows.failures_recorded.sum()))
        m[3].metric("Predictions made", int(rows.predictions_made.sum()))

        st.markdown("**By project**")
        show = rows.copy()
        show["failure_rate"] = show.failure_rate.map(
            lambda v: "—" if pd.isna(v) else f"{v:.0%}")
        show.columns = ["Project", "Builds", "Failures", "Predictions", "Failure rate"]
        st.dataframe(show, width="stretch", hide_index=True)

        if len(rows) > 1:
            fig = go.Figure(go.Bar(
                x=rows.project, y=rows.builds_recorded, marker_color=SLATE,
                name="Builds"))
            fig.add_trace(go.Bar(x=rows.project, y=rows.failures_recorded,
                                 marker_color=CLAY, name="Failures"))
            fig.update_layout(barmode="overlay", height=320,
                              plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=10, t=10, b=0),
                              font=dict(color=SLATE, size=12),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Recorded activity reflects use of this service. It will be sparse "
            "until the gate has been running against a pipeline for some time; "
            "the model evidence tab reports the full offline evaluation.")

st.divider()
st.caption(f"Prediction service at {api_base} · "
           f"Offline results from the modelling notebooks · "
           f"Loaded {datetime.now():%d %b %Y, %H:%M}")

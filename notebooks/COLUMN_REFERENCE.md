# TravisTorrent Column Reference — Keep, Drop, and Why

All 62 columns classified. Reproduce the LEAKAGE table in your methodology chapter —
it is a genuine methodological contribution and demonstrates rigour to your examiners.

---

## CATEGORY 1 — MUST DROP: outcome leakage (19 columns)

These are computed **from the build log after the build finished**. At prediction time
(before the build runs) you do not have them. Including any one of them produces a model
scoring >0.95 AUC that is worthless in production.

| Column | Why it leaks |
|---|---|
| `build_successful` | Is the label itself |
| `tr_status` | Is the label itself (source of the label) |
| `tr_log_status` | Parsed from the finished build log |
| `tr_log_bool_tests_failed` | Directly encodes failure |
| `tr_log_num_tests_failed` | Directly encodes failure |
| `tr_log_tests_failed` | Names of the tests that failed |
| `tr_log_num_tests_ok` | Only known after the run |
| `tr_log_num_tests_run` | Only known after the run |
| `tr_log_num_tests_skipped` | Only known after the run |
| `tr_log_bool_tests_ran` | Only known after the run |
| `tr_duration` | Duration is an outcome, not an input |
| `tr_log_buildduration` | Duration is an outcome, not an input |
| `tr_log_testduration` | Duration is an outcome, not an input |
| `tr_log_setup_time` | Only known after the run |
| `tr_log_analyzer` | Derived from log parsing |
| `tr_log_frameworks` | Derived from log parsing |
| `tr_log_lan` | Derived from log parsing |
| `git_merged_with` | The merge happened after the build |
| `tr_virtual_merged_into` | The merge happened after the build |

> **Self-check:** if your test AUC exceeds ~0.95, you have leaked. Real performance on
> this dataset with honest features is ~0.88 AUC. Anything far above that means a leaky
> column crept back in.

---

## CATEGORY 2 — DROP: identifiers, raw hashes, unusable (13 columns)

| Column | Why |
|---|---|
| `tr_job_id` | Job identifier, no signal |
| `tr_jobs` | Raw list of job IDs |
| `git_all_built_commits` | Raw commit-hash string |
| `gh_commits_in_push` | Raw commit-hash string |
| `git_trigger_commit` | Commit hash |
| `tr_original_commit` | Commit hash |
| `git_prev_built_commit` | Commit hash |
| `gh_description_complexity` | **85.6% missing** — not recoverable |
| `gh_pr_created_at` | Redundant with build timestamp |
| `gh_first_commit_created_at` | High missingness, low value |
| `gh_pushed_at` | High missingness |
| `gh_pull_req_num` | Identifier |
| `git_prev_commit_resolution_status` | Metadata about how the dataset was built, not about the build |

**Note on `tr_build_id`, `gh_project_name`, `gh_build_started_at`:** keep these as
*columns* for grouping, splitting, and joining — but never pass them to the model as
features. `gh_project_name` in particular would let the model memorise that `rails/rails`
fails at rate X, which does not transfer to a new project.

---

## CATEGORY 3 — BORDERLINE: use with caution (3 columns)

| Column | Issue |
|---|---|
| `gh_num_issue_comments` | Counted at *dataset-collection* time, not build time |
| `gh_num_pr_comments` | Same — a failing PR attracts more discussion afterwards |
| `gh_num_commit_comments` | Same |

These are weak reverse-causal leaks. **Default: exclude.** The notebook has an
`INCLUDE_SOCIAL = False` flag. Run the model both ways and report the delta as a
sensitivity analysis — that is a legitimate, examinable result either way.

---

## CATEGORY 4 — KEEP: valid raw features (24 columns)

All of these are knowable **before** the build starts.

### Code change — the direct signal (11)
| Column | Meaning |
|---|---|
| `git_diff_src_churn` | Lines changed in source files |
| `git_diff_test_churn` | Lines changed in test files |
| `gh_diff_files_added` | Files added |
| `gh_diff_files_deleted` | Files deleted |
| `gh_diff_files_modified` | Files modified |
| `gh_diff_tests_added` | Test files added |
| `gh_diff_tests_deleted` | Test files deleted |
| `gh_diff_src_files` | Source files touched |
| `gh_diff_doc_files` | Doc files touched |
| `gh_diff_other_files` | Other files touched |
| `gh_num_commits_on_files_touched` | Historical commit count on touched files (proxy for hotspot code) |

### Commit/push volume (2)
| Column | Note |
|---|---|
| `git_num_all_built_commits` | Commits in this build |
| `gh_num_commits_in_push` | 20.4% missing (NA for PR builds) — impute from the column above + missing flag |

### Project health snapshot (5)
| Column | Meaning |
|---|---|
| `gh_sloc` | Project size in source lines |
| `gh_test_lines_per_kloc` | Test density |
| `gh_test_cases_per_kloc` | Test case density |
| `gh_asserts_cases_per_kloc` | Assertion density |
| `gh_team_size` | Contributors active in the last 3 months |

### Context (4)
| Column | Meaning |
|---|---|
| `gh_is_pr` | PR build vs direct push — meaningfully different failure rates |
| `gh_by_core_team_member` | Author is a core contributor |
| `git_branch` | Not used raw — derive `is_default_branch` |
| `gh_lang` | Not used raw — derive `lang_java` (only Ruby/Java/JS present) |

### Keys and derivation inputs (2)
| Column | Use |
|---|---|
| `gh_build_started_at` | Temporal split, all time features, all rolling windows |
| `tr_build_number` | Sequence position within project (optional) |

### Special case
| Column | Note |
|---|---|
| `tr_prev_build` | 28.4% missing. Do **not** feed the raw build ID. Use it only to confirm build ordering; the notebook derives `prev_build_failed` by sorting within project and shifting, which is more robust. |

---

## CATEGORY 5 — ENGINEER: the features that actually carry the signal

Raw columns alone give you **AUC 0.644**. Adding the derived history features below
takes you to **AUC 0.881**. This is measured on your data, not theoretical.

### Derived ratios (8)
| Feature | Formula |
|---|---|
| `total_churn` | src_churn + test_churn |
| `total_files_changed` | added + deleted + modified |
| `test_to_src_churn_ratio` | test_churn / (src_churn + 1) |
| `churn_per_file` | total_churn / (files_changed + 1) |
| `churn_relative_to_sloc` | total_churn / (gh_sloc + 1) |
| `is_test_only_change` | no src files touched, tests changed |
| `is_doc_only_change` | only docs touched |
| `tests_touched` | any test file added/deleted |

### History — YOUR STRONGEST PREDICTORS (8)
Computed per project, in time order, with `.shift(1)` applied **before** any rolling
window so a build never sees its own outcome.

| Feature | Definition |
|---|---|
| `prev_build_failed` | **Single most important feature by a wide margin** |
| `consecutive_prior_failures` | Run length of failures ending at the previous build |
| `failure_rate_last_5` | Rolling mean over previous 5 builds |
| `failure_rate_last_20` | Rolling mean over previous 20 builds |
| `project_cum_failure_rate` | Expanding mean over all prior builds |
| `builds_so_far_in_project` | Project maturity proxy |
| `hours_since_last_build` | Cadence |
| `builds_in_last_24h` | Recent activity burst |

### Temporal / context (7)
`hour_of_day`, `day_of_week`, `is_weekend`, `month`, `is_default_branch`, `lang_java`,
plus `_was_missing` indicator flags for each imputed column.

**Total: 45 model features** — well past the 25+ your proposal promises.

---

## Measured feature importance (XGBoost, your data)

| Rank | Feature | Gain |
|---|---|---|
| 1 | `prev_build_failed` | 0.626 |
| 2 | `consecutive_prior_failures` | 0.140 |
| 3 | `failure_rate_last_5` | 0.058 |
| 4 | `failure_rate_last_20` | 0.031 |
| 5 | `is_default_branch` | 0.011 |
| 6 | `project_cum_failure_rate` | 0.010 |
| 7 | `gh_is_pr` | 0.009 |
| 8 | `git_num_all_built_commits` | 0.007 |
| 9 | `hours_since_last_build` | 0.007 |
| 10 | `builds_so_far_in_project` | 0.006 |

The top four are all history features and together account for ~86% of total gain.

**This is your headline answer to RQ1**, and it is a more interesting finding than
"code churn predicts failure," which is what everyone expects. Frame it as: *pipeline
state dominates change characteristics*. Failure is autocorrelated — a broken build
tends to stay broken until someone fixes it. Discuss the practical implication: a
useful gate needs pipeline history, so a brand-new project with no history is the
hardest case, which is exactly what your cross-project split will show.

Caveat for the write-up: use **SHAP values**, not raw gain, for your final RQ1 answer.
Gain is biased toward high-cardinality features. SHAP is the defensible choice and gives
you directionality (does high churn push risk up or down?), which gain does not.

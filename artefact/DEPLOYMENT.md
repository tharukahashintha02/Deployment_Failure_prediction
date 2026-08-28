# Deployment Guide

## 1. Create the database (Neon)

Neon's free tier does not expire, unlike Render's free Postgres which is deleted
after 30 days. Given a project running to November, Neon is the right choice.

1. Sign up at https://neon.tech (GitHub sign-in works).
2. Create a project — name it `cicd-risk-gate`, any region.
3. On the dashboard, copy the **connection string**. It looks like:

   postgresql://user:password@ep-xxx-123456.us-east-2.aws.neon.tech/neondb?sslmode=require

4. Keep it somewhere safe. It contains a password — never commit it.

## 2. Deploy the service (Render)

New + -> Web Service -> select your repository.

| Field | Value |
|---|---|
| Name | `cicd-risk-gate` |
| Branch | `main` |
| Root Directory | `artefact` |
| Runtime / Language | **Docker** |
| Instance Type | Free |

Root Directory is the field most often missed: the Dockerfile lives in
`artefact/`, not at the repository root.

## 3. Environment variables

Under **Environment**, add:

| Key | Value |
|---|---|
| `DATABASE_URL` | your Neon connection string |
| `POLICY_MODE` | `shadow` |

`DATABASE_URL` is what switches the history store from ephemeral SQLite to
persistent Postgres. Without it the service still runs, but every project
resets to cold start on each restart — and the free tier restarts often.

Do not add a persistent disk. Disks are a paid feature on Render, and the
external database removes the need for one.

## 4. Verify

Watch the deploy log. Expect:

    Model loaded: xgb-v1.0 | threshold=0.800 | policy=shadow
    History store: postgresql | connected=True | persistent=True
    INFO: Uvicorn running on http://0.0.0.0:8000

Then open `https://<your-service>.onrender.com/health`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "storage": { "backend": "postgresql", "connected": true, "persistent": true }
}
```

If `persistent` is `false`, `DATABASE_URL` is not set or is malformed.
If `status` is `degraded`, either the model failed to load or the database is
unreachable — the `storage` block distinguishes the two.

## 5. Demonstrate persistence

This is worth capturing for the dissertation, since it evidences the design
decision described in Section 4.3.

1. Record several outcomes via `POST /outcome`.
2. Call `POST /predict` and note the risk score.
3. In Render, **Manual Deploy -> Clear build cache & deploy** (a full restart).
4. Call `POST /predict` again with the same payload.

The risk score should be unchanged and `builds_in_history` should be preserved.
On the ephemeral filesystem it would have reset to zero.

## 6. Wire up GitHub Actions

1. Copy `artefact/.github/workflows/risk-gate.yml` to `.github/workflows/` at
   the **repository root** — Actions does not look inside subdirectories.
2. Repository -> Settings -> Secrets and variables -> Actions -> New secret:
   `PREDICT_URL` = `https://<your-service>.onrender.com` (no trailing slash).
3. Edit the "Run build and tests" step to your real build command.

## Known limitations to document

- **Cold start latency.** Render's free tier sleeps after 15 minutes idle;
  the first request afterwards takes roughly 30 seconds. Report this as a
  deployment limitation rather than a property of the model.
- **Fail-open gating.** If the service is unreachable the workflow allows the
  build with a warning. A gate that blocks all delivery when it goes down is
  worse than no gate.
- **Cold-start predictions.** A project with no recorded history is scored on
  code features alone. The response flags this via `history_available: false`.

## Local development

No `DATABASE_URL` needed — the store falls back to SQLite automatically:

    python -m uvicorn app.main:app --reload

To test the Postgres path locally, set `DATABASE_URL` to your Neon string.
The test suite passes identically on both backends.

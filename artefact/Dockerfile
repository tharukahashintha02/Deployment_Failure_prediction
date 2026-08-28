FROM python:3.11-slim

# Hugging Face Spaces runs containers as UID 1000, not root, so the working
# directory must be owned by that user. This is harmless on Render/Koyeb.
RUN useradd -m -u 1000 appuser

WORKDIR /code

# libgomp1 is required by the xgboost and shap wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN mkdir -p /code/data && chown -R appuser:appuser /code

USER appuser

# PORT is honoured if the platform sets it (Render, Koyeb, Fly all do).
# 7860 is the default because Hugging Face Spaces expects it and sets nothing.
ENV MODEL_DIR=/code/app/model \
    DB_PATH=/code/data/history.db \
    POLICY_MODE=shadow \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os,urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.getenv('PORT','7860')}/health\").status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

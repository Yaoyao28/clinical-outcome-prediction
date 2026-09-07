# ---- Inference image for the ICU mortality API -------------------------------
# Build:  docker build -t icu-mortality-api .
# Run:    docker run --rm -p 8000:8000 icu-mortality-api
# Test:   curl http://localhost:8000/health
#
# Python 3.12 matches the CI environment. Keep in sync with .github/workflows/tests.yml.
FROM python:3.12-slim

# XGBoost needs libgomp; curl is for the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what inference needs. .dockerignore keeps notebooks/data/results out.
COPY src/ src/
COPY app/ app/
COPY models/final_xgboost_pipeline.joblib models/final_feature_config.json models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

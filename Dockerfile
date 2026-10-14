# Readmission-risk API. Build it AFTER training (the model file is not in git):
#   python -m readmission.train
#   docker build -t readmission-api .
#   docker run --rm -p 8000:8000 readmission-api
# Python matches the version the model was trained with; `constraints.txt` pins the libraries,
# because a model saved with one scikit-learn / LightGBM version must be read by the same one.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    READMISSION_MODEL_PATH=/app/models/model.joblib

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY constraints.txt pyproject.toml README.md ./
COPY src ./src
# No dev tools, no SHAP, no matplotlib: only what serving needs.
RUN pip install -c constraints.txt .

COPY models/model.joblib ./models/model.joblib

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "readmission.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

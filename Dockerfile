# ============================================
# E-Commerce Churn Prediction – Dockerfile
# ============================================
# Multi-stage build:
#   Stage 1 – install Python dependencies
#   Stage 2 – lean runtime image
# ============================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile C-extension wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install only the runtime-relevant packages into a virtual-env
# (skip heavy dev/test/jupyter deps to keep the image small)
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        pandas numpy scipy scikit-learn \
        xgboost lightgbm imbalanced-learn \
        mlflow pydantic pydantic-settings \
        fastapi uvicorn python-multipart httpx \
        joblib pyyaml python-dotenv requests tqdm

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

# libgomp is required at runtime by LightGBM / XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Bring the virtual-env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY deploy.py       .
COPY server.py       .
COPY train_local.py  .
COPY index.html      .
COPY sample_customers.csv .
COPY models/         ./models/

# Copy pre-trained model artifacts (if any exist)
COPY artifacts/      ./artifacts/

# Default port if not supplied by Heroku
ENV PORT=8000
EXPOSE $PORT

# Start FastAPI server
CMD ["sh", "-c", "uvicorn deploy:app --host 0.0.0.0 --port ${PORT:-8000}"]

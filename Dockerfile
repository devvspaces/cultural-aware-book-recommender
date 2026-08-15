# ─── Stage 1: Build React Frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install --quiet
COPY ui/ ./
RUN npm run build

# ─── Stage 2: Python Backend Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System dependencies for C-extensions (scikit-surprise compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and data
COPY . .
COPY --from=frontend-builder /app/ui/dist ./ui/dist

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Start FastAPI server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "."]

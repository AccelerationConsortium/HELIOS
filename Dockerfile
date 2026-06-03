# ============================================================
# HELIOS — Multi-agent laboratory orchestrator
# ============================================================
# Build variants:
#   Default (simulated):  docker build -t helios .
#   With hardware:        docker build --build-arg EXTRAS=hardware -t helios:hw .
#   With ML (DQN):        docker build --build-arg EXTRAS=ml -t helios:ml .
#   Full:                 docker build --build-arg EXTRAS=all -t helios:full .
# ============================================================

FROM python:3.11-slim

ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -n "$EXTRAS" ]; then \
        pip install --no-cache-dir ".[$EXTRAS]"; \
    else \
        pip install --no-cache-dir .; \
    fi

# Copy application code
COPY app/ ./app/

# Create data directory (volume-mounted in production)
RUN mkdir -p /app/data

# Default environment variables
ENV ADAPTER_MODE=simulated \
    ADAPTER_DRY_RUN=true \
    WORKSPACE_ROOT=/app \
    DATA_DIR=/app/data \
    DB_PATH=/app/data/orchestrator.db \
    LLM_PROVIDER=mock \
    ROBOT_IP=localhost \
    RELAY_PORT=auto \
    SQUIDSTAT_PORT=auto

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Traffic Violation Detection System
# Dockerfile — CPU-only, production-ready
# ============================================

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than CUDA variant)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──
FROM python:3.11-slim AS runtime

LABEL maintainer="Flipkart Grid R2 Team"
LABEL description="Traffic Violation Detection API Server"

# Runtime system deps for OpenCV, EasyOCR, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure output & models directories exist
RUN mkdir -p /app/output /app/models && chown -R appuser:appuser /app

# EasyOCR downloads models on first run — cache them in a volume-friendly location
ENV EASYOCR_MODULE_PATH=/app/.easyocr
RUN mkdir -p /app/.easyocr && chown appuser:appuser /app/.easyocr

USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start the FastAPI server
CMD ["python", "run.py", "--server"]

# Multi-stage build for the DICOM Processing Pipeline
# Optimized for production use with minimal image size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-103 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r pipeline && useradd -r -g pipeline -m pipeline

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create directories for data and logs
RUN mkdir -p /data/input /data/output /data/checkpoints /logs \
    && chown -R pipeline:pipeline /app /data /logs

# Switch to non-root user
USER pipeline

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIPELINE_LOG_DIR=/logs \
    PIPELINE_CHECKPOINT_DIR=/data/checkpoints

# Default command
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--config", "configs/mammography_pipeline.yaml", "--input", "/data/input", "--output", "/data/output"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import src; print('ok')" || exit 1

# Labels
LABEL maintainer="Manuel" \
      description="Scalable DICOM Processing Pipeline for Medical Imaging ML" \
      version="2.0.0"

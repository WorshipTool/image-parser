# Image Parser Server - Docker Image
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps for tesseract + opencv headless runtime and git-based pip deps
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        tesseract-ocr

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Replace opencv-python with opencv-python-headless to avoid libGL dependency
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp/uploads temp/final_corrections temp/line_corrections temp/segmentation_debug

# Expose port
EXPOSE 6610

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check - verify server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6610/is-available')" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "server"]

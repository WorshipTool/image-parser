# Image Parser Server - Docker Image
# Use existing working image and just replace OpenCV
FROM image-parser-server:latest

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

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "server"]

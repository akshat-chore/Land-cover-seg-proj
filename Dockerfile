FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create workdir for temporary files
RUN mkdir -p /app/workdir

# Expose port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_CHECKPOINT=models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth
ENV DEVICE=cuda
ENV NUM_CLASSES=5
ENV ENCODER=efficientnet-b0
ENV ENCODER_WEIGHTS=imagenet
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start FastAPI server with uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

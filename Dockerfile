# Dockerfile
FROM python:3.11-slim

# Install system dependencies needed for PyTorch and decord
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Default environment variables
ENV APP_PORT=7860
ENV HF_HOME=/data/hf_cache

# Expose application port
EXPOSE ${APP_PORT}

# Run the application
CMD ["python", "app.py"]

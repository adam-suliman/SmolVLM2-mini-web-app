FROM python:3.11-slim

ENV APP_PORT=7860 \
    HF_HOME=/data/hf_cache \
    HF_HUB_CACHE=/data/hf_cache/hub \
    TRANSFORMERS_CACHE=/data/hf_cache/hub \
    HF_DATASETS_CACHE=/data/hf_cache/datasets \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

RUN mkdir -p /data/hf_cache /data/hf_cache/hub /data/hf_cache/datasets

EXPOSE 7860

CMD ["python", "app.py"]

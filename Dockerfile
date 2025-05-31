FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set environment variables for AI models
ENV TRANSFORMERS_CACHE=/tmp/model_cache
ENV HF_HOME=/tmp/model_cache
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

# Create cache directory
RUN mkdir -p /tmp/model_cache

# Copy application
COPY . .

EXPOSE 5000

# Optimized gunicorn command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
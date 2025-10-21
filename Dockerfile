FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs

# Expose port for model serving
EXPOSE 5000

# Default command
CMD ["python", "src/models/train.py"]
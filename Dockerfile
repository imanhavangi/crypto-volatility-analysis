# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Health check
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('live_state.json') else 1)"

# Run the live signal runner
CMD ["python", "-u", "live_signal_runner.py"]


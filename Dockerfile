# Usa Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies se necessarie
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first per Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)" || true

# Copy everything else in the app
COPY . .

# Create directory with correct permission for database e cache
RUN mkdir -p /app/data && chmod 777 /app/data
RUN mkdir -p /app/huggingface_cache && chmod 777 /app/huggingface_cache

# Create empty file database with permission
RUN touch /app/data/conversations.db && chmod 666 /app/data/conversations.db

# Set enviroment variables
ENV HF_HOME=/app/huggingface_cache
ENV DB_PATH=/app/data/conversations.db
ENV DEPLOYMENT_ENV=huggingface

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Health check opzionale
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start the Flask app
CMD ["python", "app.py"]
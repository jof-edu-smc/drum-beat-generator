# ==========================================
# Build the Flask Backend & Final Image
# ==========================================

# Use an official PyTorch runtime image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# FROM python:3.11-slim AS backend-runner

WORKDIR /app

# Prevent Python from writing .pyc files and enable buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Install Python dependencies
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install --no-cache-dir gunicorn

# Copy backend files and application data

COPY app.py ./
COPY models/ ./models/
COPY utilities/ ./utilities/
COPY index.html ./

EXPOSE 5001

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]
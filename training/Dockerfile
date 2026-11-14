# ==========================================
# Build the model training image
# ==========================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app

# Prevent Python from writing .pyc files and enable buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt 

# Copy the training package and Vertex entrypoints.
COPY main.py ./
COPY datasets/ ./datasets/
COPY models/ ./models/
COPY utilities/ ./utilities/

EXPOSE 8080

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]
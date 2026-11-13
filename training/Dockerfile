# ==========================================
# Build the model training image
# ==========================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app

# Prevent Python from writing .pyc files and enable buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Install Python dependencies
COPY training/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt 

# Copy the training package and Vertex entrypoints.
COPY training/main.py ./
COPY training/datasets/ ./datasets/
COPY training/models/ ./models/
COPY training/utilities/ ./utilities/

ENTRYPOINT ["python", "main.py"]
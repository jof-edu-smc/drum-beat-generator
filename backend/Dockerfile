# ==========================================
# Build the Flask Backend & Final Image
# ==========================================

FROM python:3.14-slim AS backend-runner
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

EXPOSE 8080

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
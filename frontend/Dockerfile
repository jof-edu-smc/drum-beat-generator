# ==========================================
# Build the Vite Frontend
# ==========================================

FROM node:20-alpine AS frontend-builder

WORKDIR /app

# Copy package files and install frontend dependencies
COPY package*.json ./
RUN npm ci

# Copy the frontend configuration and source code
COPY public/ /public
COPY vite.config.js index.html ./
COPY src/ ./src/

# Build the static assets (outputs to /app/dist by default)
RUN npm run build
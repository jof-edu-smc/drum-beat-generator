# ==========================================
# Build the Vite Frontend
# ==========================================

FROM node:20-alpine AS frontend-builder

WORKDIR /app

# Copy package files and install frontend dependencies
COPY package*.json ./
RUN npm ci

# Copy the frontend configuration and source code
COPY public/ ./public/
COPY vite.config.js index.html ./
COPY src/ ./src/

# Define the build argument
ARG VITE_GOOGLE_CLIENT_ID
ARG VITE_GCS_AUDIO_BASE_URL
ARG VITE_BACKEND_URL

# Expose them to the build environment
ENV VITE_GOOGLE_CLIENT_ID=$VITE_GOOGLE_CLIENT_ID
ENV VITE_GCS_AUDIO_BASE_URL=$VITE_GCS_AUDIO_BASE_URL
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

# Build the static assets (outputs to /app/dist by default)
RUN npm run build

FROM nginx:1.27-alpine AS frontend-runtime

COPY --from=frontend-builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
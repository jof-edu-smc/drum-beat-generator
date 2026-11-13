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

# Build the static assets (outputs to /app/dist by default)
RUN npm run build

FROM nginx:1.27-alpine AS frontend-runtime

COPY --from=frontend-builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
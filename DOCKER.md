# 🐳 Docker Guide - Bike Sharing Demand Prediction Service

This guide explains how to build, run, and deploy the ML service using Docker.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building the Image](#building-the-image)
- [Running the Container](#running-the-container)
- [Docker Compose](#docker-compose)
- [Publishing to Docker Hub](#publishing-to-docker-hub)
- [Versioning Strategy](#versioning-strategy)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, for development)
- Trained model files in `models/` directory
- At least 2GB of free disk space

---

## Quick Start

### Option 1: Using Docker Compose (Recommended for Development)

```bash
# Build and start all services (ML service + Redis)
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f ml-service

# Stop services
docker-compose down
```

The service will be available at: `http://localhost:8000`

### Option 2: Using Docker Commands

```bash
# Build the image
docker build -t ml-service:latest .

# Run the container
docker run -p 8000:8000 ml-service:latest
```

---

## Building the Image

### Basic Build

```bash
docker build -t ml-service:latest .
```

### Build with Specific Tag

```bash
docker build -t ml-service:v1.0.0 .
docker build -t ml-service:v1.0.0 -t ml-service:latest .
```

### Build Arguments (if needed)

The Dockerfile uses a multi-stage build for efficiency:
- **Stage 1 (builder)**: Installs all dependencies
- **Stage 2 (runtime)**: Minimal production image with only runtime dependencies

### Build Optimization

- Uses layer caching: `requirements.txt` is copied first
- Multi-stage build reduces final image size
- Non-root user for security
- Health check included

---

## Running the Container

### Basic Run

```bash
docker run -p 8000:8000 ml-service:latest
```

### Run with Environment Variables

```bash
docker run -p 8000:8000 \
  -e MODEL_NAME=xgboost_baseline \
  -e REDIS_HOST=host.docker.internal \
  -e REDIS_PORT=6379 \
  ml-service:latest
```

### Run with Volume Mounts (for model updates)

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  ml-service:latest
```

### Run in Detached Mode

```bash
docker run -d -p 8000:8000 --name bike-sharing-service ml-service:latest
```

### View Logs

```bash
docker logs bike-sharing-service
docker logs -f bike-sharing-service  # Follow logs
```

### Stop Container

```bash
docker stop bike-sharing-service
docker rm bike-sharing-service
```

---

## Docker Compose

The `docker-compose.yml` file provides a complete setup with:

- **ML Service**: FastAPI application
- **Redis**: For prediction caching and historical features

### Commands

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# Rebuild and start
docker-compose up --build

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs -f ml-service
docker-compose logs -f redis

# Restart a service
docker-compose restart ml-service

# Scale services (if needed)
docker-compose up -d --scale ml-service=3
```

### Environment Variables in docker-compose.yml

You can set environment variables in `docker-compose.yml` or use a `.env` file:

```yaml
environment:
  - MODEL_NAME=xgboost_baseline
  - REDIS_HOST=redis
  - REDIS_PORT=6379
```

---

## Publishing to Docker Hub

### 1. Login to Docker Hub

```bash
docker login
# Enter your Docker Hub username and password
```

### 2. Tag the Image

```bash
# Tag with your Docker Hub username
docker tag ml-service:latest YOUR_USERNAME/ml-service:latest
docker tag ml-service:latest YOUR_USERNAME/ml-service:v1.0.0
```

### 3. Push to Docker Hub

```bash
# Push latest
docker push YOUR_USERNAME/ml-service:latest

# Push versioned tag
docker push YOUR_USERNAME/ml-service:v1.0.0
```

### 4. Pull and Run from Docker Hub

```bash
docker pull YOUR_USERNAME/ml-service:latest
docker run -p 8000:8000 YOUR_USERNAME/ml-service:latest
```

---

## Versioning Strategy

### Tag Naming Convention

We use semantic versioning for container tags:

- `latest`: Always points to the most recent stable version
- `v1.0.0`: Specific version (major.minor.patch)
- `v1.0.0-beta`: Pre-release versions
- `v1.0.0-rc1`: Release candidates

### Example Workflow

```bash
# Build version 1.0.0
docker build -t ml-service:v1.0.0 -t ml-service:latest .

# Tag for Docker Hub
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:v1.0.0
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:latest

# Push both tags
docker push YOUR_USERNAME/ml-service:v1.0.0
docker push YOUR_USERNAME/ml-service:latest
```

### Recommended Tags per Release

```bash
# Major release
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:v1
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:v1.0
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:v1.0.0
docker tag ml-service:v1.0.0 YOUR_USERNAME/ml-service:latest

# Push all tags
docker push YOUR_USERNAME/ml-service:v1
docker push YOUR_USERNAME/ml-service:v1.0
docker push YOUR_USERNAME/ml-service:v1.0.0
docker push YOUR_USERNAME/ml-service:latest
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | Auto-detected | Name of the model to load (e.g., `xgboost_baseline`) |
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `PORT` | `8000` | Port for the FastAPI service |

### Setting Environment Variables

**In docker-compose.yml:**
```yaml
environment:
  - MODEL_NAME=xgboost_baseline
  - REDIS_HOST=redis
```

**In docker run:**
```bash
docker run -p 8000:8000 -e MODEL_NAME=xgboost_baseline ml-service:latest
```

**Using .env file:**
```bash
# .env
MODEL_NAME=xgboost_baseline
REDIS_HOST=redis
```

Then in docker-compose.yml:
```yaml
env_file:
  - .env
```

---

## Troubleshooting

### Container Won't Start

1. **Check logs:**
   ```bash
   docker logs bike-sharing-service
   ```

2. **Verify model files exist:**
   ```bash
   ls -la models/*.pkl
   ```

3. **Check port availability:**
   ```bash
   lsof -i :8000
   ```

### Model Not Found Error

- Ensure model files are in `models/` directory
- Check that model name matches the file (e.g., `xgboost_baseline.pkl`)
- Verify `MODEL_NAME` environment variable if set

### Redis Connection Issues

- Ensure Redis is running: `docker-compose up redis`
- Check Redis host/port configuration
- Service will work without Redis, but caching will be disabled

### Health Check Failing

- Wait a few seconds for the service to start
- Check service logs for errors
- Verify the service is listening on port 8000

### Image Size Too Large

- The multi-stage build should keep the image small (~500MB-1GB)
- If larger, check for unnecessary files in the build context
- Review `.dockerignore` to exclude more files

---

## Production Deployment

### Recommended Settings

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ml-service:
    image: YOUR_USERNAME/ml-service:v1.0.0
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    environment:
      - MODEL_NAME=xgboost_baseline
```

### Security Best Practices

1. ✅ Non-root user in container
2. ✅ Read-only volume mounts where possible
3. ✅ Health checks enabled
4. ✅ Resource limits
5. ✅ Use specific version tags (not `latest`) in production

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project README](../README.md)

---

**Last Updated:** January 2025  
**Maintainer:** Team 61 - MLOps Project


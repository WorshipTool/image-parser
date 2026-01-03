# Docker Deployment Guide

Complete guide for running the Image Parser Server in Docker containers.

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd image-parser

# 2. Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Build and start services
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Access server
open http://localhost:6610/docs
```

## Architecture

The Docker setup includes 3 services:

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Compose                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Server     │  │   Worker     │  │    Redis     │  │
│  │  (Flask)     │  │    (RQ)      │  │   (Queue)    │  │
│  │              │  │              │  │              │  │
│  │  Port: 6610  │  │  Background  │  │  Port: 6379  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │          │
│         └──────────────────┴──────────────────┘          │
│                     Shared Network                       │
└─────────────────────────────────────────────────────────┘
```

### Services

**1. Server** (`image-parser-server`)
- Flask HTTP API
- Handles requests and queues jobs
- Port: 6610

**2. Worker** (`image-parser-worker`)
- RQ worker process
- Processes background jobs
- No exposed ports

**3. Redis** (`image-parser-redis`)
- Job queue database
- Port: 6379

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key

# Optional - Server
PORT=6610
HOST=0.0.0.0

# Optional - Service Discovery (requires wt-bridge-module)
# Install: pip install git+https://github.com/WorshipTool/wt-bridge-module-python.git
BRIDGE_URL=http://bridge-service:5555
BRIDGE_SERVICE_NAME=docker-parser

# Optional - Redis (defaults work with docker-compose)
REDIS_HOST=redis
REDIS_PORT=6379
```

**Note:** The bridge module for service discovery is optional. If not installed, the server will run normally without service discovery features.

### Volume Mounts

```yaml
volumes:
  - ./temp:/app/temp          # Temporary files and uploads
  - ./parser:/app/parser      # Parser module (for development)
  - ./server:/app/server      # Server module (for development)
  - ./ai:/app/ai              # AI module (for development)
```

## Commands

### Start Services

```bash
# Start all services in background
docker-compose up -d

# Start with logs
docker-compose up

# Start specific service
docker-compose up -d server
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f server
docker-compose logs -f worker
docker-compose logs -f redis
```

### Rebuild After Code Changes

```bash
# Rebuild images
docker-compose build

# Rebuild and restart
docker-compose up -d --build

# Rebuild specific service
docker-compose build server
```

### Execute Commands in Container

```bash
# Access server container shell
docker-compose exec server bash

# Access worker container shell
docker-compose exec worker bash

# Run parser CLI in container
docker-compose exec server python main.py image.jpg --ai
```

## Development Workflow

### Live Code Updates

The docker-compose setup mounts source directories as volumes, so code changes are reflected immediately **without rebuilding**:

```bash
# Edit code locally
vim parser/parse.py

# Restart services to pick up changes
docker-compose restart server worker
```

### Testing Changes

```bash
# Check server health
curl http://localhost:6610/is-available

# Test file upload
curl -X POST http://localhost:6610/parse-file \
  -F "file=@test-image.jpg"

# View worker logs
docker-compose logs -f worker
```

## Production Deployment

### Build Production Image

```bash
# Build optimized image
docker build -t image-parser:latest .

# Tag for registry
docker tag image-parser:latest registry.example.com/image-parser:latest

# Push to registry
docker push registry.example.com/image-parser:latest
```

### Production docker-compose.yml

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: always

  server:
    image: registry.example.com/image-parser:latest
    ports:
      - "6610:6610"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_HOST=redis
    depends_on:
      - redis
    restart: always
    command: python -m server

  worker:
    image: registry.example.com/image-parser:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_HOST=redis
    depends_on:
      - redis
    restart: always
    deploy:
      replicas: 2  # Multiple workers for parallel processing
    command: python -m server.worker

volumes:
  redis-data:
```

### Scaling Workers

```bash
# Scale to 3 workers
docker-compose up -d --scale worker=3

# Check running workers
docker-compose ps worker
```

## Monitoring

### Health Checks

```bash
# Server health
curl http://localhost:6610/is-available

# Redis health
docker-compose exec redis redis-cli ping

# RQ Dashboard (if enabled)
open http://localhost:6610/board
```

### Container Stats

```bash
# Resource usage
docker stats

# Specific container
docker stats image-parser-server
```

### Job Queue Status

```bash
# Access RQ dashboard
open http://localhost:6610/board

# Or check via Redis CLI
docker-compose exec redis redis-cli
> LLEN rq:queue:parse_files
> KEYS rq:job:*
```

## Troubleshooting

### Server won't start

```bash
# Check logs
docker-compose logs server

# Common issues:
# 1. Port already in use
lsof -i :6610

# 2. Missing .env file
cp .env.example .env

# 3. Invalid OPENAI_API_KEY
docker-compose exec server env | grep OPENAI_API_KEY
```

### Worker not processing jobs

```bash
# Check worker logs
docker-compose logs worker

# Check Redis connection
docker-compose exec worker python -c "import redis; r=redis.Redis(host='redis'); print(r.ping())"

# Restart worker
docker-compose restart worker
```

### Redis connection failed

```bash
# Check Redis is running
docker-compose ps redis

# Check Redis logs
docker-compose logs redis

# Test connection from server
docker-compose exec server redis-cli -h redis ping
```

### Out of disk space

```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a

# Remove old images
docker image prune -a
```

### Permission issues with temp files

```bash
# Fix permissions on host
chmod -R 777 temp/

# Or run container as specific user
docker-compose run --user $(id -u):$(id -g) server python main.py
```

## Performance Optimization

### Resource Limits

Add to docker-compose.yml:

```yaml
services:
  server:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Caching Dependencies

The Dockerfile uses layer caching for faster rebuilds:

```dockerfile
# Requirements cached separately from code
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code changes don't require reinstalling deps
COPY . .
```

### Network Optimization

Use Docker networks for inter-service communication:

```yaml
networks:
  parser-network:
    driver: bridge

services:
  server:
    networks:
      - parser-network
```

## Security

### Best Practices

1. **Don't commit .env**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use secrets for production**
   ```yaml
   services:
     server:
       secrets:
         - openai_api_key
   ```

3. **Run as non-root user**
   ```dockerfile
   RUN useradd -m parser
   USER parser
   ```

4. **Scan images for vulnerabilities**
   ```bash
   docker scan image-parser:latest
   ```

## Backup and Recovery

### Backup Redis Data

```bash
# Create backup
docker-compose exec redis redis-cli SAVE
docker cp image-parser-redis:/data/dump.rdb ./backup/

# Restore from backup
docker cp ./backup/dump.rdb image-parser-redis:/data/
docker-compose restart redis
```

### Export Temp Files

```bash
# Backup temp directory
tar -czf temp-backup.tar.gz temp/

# Restore
tar -xzf temp-backup.tar.gz
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build image
        run: docker build -t image-parser:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
          docker push image-parser:${{ github.sha }}
```

## Additional Resources

- **Docker Documentation**: https://docs.docker.com
- **Docker Compose Reference**: https://docs.docker.com/compose/compose-file/
- **RQ Documentation**: https://python-rq.org
- **Server API Docs**: http://localhost:6610/docs (when running)

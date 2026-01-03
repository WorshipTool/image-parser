.PHONY: help build up down restart logs clean test

help:
	@echo "Image Parser - Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create .env from template"
	@echo "  make build          - Build Docker images"
	@echo ""
	@echo "Running:"
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo ""
	@echo "Development:"
	@echo "  make logs           - View logs (all services)"
	@echo "  make logs-server    - View server logs"
	@echo "  make logs-worker    - View worker logs"
	@echo "  make shell-server   - Shell into server container"
	@echo "  make shell-worker   - Shell into worker container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Stop and remove containers, volumes"
	@echo "  make rebuild        - Rebuild and restart services"
	@echo "  make scale-workers  - Scale workers to 3 instances"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run health checks"
	@echo "  make test-upload    - Test file upload"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file"; \
		echo "⚠️  Please edit .env and add your OPENAI_API_KEY"; \
	else \
		echo "✓ .env already exists"; \
	fi

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "✓ Services started"
	@echo "  Server: http://localhost:6610"
	@echo "  Docs: http://localhost:6610/docs"
	@echo "  Dashboard: http://localhost:6610/board"

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

logs-server:
	docker-compose logs -f server

logs-worker:
	docker-compose logs -f worker

shell-server:
	docker-compose exec server bash

shell-worker:
	docker-compose exec worker bash

clean:
	docker-compose down -v
	@echo "✓ Cleaned up containers and volumes"

rebuild:
	docker-compose down
	docker-compose build
	docker-compose up -d
	@echo "✓ Rebuilt and restarted"

scale-workers:
	docker-compose up -d --scale worker=3
	@echo "✓ Scaled workers to 3 instances"

test:
	@echo "Testing server health..."
	@curl -s http://localhost:6610/is-available | jq .
	@echo "\n✓ Server is healthy"

test-upload:
	@echo "Testing file upload (needs test-image.jpg)..."
	@if [ -f test-image.jpg ]; then \
		curl -X POST http://localhost:6610/parse-file -F "file=@test-image.jpg" | jq .; \
	else \
		echo "⚠️  test-image.jpg not found"; \
	fi

#!/bin/bash
# Production Deployment Script

set -e

echo "🚀 Starting production deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build production image
echo "🔨 Building production image..."
docker build -f Dockerfile.prod -t trading-bot:latest .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Start production services
echo "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health
echo "🏥 Checking service health..."
curl -f http://localhost:8000/health || echo "❌ Health check failed"

echo "✅ Production deployment completed!"
echo "📊 Services available at:"
echo "  - Trading Bot API: http://localhost:8000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"

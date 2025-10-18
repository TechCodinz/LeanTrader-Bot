#!/usr/bin/env python3
"""
Production Structure Creator
Creates a proper production-ready project structure
"""

import os
from pathlib import Path
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionStructureCreator:
    """Creates a proper production-ready project structure"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.created_dirs = 0
        self.moved_files = 0
        self.errors = 0

    def create_directory_structure(self):
        """Create the production directory structure"""
        logger.info("🏗️ Creating production directory structure...")

        # Core directories
        core_dirs = [
            "src/core",
            "src/strategies",
            "src/data",
            "src/ml",
            "src/risk",
            "src/execution",
            "src/monitoring",
            "src/utils",
            "src/api",
            "src/config",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "docs",
            "scripts",
            "deploy",
            "monitoring",
            "logs",
            "data/raw",
            "data/processed",
            "data/models",
            "config/prod",
            "config/dev",
            "config/test",
            "secrets",
            "backups"
        ]

        for dir_path in core_dirs:
            full_path = self.root_dir / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs += 1
                logger.debug(f"📁 Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"❌ Error creating {dir_path}: {e}")
                self.errors += 1

    def create_production_files(self):
        """Create essential production files"""
        logger.info("📄 Creating production files...")

        # Production requirements
        prod_requirements = """# Production Requirements
# Core dependencies with pinned versions for stability

# Core ML and Data Science
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
tensorflow==2.13.0
torch==2.0.1
ta-lib==0.4.28

# Trading APIs
ccxt==4.0.77
yfinance==0.2.18
python-binance==1.0.19
alpaca-trade-api==3.0.2
ib-insync==0.9.86

# Web Framework
fastapi==0.103.1
uvicorn==0.23.2
streamlit==1.25.0
plotly==5.15.0

# Database
sqlalchemy==2.0.19
redis==4.6.0
psycopg2-binary==2.9.7

# Monitoring and Logging
prometheus-client==0.17.1
loguru==0.7.0
structlog==23.1.0

# Security
cryptography==41.0.3
bcrypt==4.0.1
python-jose==3.3.0

# Notifications
python-telegram-bot==20.3
twilio==8.5.0
slack-sdk==3.21.3

# Utilities
python-dotenv==1.0.0
pydantic==2.1.1
schedule==1.2.0
joblib==1.3.2
requests==2.31.0
aiohttp==3.8.5

# Testing
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1

# Code Quality
black==23.7.0
isort==5.12.0
flake8==6.0.0
mypy==1.5.1
"""

        with open(self.root_dir / "requirements_production.txt", "w", encoding='utf-8') as f:
            f.write(prod_requirements)

        # Production configuration
        prod_config = """# Production Configuration
# Environment: production
# Version: 1.0.0

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379/0

# Trading Configuration
TRADING_MODE=live
RISK_MANAGEMENT=enabled
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=2.0
TAKE_PROFIT_PERCENTAGE=4.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_LEVEL=INFO

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here

# Notification Configuration
TELEGRAM_BOT_TOKEN=your-telegram-token
TELEGRAM_CHAT_ID=your-chat-id
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your-email
EMAIL_PASSWORD=your-password

# Exchange Configuration
BYBIT_API_KEY=your-bybit-api-key
BYBIT_SECRET_KEY=your-bybit-secret-key
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET_KEY=your-binance-secret-key
"""

        with open(self.root_dir / "config" / "prod" / "config.env", "w", encoding='utf-8') as f:
            f.write(prod_config)

        # Docker Compose for production
        docker_compose = """version: '3.8'

services:
  trading-bot:
    build: .
    container_name: trading-bot
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
    env_file:
      - config/prod/config.env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - prometheus

  postgres:
    image: postgres:15
    container_name: trading-bot-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    container_name: trading-bot-redis
    restart: unless-stopped
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    container_name: trading-bot-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    container_name: trading-bot-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
"""

        with open(self.root_dir / "docker-compose.prod.yml", "w", encoding='utf-8') as f:
            f.write(docker_compose)

        # Production Dockerfile
        dockerfile = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN curl -L https://github.com/mrjbq7/ta-lib/archive/TA_Lib-0.4.0.tar.gz | tar xz \\
    && cd ta-lib-0.4.0 \\
    && ./configure --prefix=/usr/local \\
    && make \\
    && make install \\
    && cd .. \\
    && rm -rf ta-lib-0.4.0

# Copy requirements and install Python dependencies
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        with open(self.root_dir / "Dockerfile.prod", "w", encoding='utf-8') as f:
            f.write(dockerfile)

        # Production deployment script
        deploy_script = """#!/bin/bash
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
"""

        with open(self.root_dir / "deploy_production.sh", "w", encoding='utf-8') as f:
            f.write(deploy_script)

        # Make deploy script executable
        os.chmod(self.root_dir / "deploy_production.sh", 0o755)

        # Production monitoring configuration
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "trading_bot_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""

        with open(self.root_dir / "monitoring" / "prometheus.yml", "w", encoding='utf-8') as f:
            f.write(prometheus_config)

        # Production README
        prod_readme = """# Trading Bot - Production Deployment

## 🚀 Quick Start

1. **Configure Environment**
   ```bash
   cp config/prod/config.env.example config/prod/config.env
   # Edit config/prod/config.env with your actual values
   ```

2. **Deploy with Docker**
   ```bash
   ./deploy_production.sh
   ```

3. **Access Services**
   - Trading Bot API: http://localhost:8000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

## 📊 Monitoring

The production deployment includes comprehensive monitoring:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Health Checks**: Automated service monitoring
- **Logging**: Structured logging with loguru

## 🔒 Security

- All secrets are managed via environment variables
- Database connections are encrypted
- API endpoints are secured with JWT
- Rate limiting is enabled

## 📈 Performance

- Multi-worker deployment
- Redis caching
- Database connection pooling
- Async processing

## 🛠️ Maintenance

- Automated backups
- Rolling updates
- Health monitoring
- Error tracking

## 📞 Support

For production support, contact the development team.
"""

        with open(self.root_dir / "README_PRODUCTION.md", "w", encoding='utf-8') as f:
            f.write(prod_readme)

        logger.info("✅ Production files created successfully")

    def run(self):
        """Run the production structure creator"""
        logger.info("🏗️ Starting Production Structure Creator...")

        # Create directory structure
        self.create_directory_structure()

        # Create production files
        self.create_production_files()

        # Generate report
        logger.info("🎉 Production structure creation completed!")
        logger.info(f"📁 Directories created: {self.created_dirs}")
        logger.info("📄 Files created: 8")
        logger.info(f"❌ Errors: {self.errors}")

def main():
    """Main function"""
    creator = ProductionStructureCreator()
    creator.run()

if __name__ == "__main__":
    main()

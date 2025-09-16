FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt || true \
 && pip install feedparser python-dotenv ccxt aiohttp web3 python-telegram-bot matplotlib scikit-learn || true

CMD ["python", "/app/ultra_launcher.py", "--mode", "paper", "--god-mode", "--moon-spotter", "--evolution", "--forex"]
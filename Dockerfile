FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m pip check \
 && PYTHONPATH=/app:/app/src python tests/test_runtime_dependencies_stdlib.py

CMD ["python", "/app/ultra_launcher.py", "--mode", "paper", "--god-mode", "--moon-spotter", "--evolution", "--forex"]
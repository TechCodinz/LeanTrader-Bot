FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu" \
 && pip install -r requirements.txt \
 && python -m pip check

COPY . /app

RUN PYTHONPATH=/app:/app/src python tests/test_runtime_dependencies_stdlib.py \
 && PYTHONPATH=/app:/app/src python tests/test_integrated_runtime_dependencies_stdlib.py \
 && PYTHONPATH=/app:/app/src python tests/test_langchain_agent_wiring_stdlib.py  && PYTHONPATH=/app:/app/src python tests/test_bybit_recv_window_stdlib.py \
 && PYTHONPATH=/app:/app/src python tests/test_ack_is_not_fill_stdlib.py

CMD ["python", "/app/ultra_launcher.py", "--mode", "paper", "--god-mode", "--moon-spotter", "--evolution", "--forex"]

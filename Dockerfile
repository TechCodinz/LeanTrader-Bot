FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

RUN groupadd --gid 10001 leantrader \
    && useradd --uid 10001 --gid leantrader --no-create-home --shell /usr/sbin/nologin leantrader

COPY requirements.runtime.txt ./
RUN pip install --no-cache-dir --requirement requirements.runtime.txt

COPY src ./src

# v1.61 restoration: original LeanTrader intelligence families live
# at repository root and in these legacy source packages.
# .dockerignore excludes secrets, runtime state, data and models.
COPY *.py ./
COPY scanners ./scanners
COPY traders_core ./traders_core
COPY execution ./execution

RUN mkdir -p /app/runtime /app/logs /app/data \
    && chown -R leantrader:leantrader /app/runtime /app/logs /app/data

USER 10001:10001

# Fail the image build if the restored production runtime cannot
# actually import the original intelligence families as the
# non-root LeanTrader user.
RUN python -c "from ultra_scalping_engine import UltraScalpingEngine; from ultra_continuous_trading import UltraContinuousTradingOrchestrator; from ultra_swarm_consciousness import SwarmAgent; from ultra_quantum_intelligence import MicrostructureDecoder, QuantumMomentumOscillator, AdaptiveMarketRegimeDetector; from scanners.arbitrage import cross_exchange_spreads; print('v1.61 legacy imports verified')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "/app/src/leantrader/production/healthcheck.py"]

CMD ["python", "-m", "leantrader.production.runner"]

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --gid 10001 leantrader \
    && useradd --uid 10001 --gid leantrader --no-create-home --shell /usr/sbin/nologin leantrader

COPY requirements.runtime.txt ./
RUN pip install --no-cache-dir --requirement requirements.runtime.txt

COPY src ./src
RUN mkdir -p /app/runtime /app/logs \
    && chown -R leantrader:leantrader /app/runtime /app/logs

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-m", "leantrader.production.healthcheck"]

CMD ["python", "-m", "leantrader.production.runner"]

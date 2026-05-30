FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY end_to_end_inference/requirements.txt /app/end_to_end_inference/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/end_to_end_inference/requirements.txt

RUN mkdir -p /app/temporal_retrieve_virality_signals/notebooks/embeddings \
    /app/temporal_retrieve_virality_signals/data
COPY end_to_end_inference /app/end_to_end_inference
COPY temporal_retrieve_virality_signals/notebooks/embeddings /app/temporal_retrieve_virality_signals/notebooks/embeddings
COPY temporal_retrieve_virality_signals/data/dataset_tg_economic.parquet /app/temporal_retrieve_virality_signals/data/dataset_tg_economic.parquet

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD curl -f http://127.0.0.1:8765/health || exit 1

CMD ["python", "-m", "end_to_end_inference.simple_retrieval_service", "--host", "0.0.0.0", "--port", "8765", "--embedding-key", "e5-small", "--encoder-path", "/models/multilingual-e5-small"]

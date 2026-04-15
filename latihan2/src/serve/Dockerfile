FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

FROM base AS deps
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY src/serve/requirements-serve.txt /tmp/req.txt
RUN pip install --prefix=/install -r /tmp/req.txt

FROM base AS runtime
WORKDIR /app
COPY --from=deps /install /usr/local
COPY src /app/src
COPY data/processed/dl_artifacts /app/data/processed/dl_artifacts
COPY data/processed/movies.parquet /app/data/processed/movies.parquet
COPY data/processed/nlp /app/data/processed/nlp
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]

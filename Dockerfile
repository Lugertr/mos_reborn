# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Системные пакеты: tesseract + языки, и либы для opencv/tf
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — кэшируем слой
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Затем только исходники
# (data/ и runs/ мы намеренно НЕ копируем — они будут томами)
COPY api/ ./api/
COPY preprocessing/ ./preprocessing/
COPY postprocessing/ ./postprocessing/
COPY tesseract/ ./tesseract/
COPY trocr/ ./trocr/
COPY config.py ./config.py
COPY logging_utils.py ./logging_utils.py
COPY schemas.py ./schemas.py
COPY main.py ./main.py

# Непривилегированный пользователь
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
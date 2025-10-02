# ===== Backend (FastAPI + Tesseract + TrOCR) =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Системные зависимости для Tesseract
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY main.py config.py logging_utils.py schemas.py /app/
COPY api/ /app/api/
COPY preprocessing/ /app/preprocessing/
COPY postprocessing/ /app/postprocessing/
COPY tesseract/ /app/tesseract/
COPY trocr/ /app/trocr/

# ВАЖНО: запекаем веса внутрь образа (оставьте эту строку!)
# Убедитесь, что локально есть каталог ./runs/default с charset.json, config.json, *.weights.h5
COPY runs/ /app/runs/

# env по умолчанию (можно переопределить)
ENV TROCR_ENABLED=true \
    TROCR_RUN_DIR=/app/runs/default \
    TROCR_WEIGHTS_FILE=best.weights.h5

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]

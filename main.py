"""
Purpose
-------
Точка входа FastAPI-сервиса OCR.

Что делает
----------
- Регистрирует маршруты:
    • OCR API (если модуль доступен) — загрузка изображения и распознавание построчно.
    • Train API — запуск/статус обучения/дообучения TrOCR.
- Включает CORS для фронтенда (origin'ы берутся из config.ALLOWED_ORIGINS).
- При старте (startup) опционально прогревает TrOCR (если доступен trocr.runtime.init_trocr).
- Предоставляет простой health-чек `/health`.

Где что настраивать
-------------------
- Разрешённые источники (CORS): config.ALLOWED_ORIGINS.
- Параметры TrOCR и пути к весам: см. config.py и trocr/runtime.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from logging_utils import setup_logger

# ── OCR роутер (может отсутствовать в окружении сборки; тогда сервис поднимется без OCR ручек)
try:
    from api.routes import router as ocr_router  # опционально
except Exception:
    ocr_router = None

# ── Роутер обучения/дообучения TrOCR (обязательный для наличия соответствующих эндпоинтов)
from api.train import router as train_router

# ── (опционально) троггер инициализации/прогрева TrOCR при старте приложения
logger = setup_logger("ocr")
try:
    from trocr.runtime import init_trocr  # если модуль присутствует и корректно собран TF
except Exception:
    init_trocr = None

# Название сервиса видно в /docs и /redoc
app = FastAPI(title="OCR Service (lines+OSD+TrOCR+WER+Train)")

# Подключаем роутеры (OCR — если доступен; Train — всегда)
if ocr_router:
    app.include_router(ocr_router)
app.include_router(train_router)

# CORS: даём фронту из ALLOWED_ORIGINS ходить в API любыми методами/заголовками
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    """
    Хук запуска приложения:
    - Если доступна функция init_trocr — инициализирует/прогревает TrOCR один раз.
    - Логирует флаг готовности (`trocr_ready`) для наблюдения.
    """
    if callable(init_trocr):
        ok = init_trocr()
        logger.info({"event": "startup", "trocr_ready": ok})
    else:
        logger.info({"event": "startup", "trocr_ready": False})

@app.get("/health")
def health():
    """
    Быстрый health-check эндпоинт для балансировщиков/мониторинга.
    Возвращает:
        {"status": "ok"}
    """
    return {"status": "ok"}
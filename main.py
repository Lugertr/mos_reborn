from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from logging_utils import setup_logger

# ── роуты OCR (если уже есть, не забудьте правильный импорт)
try:
    from api.routes import router as ocr_router  # опционально
except Exception:
    ocr_router = None

# ── роуты обучения
from api.train import router as train_router

# ── (опционально) прогрев TrOCR, если у вас есть runtime.init_trocr
logger = setup_logger("ocr")
try:
    from trocr.runtime import init_trocr  # если есть
except Exception:
    init_trocr = None

app = FastAPI(title="OCR Service (lines+OSD+TrOCR+WER+Train)")

if ocr_router:
    app.include_router(ocr_router)
app.include_router(train_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    if callable(init_trocr):
        ok = init_trocr()
        logger.info({"event": "startup", "trocr_ready": ok})
    else:
        logger.info({"event": "startup", "trocr_ready": False})

@app.get("/health")
def health():
    return {"status": "ok"}

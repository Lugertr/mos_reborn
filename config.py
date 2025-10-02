# config.py
from typing import List

# Разрешённые домены фронта для CORS
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

# Лимит multipart запроса (байты)
MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB

# Включение/выключение fallback в TrOCR
TROCR_ENABLED: bool = True

# Где лежит обученный ран TrOCR (внутри — charset.json, config.json)
TROCR_RUN_DIR: str = "runs/default"

# Какой файл весов загрузить из рана
TROCR_WEIGHTS_FILE: str = "best.weights.h5"

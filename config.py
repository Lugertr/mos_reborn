# config.py
from typing import List

# ---- CORS ----
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

# ---- Upload limits ----
MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB

# ---- TrOCR fallback toggle (оставляем как есть, вдруг понадобится) ----
TROCR_ENABLED: bool = True

# Где лежит обученный ран TrOCR (внутри — charset.json, config.json)
TROCR_RUN_DIR: str = "runs/default"
# Какой файл весов загрузить из рана
TROCR_WEIGHTS_FILE: str = "best.weights.h5"

# ---- Tesseract defaults (НОВЫЕ НАСТРОЙКИ) ----
# Языки: "rus+eng", "eng", "rus", и т.п.
TESS_LANGS: str = "rus"

# Page Segmentation Mode: 6 = Assume a single uniform block of text
TESS_PSM: int = 6

# OCR Engine Mode: 1 = LSTM only
TESS_OEM: int = 1

# Уровень сегментации: 4=строки (line), 5=слова (word)
TESS_LEVEL: int = 4

# Включать ли автоопределение ориентации/скрипта (OSD) перед OCR
TESS_OSD_ENABLE: bool = True

# Порог «низкой уверенности» (для решения о фолбэке или выставлении proxy-WER)
TESS_CONF_THRESHOLD: float = 70.0

# Минимальные эвристики для «сомнительных» фрагментов
TESS_MIN_WORDS: int = 2         # если слов меньше — считаем сомнительным
TESS_MIN_TEXT_LEN: int = 10     # если длина текста меньше — тоже сомнительный

# Предобработка по умолчанию: "soft" | "hard"
PREPROC_MODE_DEFAULT: str = "soft"

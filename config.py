"""
Purpose
-------
Единая точка настройки сервиса OCR:
- лимиты загрузки;
- параметры CORS для фронтенда;
- дефолты для Tesseract (языки, режимы сегментации/движка, пороги уверенности);
- переключатели и пути для TrOCR (fallback-модель);
- режим предобработки изображений по умолчанию.

Кто читает
----------
- API: см. api/routes.py (лимиты, предобработка, wer_mode, конфиденс-пороги).
- Tesseract-пайплайн: см. tesseract/segment_lines.py (LANGS/PSM/OEM/LEVEL).
- OSD-поворот: см. preprocessing/osd_utils.py (TESS_OSD_ENABLE).
- TrOCR рантайм: см. trocr/runtime.py (TROCR_RUN_DIR / TROCR_WEIGHTS_FILE / TROCR_ENABLED).

Как менять
----------
Значения задаются прямо здесь (ENV-переменные не используются). Для разных окружений
обычно делают отдельные файлы-конфиги или загружают их перед запуском.
"""

from typing import List

# ---- CORS ----
# Разрешённые origin'ы для браузерного фронтенда (Angular dev-сервер по умолчанию).
# Если фронтенд расположен на другом хосте/порту/протоколе — добавьте сюда его URL.
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

# ---- Upload limits ----
# Жёсткий лимит размера загружаемого файла (байты). Используется при чтении UploadFile.
# При превышении клиент получит HTTP 413.
MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB

# ---- TrOCR fallback toggle ----
# Глобальный переключатель использования дообучаемой модели TrOCR на «сомнительных» фрагментах.
# Если False — пайплайн ограничится Tesseract (без дорогого нейросетевого фолбэка).
TROCR_ENABLED: bool = True

# Папка ранa TrOCR: внутри ожидаются charset.json и config.json.
# Обычно это что-то вроде "runs/<имя_рана>".
TROCR_RUN_DIR: str = "runs/default"

# Имя файла весов, который грузится из папки ранa (может быть абсолютным путём).
# Типичные варианты: "best.weights.h5" (лучшие по метрике) или "last.weights.h5".
TROCR_WEIGHTS_FILE: str = "best.weights.h5"

# ---- Tesseract defaults ----
# Языки распознавания (через '+'): "rus+eng", "eng", "rus" и т.п.
# Важно: соответствующие языковые пакеты должны быть установлены в системе tesseract.
TESS_LANGS: str = "rus"

# Page Segmentation Mode (PSM):
# 6 = Assume a single uniform block of text (один однородный блок текста).
# Выбор PSM сильно влияет на разметку (page→block→par→line→word).
TESS_PSM: int = 6

# OCR Engine Mode (OEM):
# 1 = Только LSTM (современный движок Tesseract). Как правило, даёт лучшие результаты.
TESS_OEM: int = 1

# Уровень сегментации, который будет собираться из вывода image_to_data:
# 4 = строки (line), 5 = слова (word). См. tesseract/segment_lines.py.
TESS_LEVEL: int = 4

# Автоопределение ориентации/скрипта (OSD) перед OCR:
# При True изображение может быть повернуто согласно оценке Tesseract (90/180/270°).
# Реализация: preprocessing/osd_utils.py.
TESS_OSD_ENABLE: bool = True

# Порог «низкой уверенности» по данным Tesseract (в процентах, 0..100).
# Используется для решения о фолбэке в TrOCR и как прокси для WER.
TESS_CONF_THRESHOLD: float = 70.0

# Дополнительные эвристики для пометки «сомнительных» сегментов:
# - если слов меньше TESS_MIN_WORDS,
# - и/или итоговая строка короче TESS_MIN_TEXT_LEN,
# такой сегмент также может считаться кандидатом на TrOCR.
TESS_MIN_WORDS: int = 2         # если слов меньше — считаем сомнительным
TESS_MIN_TEXT_LEN: int = 10     # если длина текста меньше — тоже сомнительный

# ---- Preprocessing ----
# Режим предобработки по умолчанию: "soft" | "hard"
# - soft: CLAHE + median blur + deskew (без жёсткой бинаризации),
# - hard: то же + Otsu threshold (лучше на «грязных» сканах, хуже на тонких шрифтах).
PREPROC_MODE_DEFAULT: str = "soft"

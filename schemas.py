# schemas.py
"""
Purpose
-------
Pydantic-схемы (контракты API) для сервиса OCR.

Куда что возвращается/передаётся
--------------------------------
- /ocr_segments (POST):
    • Request: multipart/form-data с файлом и параметрами (см. api/routes.py).
    • Response: SegmentsResponse — список сегментов (строк/слов) + статистика пайплайна.

- /train/start (POST):
    • Request: TrainRequest — параметры обучения/дообучения TrOCR.
    • Response: TrainStartResponse — идентификатор фоновой задачи.

- /train/status/{job_id} (GET):
    • Response: TrainStatusResponse — текущее состояние задачи обучения.

Подсказка по stats (SegmentsResponse.stats)
-------------------------------------------
Типичный словарь:
{
  "total_lines": <int>,              # сколько сегментов собрано
  "fallback_to_trocr": <int>,        # сколько строк отправили в TrOCR (если включён)
  "conf_threshold": <float>,         # порог уверенности Tesseract для фоллбэка
  "wer_mode": "proxy"|"exact",       # режим оценки WER
  "trocr_enabled": <bool>,           # был ли включён TrOCR
  "img_width": <int>, "img_height": <int>,
  "trace": [                         # последовательность шагов с таймингами
      {"step": "...", "t_ms": <int>, ...}, ...
  ]
}
"""

from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel

# ---- уже были ----
class SegmentOut(BaseModel):
    """
    Единица результата OCR — сегмент (обычно строка текста, иногда слово,
    зависит от уровня сегментации Tesseract).

    Поля:
        coords: (x, y) — верхний левый угол bbox в пикселях исходного изображения.
        preview_value: «черновой» текст от Tesseract (для сравнения/подсветки).
        value: финальный текст сегмента. Если TrOCR был задействован на этой
               строке — здесь будет результат TrOCR; иначе — тот же Tesseract.
        wer: оценка ошибки (0..1). В proxy-режиме — эвристика; в exact — реальный WER
             относительно эталонной строки (если передан ref_text).
        width/height: размеры bbox сегмента (в пикселях).
    """
    coords: Tuple[int, int]   # [x,y] верхний-левый
    preview_value: str        # всегда Tesseract-гипотеза
    value: str                # финальная гипотеза (TrOCR при фоллбэке, иначе Tesseract)
    wer: float
    width: int
    height: int

class SegmentsResponse(BaseModel):
    """
    Ответ эндпоинта /ocr_segments.

    Поля:
        segments: список сегментов (по порядку сверху-вниз, слева-направо).
        stats: словарь с числовыми метриками и трассировкой этапов пайплайна
               (см. комментарий в шапке файла).
    """
    segments: List[SegmentOut]
    stats: dict

# ---- новое: обучение ----
class TrainRequest(BaseModel):
    """
    Параметры запуска обучения/дообучения TrOCR (передаются в /train/start).

    Разделы:
        режим:
            mode: "train" — полное обучение; "finetune" — дообучение от base_run.
            subset: "full" — использовать train/test; "postTest" — отладочный набор.

        пути:
            data_root: корень датасета (внутри ожидаются директории и *.json).
            run_dir: куда сохранять артефакты запуска (config, charset, веса, логи).
            base_run: (только для finetune) путь к базовому ранy.
            weights: имя/путь весов для старта (обычно "best.weights.h5").
            freeze_encoder: при finetune заморозить CNN-энкодер.

        гиперпараметры (опционально):
            Если поле None — используется дефолт из TrainConfig (см. trocr_modified.py).
            Сюда относятся размеры входа, длина текста, batch/epochs, lr,
            dropout’ы, архитектурные параметры трансформера, метрики мониторинга,
            параметры ранней остановки и т.д.
    """
    # режим
    mode: Literal["train", "finetune"] = "train"
    subset: Literal["full", "postTest"] = "full"

    # пути
    data_root: str = "data/handwritten"
    run_dir: str
    base_run: Optional[str] = None      # для finetune
    weights: Optional[str] = "best.weights.h5"
    freeze_encoder: Optional[bool] = False

    # гиперпараметры (опционально)
    img_h: Optional[int] = None
    img_w: Optional[int] = None
    max_text_len: Optional[int] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    learning_rate: Optional[float] = None
    label_smoothing: Optional[float] = None
    enc_dropout: Optional[float] = None
    dec_dropout: Optional[float] = None
    dec_layers: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    dff: Optional[int] = None
    monitor_metric: Optional[Literal["cer", "wer"]] = None
    mixed_precision: Optional[bool] = None
    lr_decay_rate: Optional[float] = None
    early_stop_patience: Optional[int] = None
    val_batches_for_eval: Optional[int] = None
    seed: Optional[int] = None

class TrainStartResponse(BaseModel):
    """
    Ответ на /train/start: идентификатор созданной фоновой задачи.

    Поля:
        job_id: UUID задачи — используйте его в /train/status/{job_id}.
        status: изначальный статус (обычно "queued").
        run_dir: каталог, где появятся артефакты обучения (логи/веса/конфиги).
    """
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    run_dir: str

class TrainStatusResponse(BaseModel):
    """
    Ответ на /train/status/{job_id}: текущее состояние обучения.

    Поля:
        job_id: идентификатор задачи.
        status: "queued" | "running" | "finished" | "failed".
        run_dir: путь к каталогу запуска.
        created_at/started_at/finished_at: ISO-временные метки (UTC, с суффиксом Z).
        message: текст ошибки (если статус == "failed").
        logs_path: путь к файлу лога обучения (если доступен).
    """
    job_id: str
    status: str
    run_dir: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None
    logs_path: Optional[str] = None

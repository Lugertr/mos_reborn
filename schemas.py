# schemas.py
from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel

# ---- уже были ----
class SegmentOut(BaseModel):
    coords: Tuple[int, int]   # [x,y] верхний-левый
    preview_value: str        # всегда Tesseract-гипотеза
    value: str                # финальная гипотеза (TrOCR при фоллбэке, иначе Tesseract)
    wer: float
    width: int
    height: int

class SegmentsResponse(BaseModel):
    segments: List[SegmentOut]
    stats: dict

# ---- новое: обучение ----
class TrainRequest(BaseModel):
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
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    run_dir: str

class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    run_dir: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None
    logs_path: Optional[str] = None

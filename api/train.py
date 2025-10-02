# api/train.py
import asyncio
import os
import uuid
import datetime as dt
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from logging_utils import setup_logger
from schemas import (
    TrainRequest,
    TrainStartResponse,
    TrainStatusResponse,
)
from trocr.trocr_modified import (
    TrainConfig,
    train_entry,
    finetune_entry,
)

logger = setup_logger("ocr.train")
router = APIRouter()

# Пул для фоновых задач обучения
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# Память о задачах (в проде — вынести в БД/Redis)
TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}


# --------- утилиты ---------

def _norm(path: str) -> str:
    return os.path.normpath(path)

def _ensure_dataset_present(data_root: str, subset: str) -> None:
    """
    Быстрая валидация структуры датасета:
    - subset='full'  -> train/, test/, train.json, test.json
    - subset='postTest' -> postTest/, postTest.json
    JSON формат: {"file_name": "...", "text": "..."}
    """
    data_root = _norm(data_root)
    if subset == "full":
        needed = [
            (_norm(os.path.join(data_root, "train.json")), True),
            (_norm(os.path.join(data_root, "test.json")), True),
            (_norm(os.path.join(data_root, "train")), False),
            (_norm(os.path.join(data_root, "test")), False),
        ]
    elif subset == "postTest":
        needed = [
            (_norm(os.path.join(data_root, "postTest.json")), True),
            (_norm(os.path.join(data_root, "postTest")), False),
        ]
    else:
        raise HTTPException(400, f"subset must be 'full' or 'postTest', got: {subset}")

    for path, is_file in needed:
        if is_file and not os.path.isfile(path):
            raise HTTPException(400, f"Missing file: {path}")
        if not is_file and not os.path.isdir(path):
            raise HTTPException(400, f"Missing directory: {path}")

    # Лёгкая проверка json-структуры (читаем только первые записи)
    import json
    to_check: List[str] = []
    if subset == "full":
        to_check = [os.path.join(data_root, "train.json"), os.path.join(data_root, "test.json")]
    else:
        to_check = [os.path.join(data_root, "postTest.json")]

    for jp in to_check:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if not isinstance(arr, list):
                raise ValueError("root is not a list")
            for i, obj in enumerate(arr[:10]):
                if not isinstance(obj, dict):
                    raise ValueError(f"item[{i}] is not an object")
                if "file_name" not in obj or "text" not in obj:
                    raise ValueError(f"item[{i}] missing 'file_name' or 'text'")
        except Exception as e:
            raise HTTPException(400, f"Bad JSON format in {jp}: {e}")


def _attach_file_handler(logger_name: str, log_path: str):
    """Добавить FileHandler к логгеру на время обучения и вернуть функцию-отцепитель."""
    import logging
    lg = setup_logger(logger_name)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    def _detach():
        try:
            lg.removeHandler(fh)
            fh.close()
        except Exception:
            pass

    return _detach


def _build_train_config(req: TrainRequest) -> TrainConfig:
    """Создать TrainConfig из запроса с учётом опциональных полей."""
    # Базовые значения соответствуют датаклассу TrainConfig
    cfg = TrainConfig(
        data_root=_norm(req.data_root),
        run_dir=_norm(req.run_dir),
        img_h=req.img_h or 120,
        img_w=req.img_w or 200,
        max_text_len=req.max_text_len or 80,
        batch_size=req.batch_size or 32,
        epochs=req.epochs or 40,
        steps_per_epoch=req.steps_per_epoch,
        val_batches_for_eval=req.val_batches_for_eval or 8,
        d_model=req.d_model or 256,
        num_heads=req.num_heads or 8,
        dff=req.dff or 512,
        enc_dropout=req.enc_dropout if req.enc_dropout is not None else 0.10,
        dec_dropout=req.dec_dropout if req.dec_dropout is not None else 0.10,
        dec_layers=req.dec_layers or 4,
        monitor_metric=req.monitor_metric or "cer",
        mixed_precision=bool(req.mixed_precision) if req.mixed_precision is not None else False,
        seed=req.seed or 42,
        label_smoothing=req.label_smoothing if req.label_smoothing is not None else 0.10,
        learning_rate=req.learning_rate if req.learning_rate is not None else 2e-4,
        init_weights=None,  # для train не используем; для finetune задаётся отдельно
        lr_decay_rate=req.lr_decay_rate if req.lr_decay_rate is not None else 0.98,
        early_stop_patience=req.early_stop_patience if req.early_stop_patience is not None else 6,
    )
    return cfg


def _train_worker(job_id: str, req: TrainRequest) -> None:
    """
    Функция, которая реально запускает обучение (в фоне).
    Обновляет TRAIN_JOBS[job_id] по мере выполнения.
    """
    job = TRAIN_JOBS[job_id]
    job["status"] = "running"
    logger.info({"event": "train_start", "job_id": job_id, "mode": req.mode, "subset": req.subset, "run_dir": req.run_dir})

    # Логи — в файл в каталоге run_dir
    os.makedirs(req.run_dir, exist_ok=True)
    log_path = os.path.join(req.run_dir, "train.log")
    detach = _attach_file_handler("trocr", log_path)
    detach_api = _attach_file_handler("ocr.train", log_path)  # и наш логгер туда же
    job["logs_path"] = log_path

    try:
        # Проверим датасет
        _ensure_dataset_present(req.data_root, req.subset)

        # Сконструируем конфиг
        cfg = _build_train_config(req)

        # Запуск
        if req.mode == "finetune":
            if not req.base_run:
                raise ValueError("base_run is required for finetune mode")
            finetune_entry(
                base_run=_norm(req.base_run),
                new_run=_norm(req.run_dir),
                data_root=_norm(req.data_root),
                subset=req.subset,
                weights=req.weights or "best.weights.h5",
                epochs=cfg.epochs,
                steps_per_epoch=cfg.steps_per_epoch,
                batch_size=cfg.batch_size,
                img_h=cfg.img_h,
                img_w=cfg.img_w,
                max_text_len=cfg.max_text_len,
                lr=cfg.learning_rate,
                freeze_encoder=bool(req.freeze_encoder),
                enc_dropout=cfg.enc_dropout,
                dec_dropout=cfg.dec_dropout,
                label_smoothing=cfg.label_smoothing,
                mixed_precision=cfg.mixed_precision,
                monitor_metric=cfg.monitor_metric,
                lr_decay_rate=cfg.lr_decay_rate,
                early_stop_patience=cfg.early_stop_patience,
                val_batches_for_eval=cfg.val_batches_for_eval,
                seed=cfg.seed,
            )
        else:
            # Полное обучение
            train_entry(cfg, subset=req.subset)

        job["status"] = "finished"
        job["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
        logger.info({"event": "train_done", "job_id": job_id, "status": "finished"})
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
        logger.exception({"event": "train_failed", "job_id": job_id, "error": str(e)})
    finally:
        try:
            detach()
            detach_api()
        except Exception:
            pass


# --------- эндпоинты ---------

@router.post("/train/start", response_model=TrainStartResponse)
async def start_training(req: TrainRequest):
    """
    Запускает обучение/дообучение в фоне.
    Возвращает job_id, который можно опрашивать по /train/status/{job_id}.
    """
    # Базовая валидация pydantic уже сделал; дополнительно проверим поля режима
    try:
        if req.mode == "finetune" and not req.base_run:
            raise HTTPException(400, "base_run is required for finetune mode")
    except ValidationError as e:
        raise HTTPException(400, str(e))

    # Создаём запись о задаче
    job_id = str(uuid.uuid4())
    TRAIN_JOBS[job_id] = {
        "status": "queued",
        "run_dir": _norm(req.run_dir),
        "params": req.model_dump(),
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "finished_at": None,
        "logs_path": None,
        "error": None,
    }

    # Стартуем в фоне
    loop = asyncio.get_running_loop()
    TRAIN_JOBS[job_id]["started_at"] = dt.datetime.utcnow().isoformat() + "Z"
    loop.run_in_executor(_EXECUTOR, _train_worker, job_id, req)

    return TrainStartResponse(
        job_id=job_id,
        status="queued",
        run_dir=_norm(req.run_dir),
    )


@router.get("/train/status/{job_id}", response_model=TrainStatusResponse)
async def training_status(job_id: str):
    """
    Возвращает статус задачи обучения.
    """
    job = TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"job_id not found: {job_id}")

    return TrainStatusResponse(
        job_id=job_id,
        status=job["status"],
        run_dir=job["run_dir"],
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        message=job.get("error"),
        logs_path=job.get("logs_path"),
    )

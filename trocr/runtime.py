# trocr/runtime.py
"""
Purpose
-------
Лёгкий рантайм-обёртка для инференса TrOCR из API:
  • лениво загружает модель/charset/config один раз (`init_trocr`);
  • готовит батч из PIL-изображений строк и вызывает greedy-декод;
  • предоставляет `recognize_batch(images)` для использования в пайплайне.

Notes
-----
- Параметры пути к запуску и весам берутся из config: `TROCR_RUN_DIR`, `TROCR_WEIGHTS_FILE`.
- При инициализации выполняется «прогрев» на белом квадрате (ускоряет первый реальный вызов).
"""

from typing import List, Optional
from PIL import Image
import numpy as np
import tensorflow as tf

from config import TROCR_RUN_DIR, TROCR_WEIGHTS_FILE
from logging_utils import setup_logger, timeblock
from .trocr_modified import (
    load_model_from_run, greedy_decode, preprocess_image,
    Charset, TrainConfig
)

logger = setup_logger("ocr.trocr")

_MODEL: Optional[tf.keras.Model] = None
_CHARSET: Optional[Charset] = None
_CFG: Optional[TrainConfig] = None
_READY: bool = False

def init_trocr() -> bool:
    """
    Инициализировать глобальные объекты TrOCR для инференса (модель/charset/config).

    Returns:
        True, если модель успешно загружена и прогрета; False при любой ошибке.

    Side effects:
        - Загружает модель и веса из `TROCR_RUN_DIR` / `TROCR_WEIGHTS_FILE`.
        - Выполняет «прогрев» через recognize_batch([white_img]).
        - Заполняет глобальные `_MODEL`, `_CHARSET`, `_CFG`, `_READY`.
    """
    global _MODEL, _CHARSET, _CFG, _READY
    try:
        with timeblock(logger, "trocr_load", run=TROCR_RUN_DIR):
            model, charset, cfg = load_model_from_run(TROCR_RUN_DIR)
            # Строим граф
            dummy = {
                "image": tf.zeros([1, cfg.img_h, cfg.img_w, 1], tf.float32),
                "decoder_input": tf.fill([1, cfg.max_text_len], tf.cast(charset.sos_id, tf.int32))
            }
            _ = model(dummy, training=False)
            # Веса
            wpath = TROCR_WEIGHTS_FILE
            if not wpath.startswith("/"):
                wpath = f"{TROCR_RUN_DIR.rstrip('/')}/{wpath}"
            model.load_weights(wpath)
            logger.info({"event": "trocr_weights_loaded", "path": wpath})
            _MODEL, _CHARSET, _CFG = model, charset, cfg

        # Прогрев на белом квадрате
        with timeblock(logger, "trocr_warmup"):
            img = Image.new("L", (cfg.img_w, cfg.img_h), color=255)
            recognize_batch([img])  # вызов ниже использует текущий _MODEL
        _READY = True
        return True
    except Exception as e:
        logger.error({"event": "trocr_init_failed", "error": str(e)})
        _READY = False
        return False

def _pil_to_tensor_batch(images: List[Image.Image], h: int, w: int) -> tf.Tensor:
    """
    Преобразовать список PIL.Image в 4D-тензор (B, H, W, 1) с нормализацией/паддингом.

    Args:
        images: Список кропов строк (PIL.Image, обычно "L", но допускается RGB).
        h, w: Целевой размер модели.

    Returns:
        tf.Tensor формы (batch, h, w, 1) в диапазоне [0..1], паддинг справа белым.
    """
    tensors = []
    for im in images:
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[-1] != 1:
            arr = arr[..., 0]  # берём один канал
        t = tf.convert_to_tensor(arr, dtype=tf.float32)
        # preprocess_image ожидает (H, W, 1), значение в [0..1], паддинг справа
        t = preprocess_image(t, h, w)  # вернёт (h, w, 1)
        tensors.append(t)
    return tf.stack(tensors, axis=0)  # (B, h, w, 1)

def recognize_batch(images: List[Image.Image]) -> List[str]:
    """
    Выполнить батч-инференс TrOCR и вернуть распознанные строки.

    Args:
        images: список PIL.Image, каждый — одна строка текста.

    Returns:
        Список строк-предсказаний (len == len(images)).

    Raises:
        RuntimeError: если TrOCR ещё не инициализирован (`init_trocr()` не вызывался).
    """
    if _MODEL is None or _CHARSET is None or _CFG is None:
        raise RuntimeError("TrOCR is not initialized")
    if not images:
        return []
    x = _pil_to_tensor_batch(images, _CFG.img_h, _CFG.img_w)
    enc_key_mask = None  # пусть greedy_decode построит из encoder’а
    with timeblock(logger, "trocr_infer", batch=len(images)):
        preds = greedy_decode(_MODEL, x, enc_key_mask, _CHARSET, _CFG.max_text_len)
    return preds

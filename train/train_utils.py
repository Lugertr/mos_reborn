#!/usr/bin/env python3
# train_utils.py
"""
Вспомогательные функции: подготовка данных, метрики, декодер, сохранение.
Только TensorFlow + transformers.
"""

import os
import re
import sys
import logging
import unicodedata
import string
from typing import Optional, List, Tuple, Dict, Any

# ensure config import works whether запускаешь из train/ или корня
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
import config  # type: ignore

from PIL import Image
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def find_latest_checkpoint(base_dir: str) -> Optional[str]:
    if not base_dir or not os.path.isdir(base_dir):
        return None
    pattern = re.compile(r"^checkpoint-(\d+)$")
    best_num = -1
    best_path: Optional[str] = None
    for name in os.listdir(base_dir):
        m = pattern.match(name)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if n > best_num:
            best_num = n
            best_path = os.path.join(base_dir, name)
    return best_path


def simple_wer(ref: str, hyp: str) -> float:
    if ref is None:
        ref = ""
    if hyp is None:
        hyp = ""
    r = ref.split()
    h = hyp.split()
    n = len(r)
    m = len(h)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    distance = prev[m]
    return float(distance) / float(n)


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    rus_punct = "«»—–…„“‚‘’"
    punct = string.punctuation + rus_punct
    tr = str.maketrans({c: "" for c in punct})
    s = s.translate(tr)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def postprocess_text(pred_ids: List[List[int]],
                     label_ids: List[List[int]],
                     processor,
                     normalize: bool = True) -> Tuple[List[str], List[str]]:
    # decoder -> text
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    cleaned_labels = [[l for l in seq if l != -100] for seq in label_ids]
    label_texts = processor.batch_decode(cleaned_labels, skip_special_tokens=True)
    if normalize:
        pred_texts = [normalize_text(s) for s in pred_texts]
        label_texts = [normalize_text(s) for s in label_texts]
    return pred_texts, label_texts


def compute_metrics_from_processor(eval_pred: Tuple[List[List[int]], List[List[int]]],
                                   processor) -> Dict[str, float]:
    pred_ids, label_ids = eval_pred
    pred_texts, label_texts = postprocess_text(pred_ids, label_ids, processor, normalize=True)
    wers = [simple_wer(r, h) for r, h in zip(label_texts, pred_texts)]
    avg = sum(wers) / len(wers) if wers else 0.0
    return {"wer": avg}


def get_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    return "GPU" if gpus else "CPU"


def save_model_and_processor(model, processor, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        # try HF-style save
        model.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("save_model_and_processor: model.save_pretrained failed: %s", e)
        try:
            # fallback: keras save for TF models
            model.save(out_dir, include_optimizer=False)
        except Exception as e2:
            logger.error("save_model_and_processor: fallback save failed: %s", e2)
    try:
        processor.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("save_model_and_processor: processor.save_pretrained failed: %s", e)


def transform_example(example: Dict[str, Any], images_dir: str, processor) -> Dict[str, Any]:
    """
    Возвращает pixel_values (np.float32) и labels (list[int]) для одного примера.
    Ожидается, что example содержит file_name (или image) и text.
    """
    file_name = example.get("file_name") or example.get("image")
    if file_name is None:
        raise KeyError("transform_example: в примере нет поля file_name или image")
    img_path = file_name if os.path.isabs(file_name) else os.path.join(images_dir, file_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"transform_example: изображение не найдено: {img_path}")
    img = Image.open(img_path).convert("RGB")
    out = processor(images=img, return_tensors="np")
    pv = out["pixel_values"]
    pv = pv[0]
    # возможное преобразование канальной оси: привести к (C,H,W)
    if pv.ndim == 3 and pv.shape[2] in (1, 3) and pv.shape[0] not in (1, 3):
        pv = np.transpose(pv, (2, 0, 1))
    labels = processor.tokenizer(example.get("text", ""), truncation=True, max_length=getattr(config, "GENERATION_MAX_LENGTH", 128)).input_ids
    return {"pixel_values": pv.astype(np.float32), "labels": labels}


def batched_greedy_decode_tf(model, processor, pixel_values: tf.Tensor, max_length: int) -> List[List[int]]:
    """
    Батчевый greedy-декодер на tf.while_loop. Возвращает список списков id.
    Используется, если model.generate падает или недоступен.
    """
    batch_size = tf.shape(pixel_values)[0]
    start_id = int(model.config.decoder_start_token_id or getattr(processor.tokenizer, "bos_token_id", 0) or 0)
    eos_id_val = getattr(model.config, "eos_token_id", None) or getattr(processor.tokenizer, "eos_token_id", None)
    eos_id = int(eos_id_val) if eos_id_val is not None else -1
    start_id_t = tf.constant(start_id, dtype=tf.int32)
    eos_id_t = tf.constant(eos_id, dtype=tf.int32)

    cur = tf.fill([batch_size, 1], start_id_t)
    finished = tf.zeros([batch_size], dtype=tf.bool)
    step = tf.constant(0)
    max_steps = tf.constant(max_length, dtype=tf.int32)

    def cond(cur, finished, step):
        return tf.logical_and(tf.less(step, max_steps), tf.logical_not(tf.reduce_all(finished)))

    def body(cur, finished, step):
        try:
            outputs = model(pixel_values=pixel_values, decoder_input_ids=cur, training=False)
            logits = outputs.logits
        except Exception:
            # Некоторые TF-модели не принимают decoder_input_ids в call; пробуем без него
            outputs = model(pixel_values=pixel_values, training=False)
            logits = outputs.logits
        last_logits = logits[:, -1, :]
        next_ids = tf.cast(tf.argmax(last_logits, axis=-1), tf.int32)
        next_ids_exp = tf.expand_dims(next_ids, axis=1)
        cur = tf.concat([cur, next_ids_exp], axis=1)
        if eos_id >= 0:
            finished = tf.logical_or(finished, tf.equal(next_ids, eos_id_t))
        step = step + 1
        return cur, finished, step

    cur, finished, step = tf.while_loop(cond, body, [cur, finished, step],
                                       shape_invariants=[tf.TensorShape([None, None]),
                                                         tf.TensorShape([None]),
                                                         step.get_shape()])
    try:
        seqs = cur.numpy().tolist()
    except Exception:
        seqs = tf.keras.backend.get_value(cur).tolist()
    return seqs

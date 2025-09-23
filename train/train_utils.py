# train_utils.py
"""
Вспомогательные функции: обработка данных, декодирование, вычисление метрик, сохранение.
Используется только TensorFlow и библиотеки tokenizers/transformers.
"""

import os
import re
import logging
import unicodedata
import string

from typing import Optional, List, Tuple, Dict, Any

from PIL import Image
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def find_latest_checkpoint(base_dir: str) -> Optional[str]:
    """
    Отдать путь к последнему чекпоинту (по имени папки checkpoint-<число>).
    Или None, если чекпоинтов нет.
    """
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
    """
    Простейший WER — по словам, с расстоянием Левенштейна.
    Если референс пустой — 0.0, если гипотеза непустая — 1.0.
    """
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
            deletion = prev[j] + 1
            insertion = cur[j - 1] + 1
            substitution = prev[j - 1] + cost
            cur[j] = min(deletion, insertion, substitution)
        prev, cur = cur, prev
    distance = prev[m]
    return float(distance) / float(n)


def normalize_text(s: Optional[str]) -> str:
    """
    Нормализация текста: Unicode NFC, нижний регистр, удаление управляющих символов,
    удаление пунктуации, сжатие пробелов.
    Можно изменить, если нужно сохранить регистр или специальные символы.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    # удалить управляющие символы
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
    """
    Преобразовать списки id в строки, удалить специальные токены / паддинги / -100,
    нормализовать текст, если normalize=True.
    """
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    cleaned_labels = [[l for l in seq if l != -100] for seq in label_ids]
    label_texts = processor.batch_decode(cleaned_labels, skip_special_tokens=True)
    if normalize:
        pred_texts = [normalize_text(s) for s in pred_texts]
        label_texts = [normalize_text(s) for s in label_texts]
    return pred_texts, label_texts


def compute_metrics_from_processor(eval_pred: Tuple[List[List[int]], List[List[int]]],
                                   processor) -> Dict[str, float]:
    """
    Вычисление метрик, здесь WER.
    eval_pred = (pred_ids, label_ids)
    """
    pred_ids, label_ids = eval_pred
    pred_texts, label_texts = postprocess_text(pred_ids, label_ids, processor, normalize=True)
    wers = [simple_wer(r, h) for r, h in zip(label_texts, pred_texts)]
    avg = sum(wers) / len(wers) if wers else 0.0
    return {"wer": avg}


def get_device() -> str:
    """
    Узнать, доступен ли GPU через TensorFlow.
    Возвращает строку "GPU" или "CPU".
    """
    gpus = tf.config.list_physical_devices("GPU")
    return "GPU" if gpus else "CPU"


def save_model_and_processor(model, processor, out_dir: str):
    """
    Сохраняет модель и processor (токенизатор + feature extractor) в директорию out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("save_model_and_processor: модель save_pretrained не сработала: %s", e)
    try:
        processor.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("save_model_and_processor: processor save_pretrained не сработал: %s", e)


def transform_example(example: Dict[str, Any], images_dir: str, processor) -> Dict[str, Any]:
    """
    Преобразует один элемент датасета: читает изображение + получает pixel_values,
    токенизирует текст.
    Возвращает: {"pixel_values": np.ndarray, "labels": List[int]}
    """
    file_name = example.get("file_name") or example.get("image")
    if file_name is None:
        raise KeyError("transform_example: в примере нет file_name или image")
    img_path = file_name if os.path.isabs(file_name) else os.path.join(images_dir, file_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"transform_example: изображение не найдено: {img_path}")

    img = Image.open(img_path).convert("RGB")
    # processor(feature_extractor + tokenizer) из Transformers обрабатывает изображение + текст
    out = processor(images=img, return_tensors="np")
    # pixel_values: np.ndarray, shape может быть (1, C, H, W) или (1, H, W, C)
    pv = out["pixel_values"]
    # берем первый элемент
    pv = pv[0]
    # преобразование: если channel last, возможно transpose
    if pv.ndim == 3 and pv.shape[2] in (1,3) and (pv.shape[0] not in (1,3)):
        # предположительно форма (H, W, C) -> делаем (C, H, W)
        pv = np.transpose(pv, (2, 0, 1))
    labels = processor.tokenizer(example.get("text", ""), truncation=True, max_length=config.GENERATION_MAX_LENGTH).input_ids
    return {"pixel_values": pv.astype(np.float32), "labels": labels}

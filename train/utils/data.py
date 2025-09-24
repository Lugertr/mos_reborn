# utils/data.py
# coding: utf-8
"""
Утилиты для подготовки данных для TrOCR.

Функции:
 - transform_example_py: открыть изображение, применить processor -> pixel_values + labels
 - prepare_tf_dataset: опционально генерирует TFRecord (preprocessed pixel_values) и затем
   читает TFRecord с параллельным парсингом и padded_batch, либо использует from_generator.

Комментарии и сообщения — на русском.
"""
import os
import logging
import numpy as np
from typing import Dict, Any, Iterable, List, Optional, Tuple

from PIL import Image
import tensorflow as tf

import config  # type: ignore

logger = logging.getLogger(__name__)


def transform_example_py(example: Dict[str, Any], images_dir: str, processor) -> Dict[str, Any]:
    """
    Открыть изображение, применить processor (TrOCRProcessor с ViTImageProcessor внутри) и токенизировать метку.
    Возвращает: {'pixel_values': np.ndarray (C,H,W), 'labels': List[int]}
    """
    file_name = example.get("file_name") or example.get("image")
    if file_name is None:
        raise KeyError("transform_example: отсутствует поле 'file_name' или 'image' в примере")
    img_path = file_name if os.path.isabs(file_name) else os.path.join(images_dir, file_name)
    img_path = os.path.normpath(os.path.abspath(img_path))
    images_dir_abs = os.path.normpath(os.path.abspath(images_dir))
    if not img_path.startswith(images_dir_abs):
        raise ValueError(f"Неверный путь к изображению (вне images_dir): {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"transform_example: изображение не найдено: {img_path}")

    with Image.open(img_path) as im:
        img = im.convert("RGB")
        out = processor(images=img, return_tensors="np")
    pv = out["pixel_values"]  # обычно shape (1, C, H, W) или (1, H, W, C)
    pv = pv[0]
    # если pixel_values в формате (H,W,C) — привести к (C,H,W)
    if pv.ndim == 3 and pv.shape[-1] in (1, 3) and pv.shape[0] not in (1, 3):
        pv = np.transpose(pv, (2, 0, 1))
    labels = processor.tokenizer(example.get("text", ""), truncation=True,
                                 max_length=getattr(config, "GENERATION_MAX_LENGTH", 128)).input_ids
    return {"pixel_values": pv.astype(np.float32), "labels": labels}


# ---------------- TFRecord сериализация / парсинг ----------------

def _serialize_example_to_tfrecord(pv: np.ndarray, labels: List[int]) -> tf.train.Example:
    """
    Сериализация одного примера в tf.train.Example:
      - pv_shape: int64 list
      - pv: bytes (float32.tobytes())
      - labels: int64 list
    """
    pv_shape = list(pv.shape)
    pv_bytes = pv.tobytes()
    feature = {
        "pv_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=pv_shape)),
        "pv": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pv_bytes])),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_tfrecord_from_split(hf_split: Iterable[Dict[str, Any]],
                               images_dir: str,
                               processor,
                               out_path: str) -> None:
    """
    Создать TFRecord файл из HF split (train/test). Сохраняет pv и labels.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = tf.io.TFRecordWriter(out_path)
    cnt = 0
    for ex in hf_split:
        try:
            out = transform_example_py(ex, images_dir, processor)
        except Exception as e:
            logger.warning("transform_example_py failed for %s: %s", ex.get("file_name") or ex.get("image"), e)
            continue
        pv = out["pixel_values"].astype(np.float32)
        labels = out["labels"]
        ex_tf = _serialize_example_to_tfrecord(pv, labels)
        writer.write(ex_tf.SerializeToString())
        cnt += 1
    writer.close()
    logger.info("Записан TFRecord: %s (примеров=%d)", out_path, cnt)


def _parse_tfrecord_fn(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Разбор tf.train.Example -> (pv tensor float32 (C,H,W), labels int32 vector)
    """
    feature_description = {
        "pv_shape": tf.io.VarLenFeature(tf.int64),
        "pv": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    pv_shape = tf.sparse.to_dense(parsed["pv_shape"])
    pv_raw = parsed["pv"]
    labels = tf.sparse.to_dense(parsed["labels"])
    pv_shape = tf.cast(pv_shape, tf.int32)
    pv = tf.io.decode_raw(pv_raw, tf.float32)
    pv = tf.reshape(pv, pv_shape)
    return pv, tf.cast(labels, tf.int32)


# ---------------- Dataset builder ----------------

def prepare_tf_dataset(hf_split,
                       processor,
                       images_dir: str,
                       batch_size: int,
                       shuffle: bool):
    """
    Подготовить tf.data.Dataset из HuggingFace split.

    - Если config.USE_TFRECORDS=True: создаём/используем TFRecord и читаем его с map(..., num_parallel_calls).
    - Иначе: используем Dataset.from_generator + padded_batch + prefetch.

    Возвращает батчированный tf.data.Dataset.
    """
    use_tfrecords = bool(getattr(config, "USE_TFRECORDS", False))
    tfrecords_dir = getattr(config, "TFRECORDS_DIR", os.path.join(os.path.dirname(getattr(config, "CHECKPOINTS_DIR")), "tfrecords"))
    if use_tfrecords:
        try:
            split_len = len(hf_split)
            tfrecord_name = f"pv-{split_len}.tfrecord"
        except Exception:
            tfrecord_name = "pv.tfrecord"
        tfrecord_path = os.path.join(tfrecords_dir, tfrecord_name)
        if not os.path.exists(tfrecord_path):
            logger.info("TFRecord %s не найден — создаём: %s", tfrecord_name, tfrecord_path)
            _write_tfrecord_from_split(hf_split, images_dir, processor, tfrecord_path)
        else:
            logger.info("Используем существующий TFRecord: %s", tfrecord_path)

        ds = tf.data.TFRecordDataset([tfrecord_path])
        ds = ds.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=1024)
        pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
        # определим pv_shape из конфигурации, fallback к TOKENIZER_IMAGE_SIZE
        try:
            size = getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))
            C = 3
            H, W = size[0], size[1]
            pv_shape = (C, H, W)
        except Exception:
            pv_shape = (3, getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))[0], getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))[1])
        ds = ds.padded_batch(batch_size, padded_shapes=(pv_shape, [None]), padding_values=(0.0, pad_id))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # fallback: from_generator
    def gen():
        for ex in hf_split:
            try:
                out = transform_example_py(ex, images_dir, processor)
            except Exception as e:
                logger.warning("transform_example_py failed for %s: %s", ex.get("file_name") or ex.get("image"), e)
                continue
            pv = out["pixel_values"].astype(np.float32)
            lab = list(map(int, out["labels"]))
            yield pv, lab

    it = gen()
    try:
        pv0, _ = next(it)
    except StopIteration:
        raise RuntimeError("prepare_tf_dataset: раздел датасета пуст.")
    pv_shape = pv0.shape  # e.g. (3, H, W)

    output_signature = (
        tf.TensorSpec(shape=pv_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
    if shuffle:
        try:
            ds_size = len(hf_split)
            buf = min(1000, max(100, ds_size))
        except Exception:
            buf = 1000
        ds = ds.shuffle(buffer_size=buf)
    pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    ds = ds.padded_batch(batch_size, padded_shapes=(pv_shape, [None]), padding_values=(0.0, pad_id))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

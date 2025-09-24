"""
utils/data.py — упрощённая TFRecord сериализация (предполагается фиксированный размер pv)
и быстрый pipeline с prefetch_to_device.
"""
import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any

import config  # type: ignore
logger = logging.getLogger(__name__)

# ожидаемая форма pv: (C, H, W) с H,W = TOKENIZER_IMAGE_SIZE
PV_CHANNELS = 3
PV_H = getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))[0]
PV_W = getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))[1]


def _serialize_example_fixed(pv: np.ndarray, labels: np.ndarray) -> tf.train.Example:
    """
    Упрощённая сериализация: pv -> raw bytes (float32), labels -> int64 list.
    Предполагаем фиксированную форму pv (C,H,W).
    """
    pv_raw = pv.tobytes()
    features = {
        "pv_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pv_raw])),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels.tolist())),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def _write_tfrecord(path: str, gen_fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tf.io.TFRecordWriter(path) as writer:
        for pv, labels in gen_fn():
            ex = _serialize_example_fixed(pv, np.array(labels, dtype=np.int64))
            writer.write(ex.SerializeToString())


def create_tfrecord_cache_fixed(split_ds, processor, images_dir: str, out_path: str, shards: int = 1):
    """
    Создаёт один или несколько tfrecord-файлов (shards).
    Использует preprocess_example_np из tokenizer_utils (возвращает фиксированный pv).
    """
    from utils.tokenizer_utils import preprocess_example_np
    logger.info("Creating TFRecord cache at %s (shards=%d)", out_path, shards)

    # если shards==1, просто пишем один файл
    if shards <= 1:
        def gen():
            for ex in split_ds:
                try:
                    item = preprocess_example_np(ex, images_dir, processor)
                except Exception as e:
                    logger.warning("preprocess_example_np failed for %s: %s", ex.get("file_name") or ex.get("image"), e)
                    continue
                pv = item["pixel_values"].astype(np.float32)
                labels = np.array(item["labels"], dtype=np.int64)
                yield pv, labels
        _write_tfrecord(out_path, gen)
        return [out_path]

    # sharded write: round-robin
    base = out_path.rstrip(".tfrecord")
    paths = [f"{base}-shard-{i}.tfrecord" for i in range(shards)]
    writers = [tf.io.TFRecordWriter(p) for p in paths]
    try:
        i = 0
        for ex in split_ds:
            try:
                item = preprocess_example_np(ex, images_dir, processor)
            except Exception as e:
                logger.warning("preprocess_example_np failed: %s", e)
                continue
            pv = item["pixel_values"].astype(np.float32)
            labels = np.array(item["labels"], dtype=np.int64)
            ex_rec = _serialize_example_fixed(pv, labels)
            writers[i % shards].write(ex_rec.SerializeToString())
            i += 1
    finally:
        for w in writers:
            w.close()
    return paths


def _parse_example_fixed(serialized_example):
    # pv_raw (bytes), labels (varlen int)
    feature_description = {
        "pv_raw": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    pv_raw = parsed["pv_raw"]
    labels_sparse = parsed["labels"]
    labels = tf.sparse.to_dense(labels_sparse)

    # decode pv bytes -> float32 and reshape to (C,H,W)
    def _decode_pv(pv_bytes):
        pv = tf.io.decode_raw(pv_bytes, tf.float32)
        pv = tf.reshape(pv, (PV_CHANNELS, PV_H, PV_W))
        return pv
    pv = tf.py_function(_decode_pv, [pv_raw], Tout=tf.float32)
    pv.set_shape([PV_CHANNELS, PV_H, PV_W])
    return pv, tf.cast(labels, tf.int32)


def prepare_tf_dataset(split_ds, processor, images_dir: str, batch_size: int, shuffle: bool):
    """
    Если config.USE_TFRECORD_CACHE=True — читаем (или создаём) TFRecord(ы).
    Иначе — используем generator -> from_generator (preprocessed numpy), затем padded_batch.
    После batch делаем prefetch и при наличии GPU применяем prefetch_to_device.
    """
    if getattr(config, "USE_TFRECORD_CACHE", False):
        tfrecord_dir = getattr(config, "TFRECORD_DIR", os.path.join(".", "tfrecords"))
        os.makedirs(tfrecord_dir, exist_ok=True)
        split_name = getattr(split_ds, "split", None) or "split"
        tfpath = os.path.join(tfrecord_dir, f"{split_name}.tfrecord")
        shards = getattr(config, "TFRECORD_SHARDS", 1)
        if shards <= 1:
            if not os.path.exists(tfpath):
                create_tfrecord_cache_fixed(split_ds, processor, images_dir, tfpath, shards=1)
            ds = tf.data.TFRecordDataset([tfpath])
            ds = ds.map(_parse_example_fixed, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            base = os.path.join(tfrecord_dir, f"{split_name}.tfrecord")
            shard_paths = [f"{base}-shard-{i}.tfrecord" for i in range(shards)]
            missing = [p for p in shard_paths if not os.path.exists(p)]
            if missing:
                # create shards
                create_tfrecord_cache_fixed(split_ds, processor, images_dir, base + ".tfrecord", shards=shards)
            ds = tf.data.Dataset.from_tensor_slices(shard_paths)
            ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_example_fixed, num_parallel_calls=tf.data.AUTOTUNE),
                               cycle_length=min(shards, 4), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # generator path: use preprocess_example_np to produce pv,labels
        from utils.tokenizer_utils import preprocess_example_np
        def gen_pv_lab():
            for ex in split_ds:
                try:
                    item = preprocess_example_np(ex, images_dir, processor)
                except Exception as e:
                    logger.warning("preprocess_example_np failed: %s", e)
                    continue
                yield item["pixel_values"].astype(np.float32), np.array(item["labels"], dtype=np.int32)
        ds = tf.data.Dataset.from_generator(lambda: gen_pv_lab(),
                                            output_signature=(
                                                tf.TensorSpec(shape=(PV_CHANNELS, PV_H, PV_W), dtype=tf.float32),
                                                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                            ))
    if shuffle:
        try:
            ds_size = len(split_ds)
            buf = min(1000, max(100, ds_size))
        except Exception:
            buf = 1000
        ds = ds.shuffle(buffer_size=buf)
    pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    ds = ds.padded_batch(batch_size, padded_shapes=((PV_CHANNELS, PV_H, PV_W), [None]), padding_values=(0.0, pad_id))
    # prefetch & optionally prefetch to device (GPU) to reduce host->device overhead
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # if GPU present, move data onto device early
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            device_str = "/device:GPU:0"
            ds = ds.apply(tf.data.experimental.prefetch_to_device(device_str))
        except Exception:
            logger.debug("prefetch_to_device not available / failed (non-fatal).")
    return ds

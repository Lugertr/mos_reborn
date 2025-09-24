"""
checkpoints.py — централизованная логика чекпоинтов и retention policy.
"""
import os
import json
import logging
import threading
from typing import Optional, Tuple, List

import tensorflow as tf
import config  # type: ignore

logger = logging.getLogger(__name__)
_save_lock = threading.Lock()


def _list_checkpoint_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]


def save_checkpoint(epoch: int,
                    batch: int,
                    model,
                    processor,
                    optimizer,
                    ckpt_obj,
                    reason: str = "auto",
                    hf_save: bool = True) -> str:
    """
    Сохранение чекпоинта с retention policy.
    hf_save: если False — сохраняем только TF checkpoint + metadata (быстрее).
    Возвращает путь к созданной директории.
    """
    with _save_lock:
        ckpts_dir = getattr(config, "CHECKPOINTS_DIR")
        os.makedirs(ckpts_dir, exist_ok=True)
        suffix = f"-partial-{batch}" if batch and batch > 0 else ""
        ckpt_name = f"{getattr(config, 'CHECKPOINT_PREFIX', 'checkpoint-')}{epoch}{suffix}"
        ckpt_dir = os.path.join(ckpts_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        tf_ckpt_prefix = None
        # HF save (weights/config) — опционально
        if hf_save:
            try:
                model.save_pretrained(ckpt_dir)
            except Exception:
                logger.debug("model.save_pretrained failed (non-fatal).")
            try:
                processor.save_pretrained(ckpt_dir)
            except Exception:
                if hasattr(processor, "tokenizer"):
                    try:
                        processor.tokenizer.save_pretrained(ckpt_dir)
                    except Exception:
                        logger.debug("Не удалось сохранить tokenizer в чекпоинт.")

        # TF checkpoint (обычно быстро)
        try:
            if ckpt_obj is None:
                ckpt_obj = tf.train.Checkpoint(optimizer=optimizer, model=model)
            tf_ckpt_prefix = ckpt_obj.save(os.path.join(ckpt_dir, "tf_ckpt"))
        except Exception as e:
            logger.warning("TF checkpoint save failed: %s", e)

        meta = {"epoch": epoch, "batch": batch, "tf_checkpoint": tf_ckpt_prefix, "reason": reason}
        meta_path = os.path.join(ckpt_dir, "metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False)
        except Exception as e:
            logger.warning("Не удалось записать metadata.json: %s", e)

        logger.info("Checkpoint saved: %s (tf_ckpt=%s) (reason=%s) hf_save=%s", ckpt_dir, tf_ckpt_prefix, reason, hf_save)

        # retention policy: keep last N non-best checkpoints
        try:
            _apply_retention_policy(base_dir=ckpts_dir, keep_last=getattr(config, "CHECKPOINTS_KEEP_LAST", 5))
        except Exception as e:
            logger.debug("Retention policy failed: %s", e)

        return ckpt_dir


def _apply_retention_policy(base_dir: str, keep_last: int = 5):
    """
    Удаляет старые чекпоинты, оставляя последние keep_last (и папку 'best' всегда сохраняем).
    Учитывает модификационное время директории и metadata epoch/batch при наличии.
    """
    if keep_last <= 0:
        return

    all_dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        if name == getattr(config, "BEST_CHECKPOINT_NAME", "best"):
            continue
        epoch = -1
        batch = -1
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                epoch = int(meta.get("epoch", -1) or -1)
                batch = int(meta.get("batch", -1) or -1)
            except Exception:
                epoch = -1
                batch = -1
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
        all_dirs.append({"path": path, "epoch": epoch, "batch": batch, "mtime": mtime, "name": name})

    # sort descending by (epoch, batch, mtime)
    all_dirs.sort(key=lambda x: (x["epoch"], x["batch"], x["mtime"]), reverse=True)
    to_keep = all_dirs[:keep_last]
    to_delete = all_dirs[keep_last:]

    for item in to_delete:
        try:
            _remove_dir_tree(item["path"])
            logger.info("Retention: removed old checkpoint %s", item["path"])
        except Exception as e:
            logger.warning("Retention: failed to remove %s: %s", item["path"], e)


def _remove_dir_tree(path: str):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def find_best_or_latest_checkpoint(ckpts_dir: str) -> Optional[str]:
    """
    См. прежняя реализация: выбирает 'best' если есть, иначе по metadata/mtime.
    """
    if not ckpts_dir or not os.path.isdir(ckpts_dir):
        return None

    candidates = []
    for name in os.listdir(ckpts_dir):
        path = os.path.join(ckpts_dir, name)
        if not os.path.isdir(path):
            continue
        meta_path = os.path.join(path, "metadata.json")
        meta = None
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
        except Exception:
            meta = None

        epoch = -1
        batch = -1
        if meta:
            try:
                epoch = int(meta.get("epoch", -1) or -1)
            except Exception:
                epoch = -1
            try:
                batch = int(meta.get("batch", -1) or -1)
            except Exception:
                batch = -1

        latest_tf = None
        try:
            latest_tf = tf.train.latest_checkpoint(path)
        except Exception:
            latest_tf = None

        has_hf = any(os.path.exists(os.path.join(path, fname)) for fname in ("config.json", "tf_model.h5", "pytorch_model.bin"))
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0

        candidates.append({"path": path, "name": name, "epoch": epoch, "batch": batch, "latest_tf": latest_tf, "has_hf": has_hf, "mtime": mtime})

    if not candidates:
        return None

    best_name = getattr(config, "BEST_CHECKPOINT_NAME", "best")
    for c in candidates:
        if c["name"] == best_name and (c["latest_tf"] or c["has_hf"]):
            logger.info("Selected 'best' checkpoint: %s", c["path"])
            return c["path"]

    candidates.sort(key=lambda x: (x["epoch"], x["batch"], x["mtime"]), reverse=True)
    for c in candidates:
        if c["latest_tf"] or c["has_hf"]:
            logger.info("Selected checkpoint by metadata/mtime: %s (epoch=%s batch=%s)", c["path"], c["epoch"], c["batch"])
            return c["path"]

    candidates.sort(key=lambda x: x["mtime"], reverse=True)
    logger.info("Returning most-recently modified checkpoint: %s", candidates[0]["path"])
    return candidates[0]["path"]


def restore_tf_checkpoint_from_folder(folder: str, ckpt: tf.train.Checkpoint) -> Tuple[bool, int, int]:
    """
    Попытаться восстановить TF checkpoint из папки.
    Возвращает (restored: bool, meta_epoch: int, meta_batch: int)
    """
    restored = False
    meta_epoch = 0
    meta_batch = 0
    meta_path = os.path.join(folder, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_epoch = int(meta.get("epoch", 0))
            meta_batch = int(meta.get("batch", 0) or 0)
            tf_ckpt_meta = meta.get("tf_checkpoint", None)
            if tf_ckpt_meta:
                try:
                    candidate = tf_ckpt_meta
                    if not os.path.isabs(candidate):
                        candidate_candidate = os.path.join(folder, os.path.basename(candidate))
                        if os.path.exists(candidate_candidate + ".index") or os.path.exists(candidate_candidate + ".data-00000-of-00001"):
                            candidate = candidate_candidate
                    if (os.path.exists(candidate + ".index") or os.path.exists(candidate + ".data-00000-of-00001")):
                        restore_status = ckpt.restore(candidate)
                        try:
                            restore_status.assert_existing_objects_matched()
                        except Exception:
                            logger.debug("assert_existing_objects_matched failed (non-fatal).")
                        logger.info("TF checkpoint restored from metadata.tf_checkpoint: %s", candidate)
                        restored = True
                except Exception as e:
                    logger.warning("Failed to restore TF checkpoint from metadata: %s", e)
        except Exception as e:
            logger.warning("Failed to read metadata.json: %s", e)

    if not restored:
        latest_tf = tf.train.latest_checkpoint(folder)
        if latest_tf:
            try:
                restore_status = ckpt.restore(latest_tf)
                try:
                    restore_status.assert_existing_objects_matched()
                except Exception:
                    logger.debug("assert_existing_objects_matched failed (non-fatal).")
                logger.info("TF checkpoint restored from latest_checkpoint %s", latest_tf)
                restored = True
            except Exception as e:
                logger.warning("tf.train.latest_checkpoint restore failed: %s", e)

    return restored, meta_epoch, meta_batch

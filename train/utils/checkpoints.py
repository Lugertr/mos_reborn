# coding: utf-8
"""
utils/checkpoints.py

Сохранение/восстановление чекпоинтов с поддержкой:
 - retention policy
 - асинхронной (ThreadPool) записью тяжёлых артефактов (accum_vars.npz)
 - опцией SAVE_ACCUM_VARS_ONLY_ON_EPOCH_END

Все комментарии и логи — на русском.
"""
import os
import json
import logging
import atexit
from typing import Optional, Tuple, List, Any
from concurrent.futures import ThreadPoolExecutor, Future, wait, ALL_COMPLETED

import numpy as np
import tensorflow as tf
import config  # type: ignore

logger = logging.getLogger(__name__)

# Глобальные переменные модуля
_save_lock = None  # просто оставляем None, т.к. ThreadPoolExecutor управляет конкурентностью
_executor: Optional[ThreadPoolExecutor] = None
_futures: List[Future] = []


def _ensure_executor():
    """Инициализация ThreadPoolExecutor (лениво)."""
    global _executor
    if _executor is None:
        max_workers = int(getattr(config, "CHECKPOINTS_THREADPOOL_MAX_WORKERS", 2) or 2)
        _executor = ThreadPoolExecutor(max_workers=max_workers)
        # регистрируем ожидание при выходе процесса
        atexit.register(wait_for_background_tasks)
        logger.debug("ThreadPoolExecutor инициализирован (max_workers=%d)", max_workers)


def _write_accum_vars_and_update_meta(ckpt_dir: str,
                                      accum_vars: List[Any],
                                      accum_counter: int):
    """
    Синхронная запись accum_vars (npz) и обновление metadata.json.
    Эта функция вызывается внутри фонового потока.
    """
    try:
        arr_dict = {}
        for i, v in enumerate(accum_vars):
            try:
                if hasattr(v, "numpy"):
                    arr = v.numpy()
                else:
                    arr = np.asarray(v)
            except Exception:
                try:
                    arr = tf.keras.backend.get_value(v)
                except Exception:
                    arr = None
            if arr is None:
                logger.warning("Не удалось получить accum_var[%d] для сохранения — пропускаем.", i)
                continue
            arr_dict[f"v{i}"] = arr

        if not arr_dict:
            logger.info("accum_vars пусты — не сохраняем npz.")
            return

        accum_path = os.path.join(ckpt_dir, "accum_vars.npz")
        np.savez_compressed(accum_path, **arr_dict)
        logger.info("Фоновая запись accum_vars завершена: %s", accum_path)

        # Обновляем metadata.json (добавляем имя npz и accum_counter)
        meta_path = os.path.join(ckpt_dir, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
            except Exception:
                meta = {}
        meta["accum_vars_npz"] = os.path.basename(accum_path)
        meta["accum_counter"] = int(accum_counter)
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False)
            logger.debug("metadata.json обновлён (accum_vars_npz, accum_counter) в %s", meta_path)
        except Exception as e:
            logger.exception("Не удалось обновить metadata.json: %s", e)

    except Exception as e:
        logger.exception("Ошибка в фоновой функции записи accum_vars: %s", e)


def _submit_background_task(fn, *args) -> None:
    """Отправить задачу в пул, отслеживая future."""
    _ensure_executor()
    try:
        fut = _executor.submit(fn, *args)
        _futures.append(fut)

        # при завершении — логировать и убирать из списка
        def _on_done(f: Future):
            try:
                _futures.remove(f)
            except Exception:
                pass
            try:
                exc = f.exception(timeout=0)
                if exc:
                    logger.exception("Фоновая задача завершилась с ошибкой: %s", exc)
            except Exception:
                # если .exception() блокирует, игнорируем
                pass

        fut.add_done_callback(_on_done)
    except Exception:
        logger.exception("Не удалось запустить фоновую задачу.")


def wait_for_background_tasks(timeout: Optional[float] = None):
    """
    Ожидать завершения всех фоновых задач. Вызывается при завершении процесса.
    """
    global _executor
    if not _futures:
        return
    logger.info("Ожидаем завершения %d фоновых задач сохранения...", len(_futures))
    try:
        wait(list(_futures), timeout=timeout, return_when=ALL_COMPLETED)
    except Exception:
        logger.exception("Ошибка при ожидании фоновых задач.")
    # Дополнительно можно корректно завершить executor
    try:
        if _executor:
            _executor.shutdown(wait=False)
    except Exception:
        pass


def save_checkpoint(epoch: int,
                    batch_in_epoch: int,
                    model,
                    processor,
                    optimizer,
                    ckpt_obj,
                    global_step: int = 0,
                    global_examples: int = 0,
                    accum_vars: Optional[List[Any]] = None,
                    accum_counter: int = 0,
                    reason: str = "auto",
                    hf_save: bool = True) -> str:
    """
    Сохранить чекпоинт:
      - HF-style (model/processor) если hf_save=True
      - TF checkpoint синхронно (ckpt_obj.save)
      - metadata.json пишется сразу, но heavy файл accum_vars.npz — фоново через ThreadPoolExecutor,
        если config.SAVE_ACCUM_VARS==True и правила SAVE_ACCUM_VARS_ONLY_ON_EPOCH_END позволяют.
    Возвращает путь к сохранённой папке.
    """
    ckpts_dir = getattr(config, "CHECKPOINTS_DIR")
    os.makedirs(ckpts_dir, exist_ok=True)

    suffix = f"-partial-{batch_in_epoch}" if batch_in_epoch and batch_in_epoch > 0 else ""
    ckpt_name = f"{getattr(config, 'CHECKPOINT_PREFIX', 'checkpoint-')}{epoch}{suffix}"
    ckpt_dir = os.path.join(ckpts_dir, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    tf_ckpt_prefix = None

    # HF-style save (опционально)
    if hf_save:
        try:
            model.save_pretrained(ckpt_dir)
        except Exception:
            logger.debug("model.save_pretrained не сработал (non-fatal).")
        try:
            processor.save_pretrained(ckpt_dir)
        except Exception:
            try:
                if hasattr(processor, "tokenizer"):
                    processor.tokenizer.save_pretrained(ckpt_dir)
            except Exception:
                logger.debug("Не удалось сохранить tokenizer (non-fatal).")

    # TF checkpoint (сохраняем синхронно)
    try:
        if ckpt_obj is None:
            ckpt_obj = tf.train.Checkpoint(optimizer=optimizer, model=model)
        tf_ckpt_prefix = ckpt_obj.save(os.path.join(ckpt_dir, "tf_ckpt"))
    except Exception as e:
        logger.warning("TF checkpoint сохранить не удалось: %s", e)

    # metadata без имени npz (если npz будет записан фоново — он запишется позже)
    meta = {
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
        "global_examples": int(global_examples),
        "accum_counter": int(accum_counter),
        "accum_vars_npz": None,
        "tf_checkpoint": tf_ckpt_prefix,
        "reason": reason
    }
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False)
    except Exception:
        logger.warning("Не удалось записать metadata.json (non-fatal).")

    logger.info("Сохранён чекпоинт: %s (tf_ckpt=%s) reason=%s global_step=%d examples=%d batch_in_epoch=%d",
                ckpt_dir, tf_ckpt_prefix, reason, meta["global_step"], meta["global_examples"], meta["batch_in_epoch"])

    # Решаем, сохранять ли accum_vars: учитываем глобальный флаг и опцию only-on-epoch-end
    save_accum_enabled = bool(getattr(config, "SAVE_ACCUM_VARS", True))
    save_only_on_epoch_end = bool(getattr(config, "SAVE_ACCUM_VARS_ONLY_ON_EPOCH_END", False))
    should_save_accum = save_accum_enabled and accum_vars is not None
    if should_save_accum and save_only_on_epoch_end:
        # сохраняем accum_vars только если событие — конец эпохи, либо это signal/keyboard-interrupt, либо hf_save True
        lower_reason = (reason or "").lower()
        if not (reason == "epoch-end" or lower_reason.startswith("signal") or reason == "keyboard-interrupt" or hf_save):
            should_save_accum = False

    if should_save_accum and accum_vars:
        try:
            # ставим задачу в пул на фоновое выполнение
            _submit_background_task(_write_accum_vars_and_update_meta, ckpt_dir, accum_vars, int(accum_counter))
            logger.debug("accum_vars отправлены в фон (ThreadPool) для сохранения.")
        except Exception:
            logger.exception("Не удалось отправить accum_vars в ThreadPool. Попробуем синхронно.")
            try:
                _write_accum_vars_and_update_meta(ckpt_dir, accum_vars, int(accum_counter))
            except Exception:
                logger.exception("Синхронное сохранение accum_vars тоже не удалось.")

    # retention policy (синхронно)
    try:
        _apply_retention_policy(base_dir=ckpts_dir, keep_last=getattr(config, "CHECKPOINTS_KEEP_LAST", 5))
    except Exception:
        logger.exception("Ошибка при применении retention policy.")

    return ckpt_dir


def _apply_retention_policy(base_dir: str, keep_last: int = 5):
    """
    Удалить старые чекпоинты (кроме 'best'), оставив keep_last последних.
    Сортировка по (epoch, batch, mtime).
    """
    if keep_last <= 0:
        return

    entries = []
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
                batch = int(meta.get("batch_in_epoch", -1) or -1)
            except Exception:
                epoch = -1
                batch = -1
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        entries.append({"path": path, "epoch": epoch, "batch": batch, "mtime": mtime, "name": name})

    entries.sort(key=lambda x: (x["epoch"], x["batch"], x["mtime"]), reverse=True)
    to_remove = entries[getattr(config, "CHECKPOINTS_KEEP_LAST", 5):]
    for item in to_remove:
        try:
            import shutil
            shutil.rmtree(item["path"], ignore_errors=True)
            logger.info("Retention: удалён старый чекпоинт %s", item["path"])
        except Exception:
            logger.exception("Retention: не удалось удалить %s", item["path"])


def find_best_or_latest_checkpoint(ckpts_dir: str) -> Optional[str]:
    """
    Выбрать папку 'best' если есть, иначе по (epoch,batch,mtime), иначе most-recently modified.
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
        if os.path.exists(meta_path):
            try:
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
                batch = int(meta.get("batch_in_epoch", -1) or -1)
            except Exception:
                batch = -1

        latest_tf = None
        try:
            latest_tf = tf.train.latest_checkpoint(path)
        except Exception:
            latest_tf = None

        has_hf = any(os.path.exists(os.path.join(path, f)) for f in ("config.json", "tf_model.h5", "pytorch_model.bin"))
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
            logger.info("Выбран 'best' чекпоинт: %s", c["path"])
            return c["path"]

    candidates.sort(key=lambda x: (x["epoch"], x["batch"], x["mtime"]), reverse=True)
    for c in candidates:
        if c["latest_tf"] or c["has_hf"]:
            logger.info("Выбран чекпоинт по metadata/mtime: %s (epoch=%s batch=%s)", c["path"], c["epoch"], c["batch"])
            return c["path"]

    candidates.sort(key=lambda x: x["mtime"], reverse=True)
    logger.info("Возвращаем наиболее недавно изменённый чекпоинт: %s", candidates[0]["path"])
    return candidates[0]["path"]


def restore_tf_checkpoint_from_folder(folder: str, ckpt: tf.train.Checkpoint) -> Tuple[bool, int, int, int, int, Optional[List[np.ndarray]], int]:
    """
    Восстановить TF checkpoint из папки folder.
    Возвращает tuple:
      (restored, epoch, batch_in_epoch, global_step, global_examples, accum_vars_list_or_None, accum_counter)
    """
    restored = False
    meta_epoch = 0
    meta_batch = 0
    meta_global_step = 0
    meta_global_examples = 0
    accum_vars_list = None
    accum_counter = 0

    meta_path = os.path.join(folder, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_epoch = int(meta.get("epoch", 0) or 0)
            meta_batch = int(meta.get("batch_in_epoch", 0) or 0)
            meta_global_step = int(meta.get("global_step", 0) or 0)
            meta_global_examples = int(meta.get("global_examples", 0) or 0)
            accum_counter = int(meta.get("accum_counter", 0) or 0)
            accum_np_name = meta.get("accum_vars_npz", None)
            tf_ckpt_meta = meta.get("tf_checkpoint", None)

            # Восстановление TF checkpoint по явной ссылке
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
                            logger.debug("assert_existing_objects_matched не прошёл (non-fatal).")
                        logger.info("TF чекпоинт восстановлен из metadata.tf_checkpoint: %s", candidate)
                        restored = True
                except Exception as e:
                    logger.warning("Не удалось восстановить TF чекпоинт из metadata: %s", e)

            # Если в metadata есть имя npz — пробуем загрузить accum_vars
            if accum_np_name:
                try:
                    accum_np_path = os.path.join(folder, accum_np_name)
                    if os.path.exists(accum_np_path):
                        loaded = np.load(accum_np_path)
                        arrs = []
                        i = 0
                        while f"v{i}" in loaded:
                            arrs.append(loaded[f"v{i}"])
                            i += 1
                        if arrs:
                            accum_vars_list = arrs
                            logger.info("Загружены accum_vars из %s (элементов=%d)", accum_np_path, len(arrs))
                except Exception as e:
                    logger.exception("Ошибка при загрузке accum_vars.npz: %s", e)
        except Exception as e:
            logger.warning("Не удалось прочитать metadata.json: %s", e)

    # fallback: latest_checkpoint
    if not restored:
        try:
            latest_tf = tf.train.latest_checkpoint(folder)
            if latest_tf:
                restore_status = ckpt.restore(latest_tf)
                try:
                    restore_status.assert_existing_objects_matched()
                except Exception:
                    logger.debug("assert_existing_objects_matched не прошёл (non-fatal).")
                logger.info("TF чекпоинт восстановлен из latest_checkpoint: %s", latest_tf)
                restored = True
        except Exception as e:
            logger.warning("tf.train.latest_checkpoint restore failed: %s", e)

    return restored, meta_epoch, meta_batch, meta_global_step, meta_global_examples, accum_vars_list, accum_counter

#!/usr/bin/env python3
# train_trocr.py
# coding: utf-8
"""
Основной скрипт тренировки TrOCR (TensorFlow).

Особенности:
 - tqdm прогресс-бар с точным оставшимся количеством примеров
 - опциональный TFRecord pipeline
 - gradient accumulation внутри @tf.function
 - mixed-precision опционально
 - асинхронное/фоновое сохранение чекпоинтов (через utils.checkpoints)
 - восстановление состояния включая accum_vars и accum_counter
 - установка параметра генерации через GenerationConfig (чтобы не получать warning)
"""
import os
import sys
import logging
import signal
import tempfile
import json
import warnings
from typing import Optional, List

this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

# Подавление шумных предупреждений (конкретные сообщения)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=INFO,2=WARNING,3=ERROR
warnings.filterwarnings("ignore", message="Using a slow image processor as `use_fast` is unset", category=FutureWarning)
warnings.filterwarnings("ignore", message="The class ViTFeatureExtractor is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="TensorFlow and JAX classes are deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="The name tf.get_default_graph is deprecated", category=DeprecationWarning)

try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

import config  # type: ignore
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import set_seed, TFVisionEncoderDecoderModel, GenerationConfig

from utils import checkpoints, tokenizer_utils, data as data_utils, model_utils
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# глобальный state (для signal handler и фоновых сохранений)
_global_state = {
    "epoch": 0,
    "batch": 0,
    "model": None,
    "processor": None,
    "optimizer": None,
    "ckpt": None,
    "global_step": 0,
    "global_examples": 0,
    "accum_vars": None,
    "accum_counter": None,
}


def _signal_save_and_exit(signum, frame):
    """Обработчик сигналов: сохраняет чекпоинт и ожидает фоновых задач."""
    logger.info("Сигнал %s получен — сохраняем чекпоинт...", signum)
    try:
        epoch = int(_global_state.get("epoch", 0))
        batch_in_epoch = int(_global_state.get("batch", 0))
        model = _global_state.get("model", None)
        processor = _global_state.get("processor", None)
        optimizer = _global_state.get("optimizer", None)
        ckpt_obj = _global_state.get("ckpt", None)
        global_step = int(_global_state.get("global_step", 0))
        global_examples = int(_global_state.get("global_examples", 0))
        accum_vars = _global_state.get("accum_vars", None)
        accum_counter_var = _global_state.get("accum_counter", None)
        accum_counter_val = 0
        if accum_counter_var is not None:
            try:
                accum_counter_val = int(accum_counter_var.numpy())
            except Exception:
                try:
                    accum_counter_val = int(tf.keras.backend.get_value(accum_counter_var))
                except Exception:
                    accum_counter_val = 0

        if model is None or processor is None or optimizer is None:
            logger.warning("Недостаточно данных для сохранения чекпоинта (model/processor/optimizer отсутствуют).")
        else:
            checkpoints.save_checkpoint(epoch=epoch, batch_in_epoch=batch_in_epoch,
                                        model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt_obj,
                                        global_step=global_step, global_examples=global_examples,
                                        accum_vars=accum_vars if getattr(config, "SAVE_ACCUM_VARS", True) else None,
                                        accum_counter=accum_counter_val,
                                        reason=f"signal-{signum}", hf_save=True)
            checkpoints.wait_for_background_tasks()
    except Exception:
        logger.exception("Ошибка при сохранении чекпоинта в обработчике сигнала.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_save_and_exit)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _signal_save_and_exit)


def train():
    set_seed(getattr(config, "SEED", 42))

    # GPU memory growth
    if getattr(config, "ENABLE_TF_GPU_MEMORY_GROWTH", True):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
                logger.info("Используются GPU: %s", gpus)
            except Exception:
                logger.exception("Не удалось включить memory growth для GPU")
        else:
            logger.info("GPU не найден — используется CPU")

    # Mixed precision (опционально)
    use_mixed = getattr(config, "ENABLE_MIXED_PRECISION", False)
    if use_mixed:
        try:
            from tensorflow.keras.mixed_precision import set_global_policy
            set_global_policy("mixed_float16")
            logger.info("Включена mixed precision (mixed_float16).")
        except Exception:
            logger.exception("Не удалось включить mixed precision; продолжаем без неё.")
            use_mixed = False

    # Попытка включить XLA
    try:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA JIT включён.")
    except Exception:
        logger.debug("XLA JIT не доступен или включить не удалось.")

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    logger.info("Устройство: %s", device)

    # Загрузка датасетов
    train_json = getattr(config, "TRAIN_JSON_PATH")
    test_json = getattr(config, "TEST_JSON_PATH")
    logger.info("TRAIN_JSON_PATH = %s", train_json)
    logger.info("TEST_JSON_PATH  = %s", test_json)
    if not os.path.exists(train_json) or not os.path.exists(test_json):
        raise FileNotFoundError("TRAIN/TEST json не найдены — проверьте config")

    data_files = {"train": train_json, "test": test_json}
    try:
        hf_ds = load_dataset("json", data_files=data_files)
    except Exception as e:
        logger.info("Первичная загрузка json не удалась (%s) — пробуем field='data' ...", e)
        hf_ds = load_dataset("json", data_files=data_files, field="data")

    if getattr(config, "DEBUG_TRAIN_SAMPLES", 0) > 0:
        hf_ds["train"] = hf_ds["train"].select(range(min(config.DEBUG_TRAIN_SAMPLES, len(hf_ds["train"]))))
    if getattr(config, "DEBUG_EVAL_SAMPLES", 0) > 0:
        hf_ds["test"] = hf_ds["test"].select(range(min(config.DEBUG_EVAL_SAMPLES, len(hf_ds["test"]))))

    total_train_examples = len(hf_ds["train"])
    logger.info("Всего примеров в train: %d", total_train_examples)

    # Prepare / load processor (tokenizer + ViTImageProcessor)
    processor = None
    if getattr(config, "TRAIN_TOKENIZER_FROM_TEST_SPLIT", False):
        texts = []
        for ex in hf_ds["test"]:
            t = ex.get("text") or ex.get("transcription") or ex.get("label") or ""
            if t:
                texts.append(t)
        if not texts:
            raise RuntimeError("TRAIN_TOKENIZER_FROM_TEST_SPLIT=True, но в test.json нет текстов")
        tmp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt")
        tmp_path = tmp_file.name
        try:
            for line in texts:
                tmp_file.write(line.replace("\n", " ") + "\n")
            tmp_file.close()
            tokenizer_obj = tokenizer_utils.train_bpe_tokenizer(corpus_paths=[tmp_path],
                                                               vocab_size=getattr(config, "TOKENIZER_VOCAB_SIZE", 8000),
                                                               special_tokens=getattr(config, "TOKENIZER_SPECIAL_TOKENS"))
            processor = tokenizer_utils.save_processor_from_tokenizer(tokenizer_obj,
                                                                      out_dir=getattr(config, "TOKENIZER_OUT_DIR"),
                                                                      special_tokens=getattr(config, "TOKENIZER_SPECIAL_TOKENS"),
                                                                      image_size=getattr(config, "TOKENIZER_IMAGE_SIZE"))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.debug("Не удалось удалить временный файл корпуса")
    else:
        processor = tokenizer_utils.load_processor_safe(getattr(config, "TOKENIZER_OUT_DIR"))
    logger.info("Processor готов.")

    # Подготовка директорий и оптимизатора
    os.makedirs(getattr(config, "CHECKPOINTS_DIR"), exist_ok=True)
    os.makedirs(getattr(config, "LOG_DIR"), exist_ok=True)
    os.makedirs(getattr(config, "OUTPUT_DIR"), exist_ok=True)

    base_opt = tf.keras.optimizers.Adam(learning_rate=getattr(config, "LEARNING_RATE", 5e-5))
    if use_mixed:
        try:
            from tensorflow.keras.mixed_precision import LossScaleOptimizer
            optimizer = LossScaleOptimizer(base_opt)
            logger.info("Оптимизатор: LossScaleOptimizer (mixed precision).")
        except Exception:
            optimizer = base_opt
            logger.warning("Не удалось обернуть оптимизатор в LossScaleOptimizer; используем базовый.")
    else:
        optimizer = base_opt

    # Восстановление из чекпоинта
    latest_folder = checkpoints.find_best_or_latest_checkpoint(getattr(config, "CHECKPOINTS_DIR"))
    start_epoch = 1
    resume_batch = 0
    model = None
    ckpt = None
    global_step = 0
    global_examples_processed = 0
    restored_accum_vars = None
    restored_accum_counter = 0

    if latest_folder:
        logger.info("Найден чекпоинт-подкаталог: %s — пробуем восстановиться.", latest_folder)
        try:
            model = TFVisionEncoderDecoderModel.from_pretrained(latest_folder, local_files_only=True)
            logger.info("Модель загружена через from_pretrained(%s).", latest_folder)
            try:
                processor = tokenizer_utils.load_processor_safe(latest_folder)
            except Exception:
                logger.debug("Не удалось загрузить processor из чекпоинта (non-fatal).")
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            (restored, meta_epoch, meta_batch, meta_global_step, meta_global_examples,
             accum_vars_list, accum_counter) = checkpoints.restore_tf_checkpoint_from_folder(latest_folder, ckpt)
            if restored:
                start_epoch = max(1, int(meta_epoch))
                resume_batch = int(meta_batch or 0)
                global_step = int(meta_global_step or 0)
                global_examples_processed = int(meta_global_examples or 0)
                restored_accum_vars = accum_vars_list
                restored_accum_counter = int(accum_counter or 0)
                logger.info("Восстановлено: epoch=%d batch_in_epoch=%d global_step=%d global_examples=%d accum_counter=%d",
                            start_epoch, resume_batch, global_step, global_examples_processed, restored_accum_counter)
            else:
                logger.info("TF restore не сработал в %s.", latest_folder)
        except Exception as e:
            logger.info("from_pretrained не сработал (%s) — создаём модель заново.", e)
            model = None

    if model is None:
        model = model_utils.build_model_scratch(processor)
        logger.info("Модель создана с нуля.")
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if latest_folder:
            (restored, meta_epoch, meta_batch, meta_global_step, meta_global_examples,
             accum_vars_list, accum_counter) = checkpoints.restore_tf_checkpoint_from_folder(latest_folder, ckpt)
            if restored:
                start_epoch = max(1, int(meta_epoch))
                resume_batch = int(meta_batch or 0)
                global_step = int(meta_global_step or 0)
                global_examples_processed = int(meta_global_examples or 0)
                restored_accum_vars = accum_vars_list
                restored_accum_counter = int(accum_counter or 0)
                logger.info("TF чекпоинт восстановлен: epoch=%d batch_in_epoch=%d global_step=%d global_examples=%d accum_counter=%d",
                            start_epoch, resume_batch, global_step, global_examples_processed, restored_accum_counter)

    # Обновляем глобальный state
    _global_state.update({"model": model, "processor": processor, "optimizer": optimizer, "ckpt": ckpt,
                          "global_step": global_step, "global_examples": global_examples_processed})

    # Token ids и генерация: помещаем max_length в generation_config, чтобы убрать предупреждение
    if model.config.pad_token_id is None:
        model.config.pad_token_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = int(getattr(processor.tokenizer, "bos_token_id", None) or 0)
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = int(getattr(processor.tokenizer, "eos_token_id", None) or 2)

    # Устанавливаем GenerationConfig.max_length (рекомендуемый путь в HF)
    gen_max_len = int(getattr(config, "GENERATION_MAX_LENGTH", 128))
    try:
        # Если уже есть generation_config, обновим его; иначе создадим новый
        existing_gen = {}
        if hasattr(model, "generation_config") and model.generation_config is not None:
            try:
                existing_gen = model.generation_config.to_dict() if hasattr(model.generation_config, "to_dict") else {}
            except Exception:
                existing_gen = {}
        existing_gen.update({"max_length": gen_max_len})
        model.generation_config = GenerationConfig(**existing_gen)
        logger.info("Параметры генерации заданы в model.generation_config (max_length=%d).", gen_max_len)
    except Exception:
        # fallback для старых версий transformers
        model.config.max_length = gen_max_len
        logger.warning("Не удалось создать GenerationConfig — установлен model.config.max_length=%d (fallback).", gen_max_len)

    # Подготовка датасетов
    train_images_dir = getattr(config, "TRAIN_IMAGES_DIR")
    test_images_dir = getattr(config, "TEST_IMAGES_DIR")
    train_ds = data_utils.prepare_tf_dataset(hf_ds["train"], processor, train_images_dir,
                                             batch_size=getattr(config, "PER_DEVICE_TRAIN_BATCH_SIZE", 1), shuffle=True)
    eval_ds = data_utils.prepare_tf_dataset(hf_ds["test"], processor, test_images_dir,
                                            batch_size=getattr(config, "PER_DEVICE_EVAL_BATCH_SIZE", 1), shuffle=False)

    tb_writer = tf.summary.create_file_writer(getattr(config, "LOG_DIR"))

    best_wer = float("inf")
    best_ckpt = None

    grad_clip = getattr(config, "GRAD_CLIP_NORM", None)
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    total_epochs = getattr(config, "NUM_TRAIN_EPOCHS", 3)

    accumulation_steps = max(1, getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1))
    logger.info("Обучение начнётся с эпохи %d (всего %d). resume_batch=%d accumulation_steps=%d",
                start_epoch, total_epochs, resume_batch, accumulation_steps)

    save_every_steps = getattr(config, "SAVE_CHECKPOINT_EVERY_STEPS", 0)

    # счётчики
    global_step = int(global_step)
    global_examples_processed = int(global_examples_processed)
    _global_state["global_step"] = global_step
    _global_state["global_examples"] = global_examples_processed

    # accum vars + counter: восстановление или создание новых
    def _make_accum_vars(init_list: Optional[List[np.ndarray]], model_vars: List[tf.Variable]) -> List[tf.Variable]:
        accum = []
        if init_list is not None and len(init_list) == len(model_vars):
            for i, v in enumerate(model_vars):
                arr = init_list[i]
                try:
                    tf_arr = tf.convert_to_tensor(arr, dtype=v.dtype)
                    accum_var = tf.Variable(tf_arr, trainable=False, dtype=v.dtype)
                except Exception:
                    accum_var = tf.Variable(tf.zeros_like(v), trainable=False, dtype=v.dtype)
                accum.append(accum_var)
        else:
            for v in model_vars:
                accum.append(tf.Variable(tf.zeros_like(v), trainable=False, dtype=v.dtype))
        return accum

    if restored_accum_vars is not None:
        accum_vars = _make_accum_vars(restored_accum_vars, model.trainable_variables)
    else:
        accum_vars = _make_accum_vars(None, model.trainable_variables)
    accum_counter = tf.Variable(restored_accum_counter if restored_accum_counter is not None else 0, trainable=False, dtype=tf.int32)

    _global_state["accum_vars"] = accum_vars
    _global_state["accum_counter"] = accum_counter

    # tf.function signature для уменьшения retracing
    pv_spec = train_ds.element_spec[0]
    try:
        pv_shape = pv_spec.shape.as_list()[1:]
    except Exception:
        pv_shape = list(pv_spec.shape)[1:]
    pv_sig = tf.TensorSpec(shape=[None] + pv_shape, dtype=tf.float32)
    lbl_sig = tf.TensorSpec(shape=[None, None], dtype=tf.int32)

    @tf.function(input_signature=[pv_sig, lbl_sig], reduce_retracing=True)
    def accum_train_step(pv, lb):
        pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
        labels_masked = tf.where(tf.equal(lb, pad_id), tf.constant(-100, dtype=lb.dtype), lb)
        with tf.GradientTape() as tape:
            outputs = model(pixel_values=pv, labels=labels_masked, training=True)
            loss = outputs.loss
            loss = loss / tf.cast(accumulation_steps, loss.dtype)
            if hasattr(optimizer, "get_scaled_loss"):
                scaled_loss = optimizer.get_scaled_loss(loss)  # type: ignore
            else:
                scaled_loss = loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)  # type: ignore
        for i, g in enumerate(grads):
            if g is None:
                continue
            if grad_clip is not None:
                g = tf.clip_by_norm(g, float(grad_clip))
            if g.dtype != accum_vars[i].dtype:
                g = tf.cast(g, accum_vars[i].dtype)
            accum_vars[i].assign_add(g)
        accum_counter.assign_add(1)
        apply_now = tf.equal(accum_counter, accumulation_steps)

        def _apply_and_reset():
            grads_and_vars = []
            for i, v in enumerate(model.trainable_variables):
                g = accum_vars[i]
                grads_and_vars.append((g, v))
            optimizer.apply_gradients(grads_and_vars)
            for i in range(len(accum_vars)):
                accum_vars[i].assign(tf.zeros_like(accum_vars[i]))
            accum_counter.assign(0)
            return tf.constant(1)

        tf.cond(apply_now, _apply_and_reset, lambda: tf.constant(0))
        return loss

    # tqdm прогресс-бар
    pbar = tqdm(total=total_train_examples, initial=global_examples_processed, unit="ex", desc="Train", leave=True)

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            _global_state["epoch"] = epoch
            logger.info("=== Эпоха %d/%d ===", epoch, total_epochs)
            train_loss_metric.reset_states()
            epoch_steps = 0

            ds_iter = train_ds
            if epoch == start_epoch and resume_batch and resume_batch > 0:
                logger.info("Возобновление эпохи %d: пропускаем %d батч(ей) ...", epoch, resume_batch)
                ds_iter = train_ds.skip(resume_batch)
                epoch_steps = int(resume_batch)
                resume_batch = 0

            try:
                for pv, lb in ds_iter:
                    epoch_steps += 1
                    global_step += 1

                    # число примеров в батче
                    try:
                        batch_examples = int(tf.shape(pv)[0].numpy())
                    except Exception:
                        batch_examples = int(pv.shape[0]) if pv.shape[0] is not None else 0
                    global_examples_processed += batch_examples

                    # обновление прогресс-бара
                    pbar.update(batch_examples)

                    # обновление глобального состояния
                    _global_state["batch"] = epoch_steps
                    _global_state["global_step"] = global_step
                    _global_state["global_examples"] = global_examples_processed
                    _global_state["accum_vars"] = accum_vars
                    _global_state["accum_counter"] = accum_counter

                    # шаг
                    loss = accum_train_step(pv, lb)

                    # извлечение loss
                    try:
                        loss_val = float(loss.numpy().item())
                    except Exception:
                        try:
                            loss_val = float(tf.keras.backend.get_value(loss))
                        except Exception:
                            loss_val = 0.0
                    train_loss_metric.update_state(loss_val * float(accumulation_steps))

                    # логирование
                    if global_step % getattr(config, "LOGGING_STEPS", 20) == 0:
                        avg_so_far = float(train_loss_metric.result().numpy())
                        remaining_examples = max(0, total_train_examples - global_examples_processed)
                        try:
                            accum_ctr_val = int(accum_counter.numpy())
                        except Exception:
                            accum_ctr_val = int(tf.keras.backend.get_value(accum_counter))
                        logger.info("Epoch %d step %d (global %d) — обработано %d/%d — осталось %d — accum_counter=%d — avg loss %.6f",
                                    epoch, epoch_steps, global_step, global_examples_processed, total_train_examples, remaining_examples, accum_ctr_val, avg_so_far)

                    # Periodic save (TF-only)
                    if save_every_steps and (global_step % save_every_steps == 0):
                        try:
                            accum_to_save = accum_vars if getattr(config, "SAVE_ACCUM_VARS", True) else None
                            try:
                                accum_ctr_val = int(accum_counter.numpy())
                            except Exception:
                                accum_ctr_val = int(tf.keras.backend.get_value(accum_counter))
                            logger.info("Периодическое сохранение: global_step=%d (epoch=%d step=%d) accum_counter=%d", global_step, epoch, epoch_steps, accum_ctr_val)
                            checkpoints.save_checkpoint(epoch=epoch, batch_in_epoch=epoch_steps,
                                                        model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt,
                                                        global_step=global_step, global_examples=global_examples_processed,
                                                        accum_vars=accum_to_save, accum_counter=accum_ctr_val,
                                                        reason="periodic", hf_save=False)
                        except Exception:
                            logger.exception("Периодическое сохранение не удалось.")
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — сохраняем прогресс и выходим.")
                try:
                    accum_ctr_val = int(accum_counter.numpy())
                except Exception:
                    accum_ctr_val = int(tf.keras.backend.get_value(accum_counter))
                checkpoints.save_checkpoint(epoch=epoch, batch_in_epoch=epoch_steps,
                                            model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt,
                                            global_step=global_step, global_examples=global_examples_processed,
                                            accum_vars=accum_vars if getattr(config, "SAVE_ACCUM_VARS", True) else None,
                                            accum_counter=accum_ctr_val,
                                            reason="keyboard-interrupt", hf_save=True)
                checkpoints.wait_for_background_tasks()
                raise

            avg_loss = float(train_loss_metric.result().numpy())
            logger.info("Эпоха %d завершена, средний loss %.6f. Обработано %d примеров (global_step=%d).",
                        epoch, avg_loss, global_examples_processed, global_step)

            # Evaluation
            metrics = model_utils.evaluate_and_log(model, processor, eval_ds, tb_writer, epoch, tb_examples=getattr(config, "TB_EXAMPLES_TO_LOG", 5))
            logger.info("Eval после эпохи %d: %s", epoch, metrics)

            # Save checkpoint at epoch end (HF + TF); accum_vars saved per config
            try:
                try:
                    accum_ctr_val = int(accum_counter.numpy())
                except Exception:
                    accum_ctr_val = int(tf.keras.backend.get_value(accum_counter))
                saved_ckpt_dir = checkpoints.save_checkpoint(epoch=epoch, batch_in_epoch=0,
                                                             model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt,
                                                             global_step=global_step, global_examples=global_examples_processed,
                                                             accum_vars=accum_vars if getattr(config, "SAVE_ACCUM_VARS", True) else None,
                                                             accum_counter=accum_ctr_val,
                                                             reason="epoch-end", hf_save=True)
            except Exception:
                logger.exception("Сохранение чекпоинта в конце эпохи не удалось.")
                saved_ckpt_dir = None

            # Save best by WER
            wer = metrics.get("wer", None)
            if wer is not None and wer < best_wer:
                best_wer = wer
                best_ckpt = saved_ckpt_dir or getattr(config, "CHECKPOINTS_DIR")
                best_dir = os.path.join(getattr(config, "CHECKPOINTS_DIR"), getattr(config, "BEST_CHECKPOINT_NAME", "best"))
                os.makedirs(best_dir, exist_ok=True)
                try:
                    model.save_pretrained(best_dir)
                except Exception:
                    logger.debug("model.save_pretrained(best) не удалась (non-fatal).")
                try:
                    processor.save_pretrained(best_dir)
                except Exception:
                    if hasattr(processor, "tokenizer"):
                        try:
                            processor.tokenizer.save_pretrained(best_dir)
                        except Exception:
                            logger.debug("Не удалось сохранить tokenizer в best.")
                try:
                    ckpt_best = tf.train.Checkpoint(optimizer=optimizer, model=model)
                    tf_ckpt_best = ckpt_best.save(os.path.join(best_dir, "tf_ckpt"))
                except Exception:
                    tf_ckpt_best = None
                    logger.warning("TF save для best чекпоинта не удался.")
                try:
                    accum_ctr_val = int(accum_counter.numpy())
                except Exception:
                    accum_ctr_val = int(tf.keras.backend.get_value(accum_counter))
                meta_best = {"epoch": epoch, "batch_in_epoch": 0, "global_step": global_step, "global_examples": global_examples_processed, "accum_counter": accum_ctr_val, "tf_checkpoint": tf_ckpt_best}
                try:
                    if getattr(config, "SAVE_ACCUM_VARS", True):
                        arr_dict = {}
                        for i, v in enumerate(accum_vars):
                            try:
                                arr = v.numpy()
                            except Exception:
                                try:
                                    arr = tf.keras.backend.get_value(v)
                                except Exception:
                                    arr = None
                            if arr is not None:
                                arr_dict[f"v{i}"] = arr
                        if arr_dict:
                            accum_path = os.path.join(best_dir, "accum_vars.npz")
                            np.savez_compressed(accum_path, **arr_dict)
                            meta_best["accum_vars_npz"] = os.path.basename(accum_path)
                    with open(os.path.join(best_dir, "metadata.json"), "w", encoding="utf-8") as mf:
                        json.dump(meta_best, mf, ensure_ascii=False)
                except Exception:
                    logger.warning("Не удалось записать metadata.json для best.")
                logger.info("Сохранена лучшая модель (WER=%.6f) в %s", best_wer, best_dir)

    finally:
        try:
            pbar.close()
        except Exception:
            pass
        logger.info("Ожидаем фоновых задач сохранения (если есть)...")
        checkpoints.wait_for_background_tasks()

    # Финальная модель
    logger.info("Сохраняем финальную модель в %s", getattr(config, "OUTPUT_DIR"))
    model_utils.save_final_model_and_processor(model, processor, getattr(config, "OUTPUT_DIR"))
    if best_ckpt:
        logger.info("Лучший чекпоинт: %s (WER=%.6f)", best_ckpt, best_wer)
    logger.info("Обучение завершено.")


if __name__ == "__main__":
    train()

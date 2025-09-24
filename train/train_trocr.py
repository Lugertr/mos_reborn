#!/usr/bin/env python3
"""
Главный скрипт тренировки с поддержкой mixed-precision и gradient accumulation.
"""
import os
import sys
import logging
import signal
import tempfile
import json

this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

import config  # type: ignore
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import set_seed

from utils import checkpoints, data, tokenizer_utils, model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_global_state = {"epoch": 0, "batch": 0, "model": None, "processor": None, "optimizer": None, "ckpt": None}


def _signal_save_and_exit(signum, frame):
    logger.info("Signal %s received — сохранение чекпоинта...", signum)
    try:
        epoch = int(_global_state.get("epoch", 0))
        batch = int(_global_state.get("batch", 0))
        model = _global_state.get("model", None)
        processor = _global_state.get("processor", None)
        optimizer = _global_state.get("optimizer", None)
        ckpt_obj = _global_state.get("ckpt", None)
        if model is None or processor is None or optimizer is None:
            logger.warning("Недостаточно данных для сохранения чекпоинта (model/processor/optimizer). Exit.")
        else:
            checkpoints.save_checkpoint(epoch=epoch, batch=batch, model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt_obj, reason=f"signal-{signum}", hf_save=True)
    except Exception as e:
        logger.exception("Ошибка при сохранении чекпоинта в сигнале: %s", e)
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
            except Exception as e:
                logger.warning("Не удалось включить memory growth для GPU: %s", e)
        else:
            logger.info("GPU не найден — используется CPU")

    # mixed precision
    if getattr(config, "ENABLE_MIXED_PRECISION", False):
        try:
            from tensorflow.keras import mixed_precision
            policy = "mixed_float16"
            mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled: %s", policy)
        except Exception as e:
            logger.warning("Не удалось включить mixed precision: %s", e)

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    logger.info("Устройство: %s", device)

    # Load datasets
    train_json = getattr(config, "TRAIN_JSON_PATH")
    test_json = getattr(config, "TEST_JSON_PATH")
    logger.info("TRAIN_JSON_PATH = %s", train_json)
    logger.info("TEST_JSON_PATH  = %s", test_json)
    if not os.path.exists(train_json) or not os.path.exists(test_json):
        raise FileNotFoundError("TRAIN/TEST json files not found; check config paths.")

    data_files = {"train": train_json, "test": test_json}
    try:
        hf_ds = load_dataset("json", data_files=data_files)
    except Exception as e:
        logger.info("Primary json load failed (%s) — try field='data' ...", e)
        hf_ds = load_dataset("json", data_files=data_files, field="data")

    if getattr(config, "DEBUG_TRAIN_SAMPLES", 0) > 0:
        hf_ds["train"] = hf_ds["train"].select(range(min(config.DEBUG_TRAIN_SAMPLES, len(hf_ds["train"]))))
    if getattr(config, "DEBUG_EVAL_SAMPLES", 0) > 0:
        hf_ds["test"] = hf_ds["test"].select(range(min(config.DEBUG_EVAL_SAMPLES, len(hf_ds["test"]))))

    # Processor (tokenizer + feature_extractor)
    processor = None
    if getattr(config, "TRAIN_TOKENIZER_FROM_TEST_SPLIT", False):
        texts = [ex.get("text") or ex.get("transcription") or ex.get("label") or "" for ex in hf_ds["test"]]
        texts = [t for t in texts if t]
        if not texts:
            raise RuntimeError("TRAIN_TOKENIZER_FROM_TEST_SPLIT=True but no texts in test.json")
        tmp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt")
        tmp_path = tmp_file.name
        try:
            for line in texts:
                tmp_file.write(line.replace("\n", " ") + "\n")
            tmp_file.close()
            tokenizer_obj = tokenizer_utils.train_bpe_tokenizer(corpus_paths=[tmp_path],
                                                               vocab_size=getattr(config, "TOKENIZER_VOCAB_SIZE", 8000),
                                                               special_tokens=getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]))
            processor = tokenizer_utils.save_processor_from_tokenizer(tokenizer_obj,
                                                                      out_dir=getattr(config, "TOKENIZER_OUT_DIR"),
                                                                      special_tokens=getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
                                                                      image_size=getattr(config, "TOKENIZER_IMAGE_SIZE"))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.debug("Failed to remove tmp corpus")
    else:
        processor = tokenizer_utils.load_processor_safe(getattr(config, "TOKENIZER_OUT_DIR"))
    logger.info("Processor ready.")

    # Prepare dirs & optimizer
    os.makedirs(getattr(config, "CHECKPOINTS_DIR"), exist_ok=True)
    os.makedirs(getattr(config, "LOG_DIR"), exist_ok=True)
    os.makedirs(getattr(config, "OUTPUT_DIR"), exist_ok=True)

    base_optimizer = tf.keras.optimizers.Adam(learning_rate=getattr(config, "LEARNING_RATE", 5e-5))
    # Wrap with LossScaleOptimizer if mixed precision enabled
    if getattr(config, "ENABLE_MIXED_PRECISION", False):
        try:
            from tensorflow.keras.mixed_precision import LossScaleOptimizer
            optimizer = LossScaleOptimizer(base_optimizer)
        except Exception:
            optimizer = base_optimizer
            logger.warning("Could not wrap optimizer with LossScaleOptimizer.")
    else:
        optimizer = base_optimizer

    # try restore from checkpoints
    latest_folder = checkpoints.find_best_or_latest_checkpoint(getattr(config, "CHECKPOINTS_DIR"))
    start_epoch, resume_batch = 1, 0
    model = None
    ckpt = None

    if latest_folder:
        logger.info("Found checkpoint folder %s — trying HF.from_pretrained", latest_folder)
        try:
            from transformers import TFVisionEncoderDecoderModel
            model = TFVisionEncoderDecoderModel.from_pretrained(latest_folder, local_files_only=True)
            logger.info("Model loaded via from_pretrained(%s).", latest_folder)
            try:
                processor = tokenizer_utils.load_processor_safe(latest_folder)
            except Exception:
                logger.debug("Processor load from checkpoint failed (non-fatal).")
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            restored, meta_epoch, meta_batch = checkpoints.restore_tf_checkpoint_from_folder(latest_folder, ckpt)
            if restored:
                if meta_batch and meta_batch > 0:
                    start_epoch = max(1, meta_epoch)
                    resume_batch = meta_batch
                else:
                    start_epoch = max(1, meta_epoch + 1)
                    resume_batch = 0
            else:
                start_epoch, resume_batch = 1, 0
        except Exception as e:
            logger.info("from_pretrained failed: %s — will build model from scratch", e)
            model = None

    if model is None:
        model = model_utils.build_model_scratch(processor)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if latest_folder:
            restored, meta_epoch, meta_batch = checkpoints.restore_tf_checkpoint_from_folder(latest_folder, ckpt)
            if restored:
                if meta_batch and meta_batch > 0:
                    start_epoch = max(1, meta_epoch)
                    resume_batch = meta_batch
                else:
                    start_epoch = max(1, meta_epoch + 1)
                    resume_batch = 0

    # update global state
    _global_state.update({"model": model, "processor": processor, "optimizer": optimizer, "ckpt": ckpt})

    # configure model tokens
    if model.config.pad_token_id is None:
        model.config.pad_token_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = int(getattr(processor.tokenizer, "bos_token_id", None) or 0)
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = int(getattr(processor.tokenizer, "eos_token_id", None) or 2)
    model.config.max_length = getattr(config, "GENERATION_MAX_LENGTH", 128)

    # Prepare datasets (fast pipeline)
    train_images_dir = getattr(config, "TRAIN_IMAGES_DIR")
    test_images_dir = getattr(config, "TEST_IMAGES_DIR")
    train_ds = data.prepare_tf_dataset(hf_ds["train"], processor, train_images_dir,
                                       batch_size=getattr(config, "PER_DEVICE_TRAIN_BATCH_SIZE", 1), shuffle=True)
    eval_ds = data.prepare_tf_dataset(hf_ds["test"], processor, test_images_dir,
                                      batch_size=getattr(config, "PER_DEVICE_EVAL_BATCH_SIZE", 1), shuffle=False)

    tb_writer = tf.summary.create_file_writer(getattr(config, "LOG_DIR"))
    best_wer, best_ckpt = float("inf"), None
    grad_clip = getattr(config, "GRAD_CLIP_NORM", None)
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    total_epochs = getattr(config, "NUM_TRAIN_EPOCHS", 3)

    accumulation_steps = max(1, getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1))
    logger.info("Training from epoch %d/%d, resume_batch=%d, grad_accum_steps=%d", start_epoch, total_epochs, resume_batch, accumulation_steps)

    save_every_steps = getattr(config, "SAVE_CHECKPOINT_EVERY_STEPS", 0)
    _global_state["epoch"] = start_epoch
    _global_state["batch"] = resume_batch

    # prepare accumulator variables for gradient accumulation
    accum_grads = [tf.zeros_like(v) for v in model.trainable_variables]

    for epoch in range(start_epoch, total_epochs + 1):
        _global_state["epoch"] = epoch
        logger.info("=== Epoch %d/%d ===", epoch, total_epochs)
        train_loss_metric.reset_states()
        steps = 0

        ds_iter = train_ds
        if epoch == start_epoch and resume_batch and resume_batch > 0:
            logger.info("Resuming epoch %d: skipping %d batches", epoch, resume_batch)
            ds_iter = train_ds.skip(resume_batch)
            resume_batch = 0

        try:
            for pv, lb in ds_iter:
                steps += 1
                _global_state["batch"] = steps

                with tf.GradientTape() as tape:
                    pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
                    labels_masked = tf.where(tf.equal(lb, pad_id), tf.constant(-100, dtype=lb.dtype), lb)
                    outputs = model(pixel_values=pv, labels=labels_masked, training=True)
                    loss = outputs.loss
                    # if mixed precision LossScaleOptimizer is used, scale the loss
                    if getattr(config, "ENABLE_MIXED_PRECISION", False):
                        try:
                            loss = optimizer.get_scaled_loss(loss)  # type: ignore
                        except Exception:
                            pass

                grads = tape.gradient(loss, model.trainable_variables)
                # if mixed precision, unscale grads
                if getattr(config, "ENABLE_MIXED_PRECISION", False):
                    try:
                        grads = optimizer.get_unscaled_gradients(grads)  # type: ignore
                    except Exception:
                        pass

                # clip grads
                if grad_clip is not None:
                    grads = [None if g is None else tf.clip_by_norm(g, float(grad_clip)) for g in grads]

                # accumulate grads
                for i, g in enumerate(grads):
                    if g is None:
                        continue
                    accum_grads[i] = accum_grads[i] + g

                apply_now = (steps % accumulation_steps == 0)
                if apply_now:
                    # prepare grads_and_vars (skip None)
                    grads_and_vars = [(accum_grads[i] / float(accumulation_steps), v)
                                      for i, v in enumerate(model.trainable_variables) if accum_grads[i] is not None]
                    if grads_and_vars:
                        optimizer.apply_gradients(grads_and_vars)
                    # reset accumulators
                    accum_grads = [tf.zeros_like(v) for v in model.trainable_variables]

                train_loss_metric.update_state(loss if not getattr(config, "ENABLE_MIXED_PRECISION", False) else (loss if not hasattr(loss, "numpy") else loss))

                if steps % getattr(config, "LOGGING_STEPS", 20) == 0:
                    avg = float(train_loss_metric.result().numpy())
                    logger.info("Epoch %d step %d avg loss %.6f", epoch, steps, avg)

                if save_every_steps and (steps % save_every_steps == 0):
                    logger.info("Periodic TF-only save at epoch %d step %d", epoch, steps)
                    checkpoints.save_checkpoint(epoch=epoch, batch=steps, model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt, reason="periodic", hf_save=False)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — saving and exiting")
            checkpoints.save_checkpoint(epoch=epoch, batch=steps, model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt, reason="keyboard-interrupt", hf_save=True)
            raise

        avg_loss = float(train_loss_metric.result().numpy())
        logger.info("Epoch %d done, avg loss %.6f", epoch, avg_loss)

        metrics = model_utils.evaluate_and_log(model, processor, eval_ds, tb_writer, epoch, tb_examples=getattr(config, "TB_EXAMPLES_TO_LOG", 5))
        logger.info("Eval after epoch %d: %s", epoch, metrics)

        try:
            saved_ckpt_dir = checkpoints.save_checkpoint(epoch=epoch, batch=0, model=model, processor=processor, optimizer=optimizer, ckpt_obj=ckpt, reason="epoch-end", hf_save=True)
        except Exception as e:
            logger.warning("Saving checkpoint at epoch end failed: %s", e)
            saved_ckpt_dir = None

        wer = metrics.get("wer", None)
        if wer is not None and wer < best_wer:
            best_wer = wer
            best_ckpt = saved_ckpt_dir or getattr(config, "CHECKPOINTS_DIR")
            best_dir = os.path.join(getattr(config, "CHECKPOINTS_DIR"), getattr(config, "BEST_CHECKPOINT_NAME", "best"))
            os.makedirs(best_dir, exist_ok=True)
            try:
                model.save_pretrained(best_dir)
            except Exception:
                logger.debug("model.save_pretrained(best) failed (non-fatal).")
            try:
                processor.save_pretrained(best_dir)
            except Exception:
                if hasattr(processor, "tokenizer"):
                    try:
                        processor.tokenizer.save_pretrained(best_dir)
                    except Exception:
                        logger.debug("Не удалось сохранить tokenizer в best чекпоинт.")
            try:
                ckpt_best = tf.train.Checkpoint(optimizer=optimizer, model=model)
                tf_ckpt_best = ckpt_best.save(os.path.join(best_dir, "tf_ckpt"))
            except Exception as e:
                tf_ckpt_best = None
                logger.warning("TF save for best checkpoint failed: %s", e)
            meta_best = {"epoch": epoch, "batch": 0, "tf_checkpoint": tf_ckpt_best}
            try:
                with open(os.path.join(best_dir, "metadata.json"), "w", encoding="utf-8") as mf:
                    json.dump(meta_best, mf, ensure_ascii=False)
            except Exception as e:
                logger.warning("Не удалось записать metadata.json для best: %s", e)
            logger.info("Saved best model (WER=%.6f) to %s", best_wer, best_dir)

    logger.info("Saving final model to %s", getattr(config, "OUTPUT_DIR"))
    model_utils.save_final_model_and_processor(model, processor, getattr(config, "OUTPUT_DIR"))
    if best_ckpt:
        logger.info("Best checkpoint: %s (WER=%.6f)", best_ckpt, best_wer)
    logger.info("Training finished.")


if __name__ == "__main__":
    train()

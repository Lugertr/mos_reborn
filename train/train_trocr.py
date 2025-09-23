# train_trocr.py
"""
Основной код обучения модели TrOCR с нуля, используя processor, созданный через build_tokenizer.py.
Без PyTorch.
"""

import os
import sys
import logging

from typing import Tuple, List

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, TrOCRProcessor, set_seed

import config
from train_utils import (
    transform_example,
    batched_greedy_decode_tf,
    compute_metrics_from_processor,
    save_model_and_processor,
    get_device,
    find_latest_checkpoint
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_tf_dataset(split_ds, processor: TrOCRProcessor, images_dir: str, batch_size: int, shuffle: bool):
    """
    Создать tf.data.Dataset из split_ds с transform_example, паддингом меток.
    """
    def gen():
        for ex in split_ds:
            out = transform_example(ex, images_dir, processor)
            pv = out["pixel_values"].astype(np.float32)
            lab = np.array(out["labels"], dtype=np.int32)
            yield pv, lab

    iterator = gen()
    try:
        pv0, _ = next(iterator)
    except StopIteration:
        raise RuntimeError("prepare_tf_dataset: раздел датасета пуст.")
    pv_shape = pv0.shape
    output_signature = (
        tf.TensorSpec(shape=pv_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    pad_id = getattr(processor.tokenizer, "pad_token_id", 0)
    ds = ds.padded_batch(batch_size, padded_shapes=(pv_shape, [None]), padding_values=(0.0, pad_id))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model_scratch(processor: TrOCRProcessor) -> VisionEncoderDecoderModel:
    """
    Построить модель VisionEncoderDecoder с нуля, на основе конфигурации из processor.tokenizer/vocab_size.
    """
    from transformers import ViTConfig, BertConfig, VisionEncoderDecoderConfig

    # Конфигурация энкодера
    enc_cfg = ViTConfig()  # можно настроить size, patch_size и др.

    # Конфигурация декодера
    vocab_size = getattr(processor.tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(processor.tokenizer)
    dec_cfg = BertConfig(vocab_size=vocab_size, is_decoder=True, add_cross_attention=True)

    ved_cfg = VisionEncoderDecoderConfig(encoder=enc_cfg, decoder=dec_cfg)
    ved_cfg.decoder_start_token_id = getattr(processor.tokenizer, "bos_token_id", None) or 0
    ved_cfg.pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    ved_cfg.vocab_size = vocab_size

    model = VisionEncoderDecoderModel(ved_cfg)
    return model


def evaluate_and_log(model, processor: TrOCRProcessor, eval_ds, tb_writer, epoch: int, tb_examples: int = 5) -> dict:
    """
    Оценка модели и логирование в TensorBoard: WER и примеры (REF vs PRED).
    """
    all_preds = []
    all_labels = []
    sample_pairs: List[Tuple[str, str]] = []

    for pv_batch, labels_batch in eval_ds:
        try:
            gen = model.generate(pv_batch, max_length=config.GENERATION_MAX_LENGTH, num_beams=1)
            if isinstance(gen, tf.Tensor):
                pred_ids = gen.numpy().tolist()
            else:
                pred_ids = gen.tolist()
        except Exception:
            pred_ids = batched_greedy_decode_tf(model, processor, pv_batch, max_length=config.GENERATION_MAX_LENGTH)

        lbl_np = labels_batch.numpy().tolist()
        pad_tok = getattr(processor.tokenizer, "pad_token_id", 0)
        lbl_masked = [[(int(x) if int(x) != pad_tok else -100) for x in seq] for seq in lbl_np]

        all_preds.extend(pred_ids)
        all_labels.extend(lbl_masked)

        if len(sample_pairs) < tb_examples:
            take = min(tb_examples - len(sample_pairs), len(pred_ids))
            preds_txt = processor.batch_decode(pred_ids[:take], skip_special_tokens=True)
            labs_clean = [[l for l in seq if l != -100] for seq in lbl_masked[:take]]
            refs_txt = processor.batch_decode(labs_clean, skip_special_tokens=True)
            for r, p in zip(refs_txt, preds_txt):
                sample_pairs.append((r, p))

    metrics = compute_metrics_from_processor((all_preds, all_labels), processor)

    with tb_writer.as_default():
        tf.summary.scalar("eval/wer", metrics.get("wer", 0.0), step=epoch)
        if sample_pairs:
            lines = []
            for r, p in sample_pairs:
                lines.append(f"REF: {r}\nPRED: {p}")
            tf.summary.text("eval/examples", tf.convert_to_tensor(lines), step=epoch)
    tb_writer.flush()
    return metrics


def train():
    """
    Основная функция тренировки.
    """
    set_seed(config.SEED)
    device = get_device()
    logger.info(f"Устройство: {device}")

    # Загружаем свой processor, созданный build_tokenizer.py
    try:
        processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME_OR_PATH)
    except Exception as e:
        logger.error(f"Не удалось загрузить processor из {config.MODEL_NAME_OR_PATH}: {e}")
        sys.exit(1)
    logger.info("Processor загружен.")

    # Инициализация модели с нуля
    model = build_model_scratch(processor)
    logger.info("Модель инициализирована с нуля.")

    # Проверка наличия чекпоинта для восстановления
    ckpt = find_latest_checkpoint(config.CHECKPOINTS_DIR)
    if ckpt:
        try:
            logger.info(f"Восстановление из чекпоинта {ckpt}")
            model = VisionEncoderDecoderModel.from_pretrained(ckpt, from_tf=True)
            processor = TrOCRProcessor.from_pretrained(ckpt)
            logger.info("Восстановление прошло успешно.")
        except Exception as e:
            logger.warning(f"Не удалось восстановить из чекпоинта: {e}")

    # Установка специальных токенов модели
    if model.config.pad_token_id is None:
        model.config.pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = getattr(processor.tokenizer, "bos_token_id", None) or 0
    model.config.eos_token_id = getattr(processor.tokenizer, "eos_token_id", None) or 0
    model.config.max_length = config.GENERATION_MAX_LENGTH

    # Загрузка датасета
    data_files = {
        "train": os.path.join(config.DATA_DIR, "train.json"),
        "test": os.path.join(config.DATA_DIR, "test.json")
    }
    hf_ds = load_dataset("json", data_files=data_files)

    if config.DEBUG_TRAIN_SAMPLES and config.DEBUG_TRAIN_SAMPLES > 0:
        hf_ds["train"] = hf_ds["train"].select(range(min(config.DEBUG_TRAIN_SAMPLES, len(hf_ds["train"]))))
    if config.DEBUG_EVAL_SAMPLES and config.DEBUG_EVAL_SAMPLES > 0:
        hf_ds["test"] = hf_ds["test"].select(range(min(config.DEBUG_EVAL_SAMPLES, len(hf_ds["test"]])))

    images_dir = os.path.join(config.DATA_DIR, config.IMAGES_SUBDIR)

    train_ds = prepare_tf_dataset(hf_ds["train"], processor, images_dir, batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=True)
    eval_ds = prepare_tf_dataset(hf_ds["test"], processor, images_dir, batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    tb_writer = tf.summary.create_file_writer(config.LOG_DIR)

    best_wer = float("inf")
    best_ckpt = None

    for epoch in range(1, config.NUM_TRAIN_EPOCHS + 1):
        logger.info(f"=== Эпоха {epoch}/{config.NUM_TRAIN_EPOCHS} ===")
        total_loss = 0.0
        steps = 0

        for pv, lb in train_ds:
            steps += 1
            with tf.GradientTape() as tape:
                outputs = model(pixel_values=pv, labels=lb, training=True)
                loss = outputs.loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_val = float(loss.numpy())
            total_loss += loss_val

            if steps % config.LOGGING_STEPS == 0:
                logger.info(f"Epoch {epoch} step {steps} avg loss {total_loss/steps:.6f}")

        avg_loss = total_loss / max(1, steps)
        logger.info(f"Эпоха {epoch} завершена, средний loss {avg_loss:.6f}")

        metrics = evaluate_and_log(model, processor, eval_ds, tb_writer, epoch, tb_examples=config.TB_EXAMPLES_TO_LOG)
        logger.info(f"Eval после эпохи {epoch}: {metrics}")

        # Сохранение чекпоинта
        ckpt_name = f"checkpoint-epoch{epoch}"
        ck_dir = os.path.join(config.CHECKPOINTS_DIR, ckpt_name)
        os.makedirs(ck_dir, exist_ok=True)
        model.save_pretrained(ck_dir)
        processor.save_pretrained(ck_dir)
        logger.info(f"Чекпоинт сохранён: {ck_dir}")

        if metrics.get("wer", None) is not None and metrics["wer"] < best_wer:
            best_wer = metrics["wer"]
            best_ckpt = ck_dir
            best_dir = os.path.join(config.CHECKPOINTS_DIR, "best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            logger.info(f"Лучшая модель сохранена (WER={best_wer:.6f}): {best_dir}")

    # Финальное сохранение
    model_save = config.OUTPUT_DIR
    save_model_and_processor(model, processor, model_save)
    logger.info(f"Обучение завершено. Итоговая модель сохранена: {model_save}")
    if best_ckpt:
        logger.info(f"Лучший чекпоинт: {best_ckpt} (WER={best_wer:.6f})")


if __name__ == "__main__":
    train()

#!/usr/bin/env python3
# train_trocr.py
"""
Основной скрипт тренировки TrOCR (TensorFlow).
Включает автономное создание char-level токенайзера, если в TOKENIZER_OUT_DIR
нет валидного процессора/токенайзера.
"""

import os
import sys
import logging
from typing import Tuple, List, Optional, Dict, Any

this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
import config  # type: ignore

import numpy as np
import tensorflow as tf
from datasets import load_dataset

from transformers import (
    VisionEncoderDecoderConfig,
    TFVisionEncoderDecoderModel,
    ViTConfig,
    BertConfig,
    TrOCRProcessor,
    set_seed,
    PreTrainedTokenizerFast,
    ViTFeatureExtractor,
)

# Импорт tokenizers для построения fallback токенайзера
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing

from train_utils import (
    transform_example,
    batched_greedy_decode_tf,
    compute_metrics_from_processor,
    save_model_and_processor,
    get_device,
    find_latest_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------
# CHAR-LEVEL tokenizer builder (standalone fallback)
# --------------------------
def make_additional_chars() -> List[str]:
    modern_upper = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    modern_chars = []
    for ch in modern_upper:
        modern_chars.append(ch)
        modern_chars.append(ch.lower())

    archaic = [
        "І", "Ѣ", "Ѳ", "Ѵ",
        "Ѡ", "ѡ", "Ѧ", "ѧ", "Ѩ", "ѩ", "Ѫ", "ѫ", "Ѭ", "ѭ",
        "Ѯ", "ѯ", "Ѱ", "ѱ", "Ѿ", "ѿ", "Ѹ", "ѹ", "Ѻ", "ѻ",
        "Ꙋ", "ꙋ", "Ѷ", "ѷ", "Ѽ", "ѽ", "Ҁ", "ҁ"
    ]
    archaic_chars = []
    for ch in archaic:
        archaic_chars.append(ch)
        archaic_chars.append(ch.lower())

    digits = [str(d) for d in range(10)]
    roman_upper = list("IVXLCDM")
    roman_lower = [c.lower() for c in roman_upper]

    punctuation = list(".,:;!?\"'()[]{}-–—«»/\\@#%&*№•–—")

    all_chars: List[str] = []
    for grp in (modern_chars, archaic_chars, digits, roman_upper, roman_lower, punctuation):
        for ch in grp:
            if ch not in all_chars:
                all_chars.append(ch)
    if " " not in all_chars:
        all_chars.append(" ")
    return all_chars


def build_char_level_tokenizer_and_processor(out_dir: str):
    """
    Создаёт char-level tokenizer (tokenizers lib), сохраняет tokenizer.json,
    затем создаёт PreTrainedTokenizerFast и ViTFeatureExtractor и TrOCRProcessor,
    сохраняет их в out_dir и возвращает процессор.
    """
    os.makedirs(out_dir, exist_ok=True)
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # Build vocab dict
    chars = make_additional_chars()
    vocab: Dict[str, int] = {}
    idx = 0
    for t in special_tokens:
        vocab[t] = idx
        idx += 1
    for ch in chars:
        if ch in vocab:
            continue
        vocab[ch] = idx
        idx += 1

    # Create WordLevel model
    model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    tokenizer_obj = Tokenizer(model)
    tokenizer_obj.normalizer = normalizers.Sequence([normalizers.NFC()])

    # Try splitting into single chars; fallback to ByteLevel
    try:
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Split(pattern=r"", behavior="isolated")
    except Exception:
        tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel()

    tokenizer_obj.decoder = decoders.WordPiece()

    # Post-processor: add BOS/EOS
    tokenizer_obj.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])]
    )

    # Save tokenizer.json
    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer_obj.save(tok_json_path)
    logger.info("Сохранён fallback tokenizer.json: %s", tok_json_path)

    # Create PreTrainedTokenizerFast
    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json_path,
                                       unk_token="[UNK]", pad_token="[PAD]",
                                       bos_token="[BOS]", eos_token="[EOS]")
    try:
        tok_fast.save_pretrained(out_dir)
    except Exception:
        logger.debug("tok_fast.save_pretrained failed (non-fatal)")

    # Create feature extractor
    feat = ViTFeatureExtractor(do_resize=True,
                               size=getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384)),
                               do_normalize=True)
    try:
        feat.save_pretrained(out_dir)
    except Exception:
        logger.debug("feat.save_pretrained failed (non-fatal)")

    # Build processor
    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(out_dir)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal)")

    logger.info("Собран fallback TrOCRProcessor в %s", out_dir)
    return processor


# --------------------------
# Safe processor loading (with internal fallback)
# --------------------------
def load_processor_safe(path: str) -> TrOCRProcessor:
    """
    Попытки загрузить TrOCRProcessor локально. Если нет токенайзера/processor,
    создаём char-level tokenizer и processor локально.
    """
    # 1) попытка стандартной локальной загрузки
    try:
        proc = TrOCRProcessor.from_pretrained(path, local_files_only=True)
        logger.info("TrOCRProcessor загружен из '%s' через from_pretrained(local_files_only=True).", path)
        return proc
    except Exception as e:
        logger.warning("TrOCRProcessor.from_pretrained(local_files_only=True) failed: %s", e)

    # 2) попытка загрузить токенайзер
    tok_fast: Optional[PreTrainedTokenizerFast] = None
    tok_json = os.path.join(path, "tokenizer.json")

    try:
        tok_fast = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
        logger.info("PreTrainedTokenizerFast загружен из '%s' через from_pretrained(local_files_only=True).", path)
    except Exception as e:
        logger.debug("PreTrainedTokenizerFast.from_pretrained failed: %s", e)
        if os.path.exists(tok_json):
            try:
                tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json,
                                                   unk_token="[UNK]", pad_token="[PAD]",
                                                   bos_token="[BOS]", eos_token="[EOS]")
                logger.info("PreTrainedTokenizerFast создан из tokenizer.json: %s", tok_json)
            except Exception as e2:
                logger.warning("PreTrainedTokenizerFast(tokenizer_file=...) failed: %s", e2)
        else:
            logger.info("tokenizer.json не найден в %s", path)

    # 3) load/create feature extractor
    feat: ViTFeatureExtractor
    try:
        feat = ViTFeatureExtractor.from_pretrained(path, local_files_only=True)
        logger.info("ViTFeatureExtractor загружен из %s", path)
    except Exception as e:
        logger.info("ViTFeatureExtractor.from_pretrained не сработал: %s — создаём вручную.", e)
        feat = ViTFeatureExtractor(do_resize=True,
                                   size=getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384)),
                                   do_normalize=True)
        try:
            feat.save_pretrained(path)
            logger.info("ViTFeatureExtractor конфиг сохранён в %s", path)
        except Exception:
            logger.debug("Не удалось сохранить ViTFeatureExtractor в %s", path)

    # 4) если токенайзер не найден — fallback
    if tok_fast is None:
        logger.warning("Токенайзер не найден в %s — создаём CHAR-LEVEL токенайзер (fallback).", path)
        try:
            processor = build_char_level_tokenizer_and_processor(path)
            return processor
        except Exception as e:
            logger.error("Автоматическое создание токенайзера не удалось: %s", e)
            raise RuntimeError(f"Не удалось загрузить или создать токенайзер в '{path}'. Причина: {e}")

    # assemble processor
    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(path)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal)")
    logger.info("TrOCRProcessor собран из токенайзера и feature_extractor.")
    return processor


# --------------------------
# Dataset & model utilities
# --------------------------
def prepare_tf_dataset(split_ds, processor: TrOCRProcessor, images_dir: str, batch_size: int, shuffle: bool):
    def gen():
        for ex in split_ds:
            try:
                out = transform_example(ex, images_dir, processor)
            except Exception as e:
                logger.warning("transform_example failed for %s: %s", ex.get("file_name") or ex.get("image"), e)
                continue
            pv = out["pixel_values"].astype(np.float32)
            lab = np.array(out["labels"], dtype=np.int32)
            yield pv, lab

    it = gen()
    try:
        pv0, _ = next(it)
    except StopIteration:
        raise RuntimeError("prepare_tf_dataset: раздел датасета пуст.")
    pv_shape = pv0.shape

    output_signature = (
        tf.TensorSpec(shape=pv_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    ds = ds.padded_batch(batch_size, padded_shapes=(pv_shape, [None]), padding_values=(0.0, pad_id))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model_scratch(processor: TrOCRProcessor) -> TFVisionEncoderDecoderModel:
    image_size = getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))[0]

    enc_cfg = ViTConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        image_size=image_size,
        add_pooling_layer=False,  # отключаем pooler
    )

    try:
        vocab_size = getattr(processor.tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(processor.tokenizer.get_vocab())
    except Exception:
        vocab_size = 30522

    dec_cfg = BertConfig(
        vocab_size=vocab_size,
        is_decoder=True,
        add_cross_attention=True,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
    )

    ved_cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)

    ved_cfg.pad_token_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    ved_cfg.decoder_start_token_id = int(getattr(processor.tokenizer, "bos_token_id", None) or getattr(processor.tokenizer, "cls_token_id", 0) or 0)
    ved_cfg.eos_token_id = int(getattr(processor.tokenizer, "eos_token_id", None) or getattr(processor.tokenizer, "sep_token_id", 2) or 2)
    ved_cfg.vocab_size = vocab_size
    ved_cfg.max_length = getattr(config, "GENERATION_MAX_LENGTH", 128)

    model = TFVisionEncoderDecoderModel(ved_cfg)
    logger.info("Создан TF VisionEncoderDecoderModel (vocab_size=%d)", vocab_size)
    return model


def evaluate_and_log(model, processor, eval_ds, tb_writer, epoch: int, tb_examples: int = 5):
    all_preds: List[List[int]] = []
    all_labels: List[List[int]] = []
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
        pad_tok = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
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
            lines = [f"REF: {r}\nPRED: {p}" for r, p in sample_pairs]
            tf.summary.text("eval/examples", tf.convert_to_tensor(lines), step=epoch)
    tb_writer.flush()
    return metrics


def train():
    set_seed(getattr(config, "SEED", 42))

    # GPU configuration: включаем memory growth при наличии GPU
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

    device = get_device()
    logger.info("Устройство: %s", device)

    # Загрузка или создание processor
    processor = load_processor_safe(config.TOKENIZER_OUT_DIR)
    logger.info("Processor готов.")

    # Создание модели
    model = build_model_scratch(processor)
    logger.info("Модель инициализирована с нуля (TF).")

    # Отключаем pooler слоя ViT, если существует
    try:
        vit_encoder = model.encoder.vit
        if hasattr(vit_encoder, "pooler"):
            vit_encoder.pooler.trainable = False
            logger.info("Pooler слоя ViT размечен как не обучаемый (trainable=False)")
    except Exception as e:
        logger.debug("Не удалось найти или отключить pooler: %s", e)

    # Восстановление из чекпоинта, если есть
    latest = find_latest_checkpoint(config.CHECKPOINTS_DIR)
    if latest:
        try:
            logger.info("Попытка восстановления из чекпоинта %s (local)...", latest)
            model = TFVisionEncoderDecoderModel.from_pretrained(latest, local_files_only=True)
            # Постараемся восстановить processor тоже
            try:
                processor = load_processor_safe(latest)
            except Exception:
                logger.warning("Не удалось загрузить processor из чекпоинта %s — оставляем прежний", latest)
            logger.info("Восстановление из чекпоинта прошло успешно.")
        except Exception as e:
            logger.warning("Не удалось восстановить модель из чекпоинта: %s", e)

    # Убедимся, что спецтокены выставлены
    if model.config.pad_token_id is None:
        model.config.pad_token_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = int(getattr(processor.tokenizer, "bos_token_id", None) or 0)
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = int(getattr(processor.tokenizer, "eos_token_id", None) or 2)
    model.config.max_length = getattr(config, "GENERATION_MAX_LENGTH", 128)

    # Загрузка dataset
    data_files = {
        "train": os.path.join(config.DATA_DIR, "train.json"),
        "test": os.path.join(config.DATA_DIR, "test.json"),
    }

    try:
        hf_ds = load_dataset("json", data_files=data_files)
    except Exception:
        logger.info("Первичная загрузка json не удалась — пробуем с field='data' ...")
        hf_ds = load_dataset("json", data_files=data_files, field="data")

    if getattr(config, "DEBUG_TRAIN_SAMPLES", 0) > 0:
        hf_ds["train"] = hf_ds["train"].select(range(min(config.DEBUG_TRAIN_SAMPLES, len(hf_ds["train"]))))

    if getattr(config, "DEBUG_EVAL_SAMPLES", 0) > 0:
        hf_ds["test"] = hf_ds["test"].select(range(min(config.DEBUG_EVAL_SAMPLES, len(hf_ds["test"]))))

    images_dir = os.path.join(config.DATA_DIR, config.IMAGES_SUBDIR)

    train_ds = prepare_tf_dataset(hf_ds["train"], processor, images_dir,
                                  batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=True)
    eval_ds = prepare_tf_dataset(hf_ds["test"], processor, images_dir,
                                 batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False)

    # Создаём нужные директории
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    tb_writer = tf.summary.create_file_writer(config.LOG_DIR)

    best_wer: float = float("inf")
    best_ckpt: Optional[str] = None

    for epoch in range(1, config.NUM_TRAIN_EPOCHS + 1):
        logger.info("=== Эпоха %d/%d ===", epoch, config.NUM_TRAIN_EPOCHS)
        total_loss = 0.0
        steps = 0

        for pv, lb in train_ds:
            steps += 1
            with tf.GradientTape() as tape:
                pad_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
                labels_masked = tf.where(tf.equal(lb, pad_id),
                                         tf.constant(-100, dtype=lb.dtype),
                                         lb)
                outputs = model(pixel_values=pv, labels=labels_masked, training=True)
                loss = outputs.loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_val = float(loss.numpy())
            total_loss += loss_val

            if steps % config.LOGGING_STEPS == 0:
                logger.info("Epoch %d step %d avg loss %.6f", epoch, steps, total_loss / steps)

        avg_loss = total_loss / max(1, steps)
        logger.info("Эпоха %d завершена, средний loss %.6f", epoch, avg_loss)

        metrics = evaluate_and_log(model, processor, eval_ds, tb_writer, epoch, tb_examples=config.TB_EXAMPLES_TO_LOG)
        logger.info("Eval после эпохи %d: %s", epoch, metrics)

        ckpt_name = f"checkpoint-{epoch}"
        ckpt_dir = os.path.join(config.CHECKPOINTS_DIR, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            model.save_pretrained(ckpt_dir)
            try:
                processor.save_pretrained(ckpt_dir)
            except Exception:
                if hasattr(processor, "tokenizer"):
                    try:
                        processor.tokenizer.save_pretrained(ckpt_dir)
                    except Exception:
                        logger.debug("Не удалось сохранить tokenizer в чекпоинт.")
            logger.info("Чекпоинт сохранён: %s", ckpt_dir)
        except Exception as e:
            logger.warning("Сохранение чекпоинта не удалось: %s", e)

        wer = metrics.get("wer", None)
        if wer is not None and wer < best_wer:
            best_wer = wer
            best_ckpt = ckpt_dir
            best_dir = os.path.join(config.CHECKPOINTS_DIR, "best")
            try:
                model.save_pretrained(best_dir)
                try:
                    processor.save_pretrained(best_dir)
                except Exception:
                    if hasattr(processor, "tokenizer"):
                        try:
                            processor.tokenizer.save_pretrained(best_dir)
                        except Exception:
                            logger.debug("Не удалось сохранить tokenizer в best чекпоинт.")
                logger.info("Сохранена лучшая модель (WER=%.6f) в %s", best_wer, best_dir)
            except Exception as e:
                logger.warning("Не удалось сохранить лучшую модель: %s", e)

    logger.info("Сохраняем финальную модель в %s", config.OUTPUT_DIR)
    save_model_and_processor(model, processor, config.OUTPUT_DIR)
    if best_ckpt:
        logger.info("Лучший чекпоинт: %s (WER=%.6f)", best_ckpt, best_wer)
    logger.info("Обучение завершено.")


if __name__ == "__main__":
    train()

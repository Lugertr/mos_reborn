"""
model_utils.py — сборка модели, fallback decode, evaluate & final save.
"""
import logging
from typing import List, Tuple, Dict

import tensorflow as tf
from transformers import (
    VisionEncoderDecoderConfig,
    TFVisionEncoderDecoderModel,
    ViTConfig,
    BertConfig,
)

import config  # type: ignore
from .metrics import compute_metrics_from_processor

logger = logging.getLogger(__name__)


def build_model_scratch(processor) -> TFVisionEncoderDecoderModel:
    image_size = getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))[0]

    enc_cfg = ViTConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        image_size=image_size,
        add_pooling_layer=False,
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


def batched_greedy_decode_tf(model, processor, pixel_values: tf.Tensor, max_length: int) -> List[List[int]]:
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
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Model returned no logits for decoder_input_ids call.")
        except Exception:
            outputs = model(pixel_values=pixel_values, training=False)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Model did not return logits in generation fallback.")
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


def evaluate_and_log(model, processor, eval_ds, tb_writer, epoch: int, tb_examples: int = 5):
    all_preds = []
    all_labels = []
    sample_pairs = []

    for pv_batch, labels_batch in eval_ds:
        try:
            gen = model.generate(pv_batch, max_length=getattr(config, "GENERATION_MAX_LENGTH", 128), num_beams=1)
            pred_ids = gen.numpy().tolist() if isinstance(gen, tf.Tensor) else gen.tolist()
        except Exception:
            pred_ids = batched_greedy_decode_tf(model, processor, pv_batch, max_length=getattr(config, "GENERATION_MAX_LENGTH", 128))

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


def save_final_model_and_processor(model, processor, out_dir: str):
    import os
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("save_final_model_and_processor: model.save_pretrained failed: %s", e)
        try:
            model.save(out_dir, include_optimizer=False)
        except Exception as e2:
            logger.error("save_final_model_and_processor: fallback save failed: %s", e2)
    try:
        processor.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("save_final_model_and_processor: processor.save_pretrained failed: %s", e)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
import json
import math
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_USE_XLA", "0")
os.environ.setdefault("TF_ENABLE_LAYOUT_OPTIMIZER", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

logger = logging.getLogger("trocr")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def setup_hardware():
    try:
        tf.config.optimizer.set_jit(False)
        tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    except Exception as e:
        logger.warning(f"Could not disable XLA JIT / layout optimizer: {e}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            lg = tf.config.list_logical_devices('GPU')
            logger.info(f"GPU detected: {len(gpus)} physical, {len(lg)} logical; memory growth enabled.")
        except Exception as e:
            logger.warning(f"Could not set memory growth: {e}")
    else:
        logger.warning("No GPU detected. Falling back to CPU. Training will be slower.")


REQUIRED_CHARS = (
    " "
    ".,;:!?\'\"-–—()[]{}«»/@#&%+*=<>_"
    "0123456789"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "ѢѣѲѳІіѴѵ"
    "IVXLCDMivxlcdm"
)

SPECIAL_TOKENS = {
    "PAD": "<pad>",
    "SOS": "<s>",
    "EOS": "</s>",
    "UNK": "<unk>",
}


class Charset:
    def __init__(self, extra_texts: List[str] | None = None):
        base_chars = list(dict.fromkeys(REQUIRED_CHARS))
        extra = []
        if extra_texts:
            seen = set(base_chars)
            for t in extra_texts:
                for ch in t:
                    if ch not in seen:
                        extra.append(ch)
                        seen.add(ch)
        self.tokens = [
            SPECIAL_TOKENS["PAD"],
            SPECIAL_TOKENS["SOS"],
            SPECIAL_TOKENS["EOS"],
            SPECIAL_TOKENS["UNK"],
        ] + base_chars + extra
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.char_to_id = {ch: i for i, ch in enumerate(self.tokens) if len(ch) == 1}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["PAD"]]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["SOS"]]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["EOS"]]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["UNK"]]

    def encode(self, text: str, add_sos_eos: bool = False) -> List[int]:
        ids = [self.char_to_id.get(ch, self.unk_id) for ch in text]
        if add_sos_eos:
            return [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        out = []
        for i in ids:
            if i == self.pad_id:
                if strip_special:
                    continue
            tok = self.id_to_token.get(int(i), SPECIAL_TOKENS["UNK"])
            if strip_special and tok in SPECIAL_TOKENS.values():
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Charset":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        c = Charset(extra_texts=None)
        c.tokens = obj["tokens"]
        c.token_to_id = {tok: i for i, tok in enumerate(c.tokens)}
        c.id_to_token = {i: tok for tok, i in c.token_to_id.items()}
        c.char_to_id = {ch: i for i, ch in enumerate(c.tokens) if len(ch) == 1}
        return c


def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[n][m]


def cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return float(len(hyp_chars) > 0)
    dist = _levenshtein(ref_chars, hyp_chars)
    return dist / max(1, len(ref_chars))


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return float(len(hyp_words) > 0)
    dist = _levenshtein(ref_words, hyp_words)
    return dist / max(1, len(ref_words))


@dataclass
class Sample:
    path: str
    text: str


def read_json_list(json_path: str, images_dir: str) -> List[Sample]:
    with open(json_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    samples: List[Sample] = []
    missing = 0
    empty = 0
    for obj in arr:
        fn = obj.get("file_name")
        text = obj.get("text", "")
        if not fn:
            continue
        path = os.path.join(images_dir, fn)
        if not os.path.exists(path):
            logger.warning(f"Файл отсутствует: {path}")
            missing += 1
            continue
        if text is None or len(str(text)) == 0:
            logger.warning(f"Пустая целевая строка для файла: {path}")
            empty += 1
            continue
        samples.append(Sample(path=path, text=str(text)))
    logger.info(f"Загружено {len(samples)} образцов из {json_path}. Пропущено (нет файла/пустой текст): {missing}/{empty}")
    return samples


def preprocess_image(img, img_h: int, img_w: int):
    img = tf.image.convert_image_dtype(img, tf.float32)
    if img.shape.rank == 3 and img.shape[-1] != 1:
        if tf.shape(img)[-1] > 1:
            img = img[..., :3]
        img = tf.image.rgb_to_grayscale(img)
    elif img.shape.rank == 2:
        img = tf.expand_dims(img, -1)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.cast(img_h, tf.float32) / tf.cast(h, tf.float32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [img_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    if new_w < img_w:
        pad_w = img_w - new_w
        pad = tf.fill([img_h, pad_w, 1], 1.0)
        img = tf.concat([img, pad], axis=1)
    else:
        img = tf.image.crop_to_bounding_box(img, 0, 0, img_h, img_w)
    return img


def augment_image(img):
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_brightness(img, 0.1)
    noise = tf.random.normal(tf.shape(img), stddev=0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    max_shift = 4
    pad = tf.pad(img, [[0, 0], [max_shift, max_shift], [0, 0]], constant_values=1.0)
    start = tf.random.uniform([], 0, 2 * max_shift + 1, dtype=tf.int32)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    img = tf.image.crop_to_bounding_box(pad, 0, start, h, w)
    return img


def make_dataset(samples: List[Sample], charset: Charset, img_h: int, img_w: int,
                 max_text_len: int, batch_size: int, shuffle: bool = True,
                 repeat: bool = True, shard_desc: str = "", include_meta: bool = False,
                 augment: bool = False) -> tf.data.Dataset:
    paths = [s.path for s in samples]
    texts = [s.text for s in samples]
    enc_T_max = int(math.ceil(img_w / 8.0))

    encoded = [charset.encode(t, add_sos_eos=True) for t in texts]
    too_long = sum(1 for ids in encoded if len(ids) > max_text_len)
    if too_long:
        logger.warning(f"{too_long} примеров длиннее max_text_len={max_text_len}. Они будут усечены.")

    def gen():
        N = len(paths)
        step = max(10, min(1000, max(1, N // 10)))
        for i, (p, t, ids) in enumerate(zip(paths, texts, encoded)):
            if i % step == 0 or i == N - 1:
                j = min(i + step, N)
                logger.info(f"[{shard_desc}] Обрабатываю файлы {i+1}–{j} из {N}")
            yield p, t, np.array(ids, dtype=np.int32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )

    def _load_map(path, text, ids):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
        if img.shape.rank == 3 and img.shape[-1] != 1:
            if tf.shape(img)[-1] > 1:
                img = img[..., :3]
            img = tf.image.rgb_to_grayscale(img)
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        scale = tf.cast(img_h, tf.float32) / tf.cast(h, tf.float32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        valid_w = tf.minimum(new_w, tf.cast(img_w, tf.int32))
        valid_len = tf.cast(tf.math.ceil(tf.cast(valid_w, tf.float32) / 8.0), tf.int32)
        valid_len = tf.minimum(valid_len, tf.constant(enc_T_max, dtype=tf.int32))
        tail_len = tf.constant(enc_T_max, dtype=tf.int32) - valid_len
        enc_key_mask = tf.concat([
            tf.ones([valid_len], tf.bool),
            tf.zeros([tail_len], tf.bool)
        ], axis=0)
        img = preprocess_image(img, img_h, img_w)
        if augment:
            img = augment_image(img)
        ids = ids[:max_text_len]
        pad_len = max_text_len - tf.shape(ids)[0]
        ids_padded = tf.pad(ids, [[0, pad_len]], constant_values=charset.pad_id)
        dec_inp = tf.concat([ids_padded[:-1], tf.constant([charset.pad_id], dtype=tf.int32)], axis=0)
        target = tf.concat([ids_padded[1:], tf.constant([charset.pad_id], dtype=tf.int32)], axis=0)
        eos_seen_before = tf.math.cumsum(tf.cast(tf.equal(target, charset.eos_id), tf.int32), axis=0, exclusive=True)
        until_eos_inclusive = tf.equal(eos_seen_before, 0)
        target_not_pad = tf.cast(
            tf.logical_and(tf.not_equal(target, charset.pad_id), until_eos_inclusive),
            tf.float32
        )
        out = {"image": img,
               "decoder_input": dec_inp,
               "target": target,
               "target_mask": target_not_pad,
               "enc_key_mask": enc_key_mask}
        if include_meta:
            out.update({"text": text, "path": path})
        return out

    if shuffle:
        ds = ds.shuffle(buffer_size=min(8192, len(paths)))
    ds = ds.map(_load_map, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, max_len: int = 10000, **kwargs):
        super().__init__(**kwargs)
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pe = tf.constant(pe[None, ...], dtype=tf.float32)

    def call(self, x):
        t = tf.shape(x)[1]
        return x + tf.cast(self.pe[:, :t, :], x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

def conv_encoder(img_h: int, img_w: int, d_model: int, dropout: float = 0.1) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(img_h, img_w, 1), name="image")
    x = inp
    for filters, k, s in [(64, 3, 1), (64, 3, 2), (128, 3, 1), (128, 3, 2), (256, 3, 1), (256, 3, 2), (512, 3, 1)]:
        x = tf.keras.layers.Conv2D(filters, k, strides=s, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Permute((2, 1, 3))(x)
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = PositionalEncoding(d_model)(x)
    return tf.keras.Model(inp, x, name="cnn_encoder")


def transformer_decoder(num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout: float, max_len: int, pad_id: int) -> tf.keras.Model:
    dec_inp = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="decoder_input")
    enc_out = tf.keras.Input(shape=(None, d_model), dtype=tf.float32, name="encoder_output")
    enc_key_mask_inp = tf.keras.Input(shape=(None,), dtype=tf.bool, name="enc_key_mask")
    enc_attn_mask = tf.keras.layers.Lambda(lambda inputs: tf.tile(tf.expand_dims(inputs[0], 1), [1, tf.shape(inputs[1])[1], 1]))([enc_key_mask_inp, dec_inp])
    tok_emb = tf.keras.layers.Embedding(vocab_size, d_model, name="tok_emb")
    x = tok_emb(dec_inp)
    x = PositionalEncoding(d_model)(x)
    dec_pad = tf.keras.layers.Lambda(lambda t: tf.not_equal(t, pad_id))(dec_inp)
    dec_attn_mask = tf.keras.layers.Lambda(lambda m: tf.tile(tf.expand_dims(m, 1), [1, tf.shape(m)[1], 1]))(dec_pad)
    for _ in range(num_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
        attn_out1 = attn1(x1, x1, attention_mask=dec_attn_mask, use_causal_mask=True)
        x = x + tf.keras.layers.Dropout(dropout)(attn_out1)
        x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
        attn_out2 = attn2(x2, enc_out, attention_mask=enc_attn_mask)
        x = x + tf.keras.layers.Dropout(dropout)(attn_out2)
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])
        x = x + tf.keras.layers.Dropout(dropout)(ff(x3))
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    class TiedOutput(tf.keras.layers.Layer):
        def __init__(self, emb_layer, vocab_size, **kwargs):
            super().__init__(**kwargs)
            self.emb_layer = emb_layer
            self.vocab_size = vocab_size
        def build(self, input_shape):
            self.bias = self.add_weight(name="bias", shape=(self.vocab_size,), initializer="zeros")
        def call(self, x):
            E = self.emb_layer.embeddings
            return tf.linalg.matmul(x, E, transpose_b=True) + self.bias

    logits = TiedOutput(tok_emb, vocab_size)(x)
    return tf.keras.Model([dec_inp, enc_out, enc_key_mask_inp], logits, name="transformer_decoder")


class TrOCR(tf.keras.Model):
    def __init__(self, img_h: int, img_w: int, vocab_size: int, max_text_len: int,
                 d_model: int = 256, num_heads: int = 8, dff: int = 512,
                 enc_dropout: float = 0.1, dec_dropout: float = 0.1, dec_layers: int = 4,
                 pad_id: int = 0):
        super().__init__()
        self.max_text_len = max_text_len
        self.encoder = conv_encoder(img_h, img_w, d_model, dropout=enc_dropout)
        self.decoder = transformer_decoder(dec_layers, d_model, num_heads, dff, vocab_size, dropout=dec_dropout, max_len=max_text_len, pad_id=pad_id)

    def call(self, inputs, training=False):
        img = inputs["image"]
        dec_inp = inputs["decoder_input"]
        enc_key_mask = inputs.get("enc_key_mask")
        enc = self.encoder(img, training=training)
        if enc_key_mask is None:
            B = tf.shape(enc)[0]
            T_k = tf.shape(enc)[1]
            enc_key_mask = tf.ones([B, T_k], dtype=tf.bool)
        logits = self.decoder([dec_inp, enc, enc_key_mask], training=training)
        return logits


def masked_ce(y_true, y_pred, mask, vocab_size, label_smoothing: float = 0.0):
    y_pred = tf.cast(y_pred, tf.float32)
    if label_smoothing and label_smoothing > 0.0:
        y1 = tf.one_hot(y_true, depth=vocab_size, dtype=tf.float32)
        loss = tf.keras.losses.categorical_crossentropy(y1, y_pred, from_logits=True, label_smoothing=label_smoothing)
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(mask, tf.float32)
    loss = loss * mask
    denom = tf.reduce_sum(mask)
    return tf.reduce_sum(loss) / tf.maximum(denom, 1.0)


class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds: tf.data.Dataset, charset: Charset, max_batches: int, run_dir: str,
                 best_ckpt_path: str, monitor_metric: str = 'cer'):
        super().__init__()
        self.val_ds = val_ds
        self.charset = charset
        self.max_batches = max_batches
        self.run_dir = run_dir
        self.best_ckpt_path = best_ckpt_path
        self.best_score = float('inf')
        self.monitor_metric = monitor_metric

    def on_epoch_end(self, epoch, logs=None):
        total_cer, total_wer, n = 0.0, 0.0, 0
        lines = []
        for b, batch in enumerate(self.val_ds.take(self.max_batches)):
            preds = greedy_decode(self.model, batch["image"], batch.get("enc_key_mask"), self.charset, self.model.max_text_len)
            refs = [x.decode('utf-8') for x in batch['text'].numpy().tolist()]
            for i in range(len(preds)):
                pred = preds[i]
                ref = refs[i]
                _cer = cer(ref, pred)
                _wer = wer(ref, pred)
                total_cer += _cer
                total_wer += _wer
                n += 1
                lines.append(f"{batch['path'][i].numpy().decode('utf-8')} | pred: \"{pred}\" | ref: \"{ref}\" | wer: {_wer:.4f}")
        avg_cer = total_cer / max(1, n)
        avg_wer = total_wer / max(1, n)
        log_path = os.path.join(self.run_dir, f"eval_epoch_{epoch+1:03d}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("".join(lines + ["", f"AVG CER: {avg_cer:.4f}", f"AVG WER: {avg_wer:.4f}"]))
        logger.info(f"[Eval] Epoch {epoch+1}: CER={avg_cer:.4f} WER={avg_wer:.4f} -> report: {log_path}")
        if (self.monitor_metric == 'cer' and avg_cer < self.best_score) or (self.monitor_metric == 'wer' and avg_wer < self.best_score):
            self.best_score = avg_cer if self.monitor_metric == 'cer' else avg_wer
            self.model.save_weights(self.best_ckpt_path)
            logger.info(f"New best {self.monitor_metric.upper()} {self.best_score:.4f}. Saved weights to {self.best_ckpt_path}")


def greedy_decode(model: TrOCR, images: tf.Tensor, enc_key_mask: tf.Tensor, charset: Charset, max_len: int) -> List[str]:
    B = tf.shape(images)[0]
    dec = tf.fill([B, max_len], tf.cast(charset.pad_id, tf.int32))
    first_col = tf.stack([tf.range(B, dtype=tf.int32), tf.zeros([B], dtype=tf.int32)], axis=1)
    dec = tf.tensor_scatter_nd_update(dec, first_col, tf.fill([B], tf.cast(charset.sos_id, tf.int32)))
    enc = model.encoder(images, training=False)
    if enc_key_mask is None:
        T_k = tf.shape(enc)[1]
        enc_key_mask = tf.ones([B, T_k], dtype=tf.bool)
    finished = tf.zeros([B], dtype=tf.bool)
    for t in range(1, max_len):
        logits = model.decoder([dec, enc, enc_key_mask], training=False)
        step_logits = logits[:, t-1, :]
        next_id = tf.argmax(step_logits, axis=-1, output_type=tf.int32)
        idx = tf.stack([tf.range(B, dtype=tf.int32), tf.fill([B], tf.cast(t, tf.int32))], axis=1)
        dec = tf.tensor_scatter_nd_update(dec, idx, next_id)
        finished = tf.logical_or(finished, tf.equal(next_id, charset.eos_id))
        if tf.reduce_all(finished):
            break
    out_texts = []
    dec_np = dec.numpy()
    for seq in dec_np:
        ids = list(seq)
        if charset.eos_id in ids:
            cut = ids.index(charset.eos_id)
            ids = ids[1:cut]
        else:
            ids = ids[1:]
        out_texts.append(charset.decode(ids))
    return out_texts


@dataclass
class TrainConfig:
    data_root: str
    run_dir: str
    img_h: int = 64
    img_w: int = 512
    max_text_len: int = 80
    batch_size: int = 32
    epochs: int = 40
    steps_per_epoch: Optional[int] = None
    val_batches_for_eval: int = 4
    d_model: int = 256
    num_heads: int = 8
    dff: int = 512
    enc_dropout: float = 0.1
    dec_dropout: float = 0.1
    dec_layers: int = 4
    monitor_metric: str = 'cer'
    mixed_precision: bool = False
    seed: int = 42
    label_smoothing: float = 0.0
    learning_rate: float = 2e-4
    init_weights: Optional[str] = None


def save_config(cfg: TrainConfig, charset: Charset):
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w", encoding="utf-8") as f:
        obj = asdict(cfg)
        obj["vocab_size"] = len(charset.tokens)
        json.dump(obj, f, ensure_ascii=False, indent=2)
    charset.save(os.path.join(cfg.run_dir, "charset.json"))


def build_and_compile(cfg: TrainConfig, vocab_size: int, charset: Charset, freeze_encoder: bool = False) -> tf.keras.Model:
    if cfg.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled (float16)")
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
    model = TrOCR(
        img_h=cfg.img_h, img_w=cfg.img_w, vocab_size=vocab_size, max_text_len=cfg.max_text_len,
        d_model=cfg.d_model, num_heads=cfg.num_heads, dff=cfg.dff,
        enc_dropout=cfg.enc_dropout, dec_dropout=cfg.dec_dropout, dec_layers=cfg.dec_layers,
        pad_id=charset.pad_id,
    )

    class TrainWrapper(tf.keras.Model):
        def __init__(self, inner: TrOCR, vocab_size: int, label_smoothing: float = 0.05, **kwargs):
            super().__init__(**kwargs)
            self.inner = inner
            self.vocab_size = int(vocab_size)
            self.label_smoothing = float(label_smoothing)
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        @property
        def metrics(self):
            return [self.loss_tracker]
        def call(self, inputs, training=False):
            return self.inner(inputs, training=training)
        def train_step(self, data):
            x = {"image": data["image"], "decoder_input": data["decoder_input"], "enc_key_mask": data.get("enc_key_mask")}
            y_true = data["target"]
            mask = data["target_mask"]
            with tf.GradientTape() as tape:
                logits = self.inner(x, training=True)
                loss = masked_ce(y_true, logits, mask, self.vocab_size, self.label_smoothing)
            grads = tape.gradient(loss, self.inner.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.inner.trainable_variables))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}
        def test_step(self, data):
            x = {"image": data["image"], "decoder_input": data["decoder_input"], "enc_key_mask": data.get("enc_key_mask")}
            y_true = data["target"]
            mask = data["target_mask"]
            logits = self.inner(x, training=False)
            loss = masked_ce(y_true, logits, mask, self.vocab_size, self.label_smoothing)
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}
        @property
        def max_text_len(self):
            return self.inner.max_text_len
        @property
        def encoder(self):
            return self.inner.encoder
        @property
        def decoder(self):
            return self.inner.decoder
        def save_weights(self, *args, **kwargs):
            return self.inner.save_weights(*args, **kwargs)
        def load_weights(self, *args, **kwargs):
            ret = self.inner.load_weights(*args, **kwargs)
            return ret

    wrapper = TrainWrapper(model, vocab_size=vocab_size, label_smoothing=cfg.label_smoothing)
    if freeze_encoder:
        wrapper.encoder.trainable = False
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0), jit_compile=False)
    return wrapper


def prepare_datasets(cfg: TrainConfig, subset: str) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], Charset, List[Sample], List[Sample]]:
    root = cfg.data_root
    if subset == 'full':
        train_json = os.path.join(root, 'train.json')
        test_json = os.path.join(root, 'test.json')
        train_dir = os.path.join(root, 'train')
        test_dir = os.path.join(root, 'test')
        train_samples = read_json_list(train_json, train_dir)
        val_samples = read_json_list(test_json, test_dir)
    elif subset == 'tt':
        tt_json = os.path.join(root, 'tt.json')
        tt_dir = os.path.join(root, 'tt')
        train_samples = read_json_list(tt_json, tt_dir)
        val_samples = train_samples
    else:
        raise ValueError("subset must be 'full' or 'tt'")
    charset = Charset(extra_texts=[s.text for s in train_samples + val_samples])
    train_ds = make_dataset(train_samples, charset, cfg.img_h, cfg.img_w, cfg.max_text_len,
                            cfg.batch_size, shuffle=True, repeat=True, shard_desc='train', augment=True)
    val_ds = make_dataset(val_samples, charset, cfg.img_h, cfg.img_w, cfg.max_text_len,
                          cfg.batch_size, shuffle=False, repeat=False, shard_desc='val', include_meta=True, augment=False)
    return train_ds, val_ds, charset, train_samples, val_samples


def train_entry(cfg: TrainConfig, subset: str):
    setup_hardware()
    train_ds, val_ds, charset, train_samples, val_samples = prepare_datasets(cfg, subset)
    save_config(cfg, charset)
    model = build_and_compile(cfg, vocab_size=len(charset.tokens), charset=charset)
    if cfg.init_weights:
        dummy = {"image": tf.zeros([1, cfg.img_h, cfg.img_w, 1], tf.float32),
                 "decoder_input": tf.fill([1, cfg.max_text_len], tf.cast(charset.sos_id, tf.int32))}
        _ = model(dummy, training=False)
        wpath = cfg.init_weights if os.path.isabs(cfg.init_weights) else os.path.join(cfg.run_dir, cfg.init_weights)
        model.load_weights(wpath)
        logger.info(f"Loaded initial weights from {wpath}")
    os.makedirs(os.path.join(cfg.run_dir, 'checkpoints'), exist_ok=True)
    ckpt_path = os.path.join(cfg.run_dir, 'checkpoints', 'epoch_{epoch:03d}.weights.h5')
    best_path = os.path.join(cfg.run_dir, 'best.weights.h5')
    val_steps = min(cfg.val_batches_for_eval, max(1, math.ceil(len(val_samples) / cfg.batch_size)))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_freq='epoch'),
        EvalCallback(
            val_ds=val_ds,
            charset=charset,
            max_batches=val_steps,
            run_dir=cfg.run_dir,
            best_ckpt_path=best_path,
            monitor_metric=cfg.monitor_metric,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.run_dir, 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    ]
    steps_per_epoch = cfg.steps_per_epoch or math.ceil(len(train_samples) / cfg.batch_size)
    logger.info("Гиперпараметры: " + json.dumps(asdict(cfg), ensure_ascii=False))
    logger.info(f"Размер словаря: {len(charset.tokens)}. Пример токенов: {charset.tokens[:60]} ...")
    model.fit(
        train_ds,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    model.save_weights(os.path.join(cfg.run_dir, 'last.weights.h5'))
    logger.info("Обучение завершено. Сохранены last.weights.h5 и best.weights.h5 (по метрике из EvalCallback).")


def load_model_from_run(run_dir: str) -> Tuple[tf.keras.Model, Charset, TrainConfig]:
    charset_path = os.path.join(run_dir, 'charset.json')
    cfg_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(charset_path):
        raise FileNotFoundError(f"Не найден {charset_path}.")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Не найден {cfg_path}.")
    charset = Charset.load(charset_path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_obj = json.load(f)
    cfg = TrainConfig(**{k: v for k, v in cfg_obj.items() if k in TrainConfig.__annotations__})
    model = build_and_compile(cfg, vocab_size=len(charset.tokens), charset=charset)
    return model, charset, cfg


def infer_entry(data_root: str, split: str, run_dir: str, weights: str,
                img_h: Optional[int], img_w: Optional[int], max_text_len: Optional[int], batch_size: int):
    _, charset, cfg = load_model_from_run(run_dir)
    if img_h: cfg.img_h = img_h
    if img_w: cfg.img_w = img_w
    if max_text_len: cfg.max_text_len = max_text_len
    model = build_and_compile(cfg, vocab_size=len(charset.tokens), charset=charset)
    dummy = {"image": tf.zeros([1, cfg.img_h, cfg.img_w, 1], tf.float32),
             "decoder_input": tf.fill([1, cfg.max_text_len], tf.cast(charset.sos_id, tf.int32))}
    _ = model(dummy, training=False)
    wpath = weights if os.path.isabs(weights) else os.path.join(run_dir, weights)
    model.load_weights(wpath)
    if split == 'tt':
        json_path = os.path.join(data_root, 'tt.json'); images_dir = os.path.join(data_root, 'tt')
    elif split == 'test':
        json_path = os.path.join(data_root, 'test.json'); images_dir = os.path.join(data_root, 'test')
    elif split == 'train':
        json_path = os.path.join(data_root, 'train.json'); images_dir = os.path.join(data_root, 'train')
    else:
        raise ValueError("split must be tt/test/train")
    samples = read_json_list(json_path, images_dir)
    ds = make_dataset(samples, charset, cfg.img_h, cfg.img_w, cfg.max_text_len,
                      batch_size, shuffle=False, repeat=False, shard_desc='infer', include_meta=True, augment=False)
    total_cer, total_wer, n = 0.0, 0.0, 0
    for batch in ds:
        preds = greedy_decode(model, batch["image"], batch.get("enc_key_mask"), charset, cfg.max_text_len)
        refs = [x.decode('utf-8') for x in batch['text'].numpy().tolist()]
        for i in range(len(preds)):
            pred = preds[i]
            ref = refs[i]
            _cer = cer(ref, pred)
            _wer = wer(ref, pred)
            total_cer += _cer
            total_wer += _wer
            n += 1
            print(f"{batch['path'][i].numpy().decode('utf-8')} | pred: \"{pred}\" | ref: \"{ref}\" | wer: {_wer:.4f}")
    if n > 0:
        print(f"AVG CER: {total_cer / n:.4f}")
        print(f"AVG WER: {total_wer / n:.4f}")


def finetune_entry(base_run: str, new_run: str, data_root: str, subset: str,
                   weights: str, epochs: int, steps_per_epoch: Optional[int], batch_size: int,
                   img_h: Optional[int], img_w: Optional[int], max_text_len: Optional[int],
                   lr: float, freeze_encoder: bool, enc_dropout: Optional[float], dec_dropout: Optional[float],
                   label_smoothing: Optional[float], mixed_precision: bool):
    setup_hardware()
    _, base_charset, base_cfg = load_model_from_run(base_run)
    cfg = TrainConfig(
        data_root=data_root or base_cfg.data_root,
        run_dir=new_run,
        img_h=img_h or base_cfg.img_h,
        img_w=img_w or base_cfg.img_w,
        max_text_len=max_text_len or base_cfg.max_text_len,
        batch_size=batch_size or base_cfg.batch_size,
        epochs=epochs or base_cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        d_model=base_cfg.d_model,
        num_heads=base_cfg.num_heads,
        dff=base_cfg.dff,
        enc_dropout=enc_dropout if enc_dropout is not None else base_cfg.enc_dropout,
        dec_dropout=dec_dropout if dec_dropout is not None else base_cfg.dec_dropout,
        dec_layers=base_cfg.dec_layers,
        monitor_metric=base_cfg.monitor_metric,
        mixed_precision=mixed_precision if mixed_precision is not None else base_cfg.mixed_precision,
        label_smoothing=label_smoothing if label_smoothing is not None else base_cfg.label_smoothing,
        learning_rate=lr if lr is not None else base_cfg.learning_rate,
    )
    if subset not in ('full', 'tt'):
        raise ValueError("subset must be 'full' or 'tt'")
    train_ds, val_ds, charset_tmp, train_samples, val_samples = prepare_datasets(cfg, subset)
    charset = base_charset
    save_config(cfg, charset)
    model = build_and_compile(cfg, vocab_size=len(charset.tokens), charset=charset, freeze_encoder=freeze_encoder)
    dummy = {"image": tf.zeros([1, cfg.img_h, cfg.img_w, 1], tf.float32),
             "decoder_input": tf.fill([1, cfg.max_text_len], tf.cast(charset.sos_id, tf.int32))}
    _ = model(dummy, training=False)
    wpath = weights
    if weights in ('best.weights.h5', 'last.weights.h5') and not os.path.isabs(weights):
        wpath = os.path.join(base_run, weights)
    model.load_weights(wpath)
    logger.info(f"Loaded base weights from {wpath}")
    os.makedirs(os.path.join(cfg.run_dir, 'checkpoints'), exist_ok=True)
    ckpt_path = os.path.join(cfg.run_dir, 'checkpoints', 'epoch_{epoch:03d}.weights.h5')
    best_path = os.path.join(cfg.run_dir, 'best.weights.h5')
    val_steps = min(cfg.val_batches_for_eval, max(1, math.ceil(len(val_samples) / cfg.batch_size)))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_freq='epoch'),
        EvalCallback(val_ds, charset, val_steps, cfg.run_dir, best_path, monitor_metric=cfg.monitor_metric),
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.run_dir, 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    ]
    steps = cfg.steps_per_epoch or math.ceil(len(train_samples) / cfg.batch_size)
    logger.info("Гиперпараметры (fine-tune): " + json.dumps(asdict(cfg), ensure_ascii=False))
    model.fit(
        train_ds,
        epochs=cfg.epochs,
        steps_per_epoch=steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    model.save_weights(os.path.join(cfg.run_dir, 'last.weights.h5'))
    logger.info("Дообучение завершено. Сохранены last.weights.h5 и best.weights.h5.")


def parse_args():
    p = argparse.ArgumentParser(description="TrOCR (TensorFlow/Keras)")
    sub = p.add_subparsers(dest='cmd', required=True)
    default_data_root = os.path.join('data', 'handwritten')
    p_train = sub.add_parser('train')
    p_train.add_argument('--data-root', type=str, default=default_data_root)
    p_train.add_argument('--run-dir', type=str, required=True)
    p_train.add_argument('--epochs', type=int, default=40)
    p_train.add_argument('--steps-per-epoch', type=int, default=None)
    p_train.add_argument('--batch-size', type=int, default=32)
    p_train.add_argument('--img-h', type=int, default=64)
    p_train.add_argument('--img-w', type=int, default=512)
    p_train.add_argument('--max-text-len', type=int, default=80)
    p_train.add_argument('--mixed-precision', action='store_true')
    p_train.add_argument('--enc-dropout', type=float, default=0.1)
    p_train.add_argument('--dec-dropout', type=float, default=0.1)
    p_train.add_argument('--label-smoothing', type=float, default=0.05)
    p_train.add_argument('--learning-rate', type=float, default=2e-4)
    p_train.add_argument('--init-weights', type=str, default=None)

    p_smoke = sub.add_parser('smoke')
    p_smoke.add_argument('--data-root', type=str, default=default_data_root)
    p_smoke.add_argument('--run-dir', type=str, required=True)
    p_smoke.add_argument('--epochs', type=int, default=5)
    p_smoke.add_argument('--steps-per-epoch', type=int, default=200)
    p_smoke.add_argument('--batch-size', type=int, default=32)
    p_smoke.add_argument('--img-h', type=int, default=64)
    p_smoke.add_argument('--img-w', type=int, default=512)
    p_smoke.add_argument('--max-text-len', type=int, default=32)
    p_smoke.add_argument('--enc-dropout', type=float, default=0.05)
    p_smoke.add_argument('--dec-dropout', type=float, default=0.05)
    p_smoke.add_argument('--label-smoothing', type=float, default=0.1)
    p_smoke.add_argument('--learning-rate', type=float, default=2e-4)
    p_smoke.add_argument('--init-weights', type=str, default=None)

    p_infer = sub.add_parser('infer')
    p_infer.add_argument('--data-root', type=str, default=default_data_root)
    p_infer.add_argument('--split', type=str, default='tt', choices=['tt','test','train'])
    p_infer.add_argument('--run-dir', type=str, required=True)
    p_infer.add_argument('--weights', type=str, default='best.weights.h5')
    p_infer.add_argument('--batch-size', type=int, default=32)
    p_infer.add_argument('--img-h', type=int, default=None)
    p_infer.add_argument('--img-w', type=int, default=None)
    p_infer.add_argument('--max-text-len', type=int, default=None)

    p_ft = sub.add_parser('finetune')
    p_ft.add_argument('--base-run', type=str, required=True)
    p_ft.add_argument('--new-run', type=str, required=True)
    p_ft.add_argument('--data-root', type=str, default=default_data_root)
    p_ft.add_argument('--subset', type=str, choices=['full','tt'], default='full')
    p_ft.add_argument('--weights', type=str, default='best.weights.h5')
    p_ft.add_argument('--epochs', type=int, default=10)
    p_ft.add_argument('--steps-per-epoch', type=int, default=None)
    p_ft.add_argument('--batch-size', type=int, default=32)
    p_ft.add_argument('--img-h', type=int, default=None)
    p_ft.add_argument('--img-w', type=int, default=None)
    p_ft.add_argument('--max-text-len', type=int, default=None)
    p_ft.add_argument('--lr', type=float, default=5e-5)
    p_ft.add_argument('--freeze-encoder', action='store_true')
    p_ft.add_argument('--enc-dropout', type=float, default=None)
    p_ft.add_argument('--dec-dropout', type=float, default=None)
    p_ft.add_argument('--label-smoothing', type=float, default=None)
    p_ft.add_argument('--mixed-precision', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd in ('train', 'smoke'):
        subset = 'full' if args.cmd == 'train' else 'tt'
        cfg = TrainConfig(
            data_root=args.data_root,
            run_dir=args.run_dir,
            img_h=args.img_h,
            img_w=args.img_w,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            mixed_precision=getattr(args, 'mixed_precision', False),
            enc_dropout=getattr(args, 'enc_dropout', 0.1),
            dec_dropout=getattr(args, 'dec_dropout', 0.1),
            label_smoothing=getattr(args, 'label_smoothing', 0.0),
            learning_rate=getattr(args, 'learning_rate', 2e-4),
            init_weights=getattr(args, 'init_weights', None),
        )
        train_entry(cfg, subset=subset)
    elif args.cmd == 'infer':
        infer_entry(
            data_root=args.data_root,
            split=args.split,
            run_dir=args.run_dir,
            weights=args.weights,
            img_h=args.img_h,
            img_w=args.img_w,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
        )
    elif args.cmd == 'finetune':
        finetune_entry(
            base_run=args.base_run,
            new_run=args.new_run,
            data_root=args.data_root,
            subset=args.subset,
            weights=args.weights,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            img_h=args.img_h,
            img_w=args.img_w,
            max_text_len=args.max_text_len,
            lr=args.lr,
            freeze_encoder=args.freeze_encoder,
            enc_dropout=args.enc_dropout,
            dec_dropout=args.dec_dropout,
            label_smoothing=args.label_smoothing,
            mixed_precision=args.mixed_precision,
        )
    else:
        raise ValueError("Unknown command")


if __name__ == '__main__':
    main()

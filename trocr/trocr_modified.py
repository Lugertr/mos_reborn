#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose
-------
TensorFlow/Keras-реализация TrOCR с минимально необходимой обвязкой:
  • конвейер данных (чтение json-списков, препроцессинг, аугментации),
  • модель: CNN-энкодер → последовательность признаков → Transformer-декодер,
  • обучение с masked CE и валидацией через Greedy decode (EvalCallback),
  • функции train/finetune и загрузка сохранённого ранa для инференса.

Design notes
------------
- Вход — кроп строки; сначала нормализация по высоте до `img_h`, затем паддинг/кроп до `img_w`.
- Маска ключей энкодера (`enc_key_mask`) строится из фактической ширины признаков (W/8).
- Выходной словарь жёстко задан через Charset (включая дореформенные буквы и римские цифры).
- Сохранение `config.json` и `charset.json` в папку run позволяет воспроизводить инференс.
"""

from __future__ import annotations
import os
import sys
import json
import math
import argparse
import logging
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np

# До инициализации TF — переменные окружения
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_USE_XLA", "0")
os.environ.setdefault("TF_ENABLE_LAYOUT_OPTIMIZER", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# -------------------- ЛОГГЕР --------------------
logger = logging.getLogger("trocr")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# -------------------- АППАРАТУРА --------------------
def setup_hardware(seed: Optional[int] = None):
    """
    Подготовка окружения TF: отключение XLA/оптимизаторов макета, memory growth на GPU,
    установка сидов.

    Args:
        seed: Если задан — детерминирует NumPy/TF/Random.

    Notes:
        Без memory growth TF может «съесть» всю видеопамять на старте.
    """
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

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seeds set to {seed}")


# -------------------- СИМВОЛЫ/ТОКЕНЫ --------------------
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
    """
    Словарь символов и служебных токенов:
      • фиксированный набор REQUIRED_CHARS,
      • дополнительные символы из обучающего текста (опционально),
      • отображения char↔id и token↔id.

    Methods:
        encode(text, add_sos_eos): список id (с опциональными <s> ... </s>)
        decode(ids, strip_special): строка без служебных токенов (по умолчанию)
        save/load: сериализация в JSON-файл.
    """
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
    def pad_id(self) -> int: return self.token_to_id[SPECIAL_TOKENS["PAD"]]
    @property
    def sos_id(self) -> int: return self.token_to_id[SPECIAL_TOKENS["SOS"]]
    @property
    def eos_id(self) -> int: return self.token_to_id[SPECIAL_TOKENS["EOS"]]
    @property
    def unk_id(self) -> int: return self.token_to_id[SPECIAL_TOKENS["UNK"]]

    def encode(self, text: str, add_sos_eos: bool = False) -> List[int]:
        """Преобразовать строку в последовательность id (с опциональными <s> и </s>)."""
        ids = [self.char_to_id.get(ch, self.unk_id) for ch in text]
        if add_sos_eos:
            return [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """
        Преобразовать id → строка.

        Args:
            ids: последовательность индексов.
            strip_special: если True — служебные токены не включаются в вывод.
        """
        out = []
        for i in ids:
            if i == self.pad_id and strip_special:
                continue
            tok = self.id_to_token.get(int(i), SPECIAL_TOKENS["UNK"])
            if strip_special and tok in SPECIAL_TOKENS.values():
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: str):
        """Сохранить список токенов в JSON-файл."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Charset":
        """Загрузить словарь из JSON-файла, восстановив отображения id↔token/char."""
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        c = Charset(extra_texts=None)
        c.tokens = obj["tokens"]
        c.token_to_id = {tok: i for i, tok in enumerate(c.tokens)}
        c.id_to_token = {i: tok for tok, i in c.token_to_id.items()}
        c.char_to_id = {ch: i for i, ch in enumerate(c.tokens) if len(ch) == 1}
        return c


# -------------------- МЕТРИКИ --------------------
def _levenshtein(a: List[str], b: List[str]) -> int:
    """Стандартное расстояние Левенштейна между последовательностями символов/слов."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate (по символам)."""
    ref_chars, hyp_chars = list(ref), list(hyp)
    if len(ref_chars) == 0:
        return float(len(hyp_chars) > 0)
    return _levenshtein(ref_chars, hyp_chars) / max(1, len(ref_chars))

def wer(ref: str, hyp: str) -> float:
    """Word Error Rate (по словам)."""
    ref_words, hyp_words = ref.split(), hyp.split()
    if len(ref_words) == 0:
        return float(len(hyp_words) > 0)
    return _levenshtein(ref_words, hyp_words) / max(1, len(ref_words))


@dataclass
class Sample:
    """Единица датасета: путь к картинке и соответствующий целевой текст."""
    path: str
    text: str

def read_json_list(json_path: str, images_dir: str) -> List[Sample]:
    """
    Прочитать список образцов из JSON: [{"file_name": "...", "text": "..."}].

    Логирует предупреждения для отсутствующих файлов и пустых целевых строк.
    """
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


# -------------------- ПРЕПРОЦЕССИНГ --------------------
def preprocess_image(img, img_h: int, img_w: int):
    """
    Нормализовать вход под размеры модели:
      • конвертация в float32 [0..1] и одноканальный вид (H, W, 1),
      • масштабирование по высоте до img_h с сохранением пропорций,
      • правый паддинг до ширины img_w (белым), либо кроп при избыточной ширине.
    """
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
        pad = tf.fill([img_h, pad_w, 1], 1.0)  # белая «подложка»
        img = tf.concat([img, pad], axis=1)
    else:
        img = tf.image.crop_to_bounding_box(img, 0, 0, img_h, img_w)
    return img

def augment_image(img):
    """
    Небольшие аугментации для робастности: контраст/яркость/шум/случайный сдвиг по ширине.
    Используется только для train-пайплайна.
    """
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
    """
    Построить tf.data.Dataset для обучения/валидации/инференса.

    Args:
        samples: список Sample(path,text).
        charset: словарь символов для кодирования целей.
        img_h, img_w: размеры входа модели.
        max_text_len: фиксированная длина декодера.
        batch_size: размер батча.
        shuffle/repeat: поведение датасета.
        shard_desc: метка для логов ("train"/"val"/"infer").
        include_meta: если True — в батч будут добавлены "text"/"path".
        augment: если True — применяются лёгкие аугментации.

    Returns:
        Dataset с полями:
            image, decoder_input, target, target_mask, enc_key_mask [, text, path]
    """
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


# -------------------- МОДЕЛЬ --------------------
class PositionalEncoding(tf.keras.layers.Layer):
    """Синусоидальные позиционные эмбеддинги (как в оригинальном Transformer)."""
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
    """
    Лёгкий CNN-энкодер: несколько Conv2D-блоков с downsample по ширине, затем
    усреднение по высоте → Dense(d_model) → позиционные эмбеддинги.
    """
    inp = tf.keras.Input(shape=(img_h, img_w, 1), name="image")
    x = inp
    for filters, k, s in [(64, 3, 1), (64, 3, 2), (128, 3, 1), (128, 3, 2),
                        (256, 3, 1), (256, 3, 2), (512, 3, 1)]:
        x = tf.keras.layers.Conv2D(filters, k, strides=s, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Permute((2, 1, 3))(x)              # (B, W', H', C) -> (B, H', W', C), затем
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)  # усреднение по "высоте"
    x = tf.keras.layers.Dense(d_model)(x)
    x = PositionalEncoding(d_model)(x)
    return tf.keras.Model(inp, x, name="cnn_encoder")

def transformer_decoder(num_layers: int, d_model: int, num_heads: int, dff: int,
                        vocab_size: int, dropout: float, max_len: int, pad_id: int) -> tf.keras.Model:
    """
    Transformer-декодер с:
      • causal self-attention по декодерным позициям,
      • cross-attention на выход энкодера (маска по enc_key_mask),
      • feed-forward блоки и weight tying для выходного слоя (через Embedding).
    """
    dec_inp = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="decoder_input")
    enc_out = tf.keras.Input(shape=(None, d_model), dtype=tf.float32, name="encoder_output")
    enc_key_mask_inp = tf.keras.Input(shape=(None,), dtype=tf.bool, name="enc_key_mask")
    enc_attn_mask = tf.keras.layers.Lambda(
        lambda inputs: tf.tile(tf.expand_dims(inputs[0], 1), [1, tf.shape(inputs[1])[1], 1])
    )([enc_key_mask_inp, dec_inp])
    tok_emb = tf.keras.layers.Embedding(vocab_size, d_model, name="tok_emb")
    x = tok_emb(dec_inp)
    x = PositionalEncoding(d_model)(x)
    dec_pad = tf.keras.layers.Lambda(lambda t: tf.not_equal(t, pad_id))(dec_inp)
    dec_attn_mask = tf.keras.layers.Lambda(
        lambda m: tf.tile(tf.expand_dims(m, 1), [1, tf.shape(m)[1], 1])
    )(dec_pad)
    for _ in range(num_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        attn_out1 = attn1(x1, x1, attention_mask=dec_attn_mask, use_causal_mask=True)
        x = x + tf.keras.layers.Dropout(dropout)(attn_out1)

        x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
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
        """Выходной слой, «связанный» с матрицей эмбеддингов (weight tying)."""
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
    """
    Модель: CNN-encoder → Transformer-decoder.
    Хранит `max_text_len` и открывает доступ к encoder/decoder для внешних вызовов.
    """
    def __init__(self, img_h: int, img_w: int, vocab_size: int, max_text_len: int,
                d_model: int = 256, num_heads: int = 8, dff: int = 512,
                enc_dropout: float = 0.1, dec_dropout: float = 0.1, dec_layers: int = 4,
                pad_id: int = 0):
        super().__init__()
        self.max_text_len = max_text_len
        self.encoder = conv_encoder(img_h, img_w, d_model, dropout=enc_dropout)
        self.decoder = transformer_decoder(dec_layers, d_model, num_heads, dff, vocab_size,
                                        dropout=dec_dropout, max_len=max_text_len, pad_id=pad_id)

    def call(self, inputs, training=False):
        """
        Прямой проход: inputs = {"image", "decoder_input", ["enc_key_mask"]} → logits.

        Если `enc_key_mask` не передан — берётся маска из полной длины выхода энкодера.
        """
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


# -------------------- ЛОСС --------------------
def masked_ce(y_true, y_pred, mask, vocab_size, label_smoothing: float = 0.0):
    """
    Маскированная кросс-энтропия для последовательностей:
      • поддержка label smoothing,
      • усреднение только по валидным (mask==1) позициям.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    if label_smoothing and label_smoothing > 0.0:
        y1 = tf.one_hot(y_true, depth=vocab_size, dtype=tf.float32)
        loss = tf.keras.losses.categorical_crossentropy(
            y1, y_pred, from_logits=True, label_smoothing=label_smoothing
        )
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(mask, tf.float32)
    loss = loss * mask
    denom = tf.reduce_sum(mask)
    return tf.reduce_sum(loss) / tf.maximum(denom, 1.0)


# -------------------- КОЛЛБЭКИ --------------------
class EvalCallback(tf.keras.callbacks.Callback):
    """
    Валидация после каждой эпохи через greedy-декод:
      • считает CER/WER на `max_batches` батчах вал-данных,
      • пишет отчёт в run_dir,
      • сохраняет best.weights.h5 по выбранной метрике (cer|wer).
    """
    def __init__(self, val_ds: tf.data.Dataset, charset: Charset, max_batches: int,
                run_dir: str, best_ckpt_path: str, monitor_metric: str = 'cer'):
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
            preds = greedy_decode(self.model, batch["image"], batch.get("enc_key_mask"),
                                self.charset, self.model.max_text_len)
            refs = [x.decode('utf-8') for x in batch['text'].numpy().tolist()]
            for i in range(len(preds)):
                pred, ref = preds[i], refs[i]
                _cer, _wer = cer(ref, pred), wer(ref, pred)
                total_cer += _cer
                total_wer += _wer
                n += 1
                lines.append(
                    f"{batch['path'][i].numpy().decode('utf-8')} | pred: \"{pred}\" | ref: \"{ref}\" | wer: {_wer:.4f}"
                )
        avg_cer = total_cer / max(1, n)
        avg_wer = total_wer / max(1, n)
        log_path = os.path.join(self.run_dir, f"eval_epoch_{epoch+1:03d}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines + ["", f"AVG CER: {avg_cer:.4f}", f"AVG WER: {avg_wer:.4f}"]))
        logger.info(f"[Eval] Epoch {epoch+1}: CER={avg_cer:.4f} WER={avg_wer:.4f} -> report: {log_path}")

        # сохранение лучшего чекпойнта
        score = avg_cer if self.monitor_metric == 'cer' else avg_wer
        if score < self.best_score:
            self.best_score = score
            self.model.save_weights(self.best_ckpt_path)
            logger.info(f"New best {self.monitor_metric.upper()} {self.best_score:.4f}. "
                        f"Saved weights to {self.best_ckpt_path}")

def greedy_decode(model: TrOCR, images: tf.Tensor, enc_key_mask: tf.Tensor,
                charset: Charset, max_len: int) -> List[str]:
    """
    Простой жадный декодер:
      • инициализирует <s> и пошагово выбирает argmax для следующего токена,
      • останавливается по EOS или по достижении max_len,
      • возвращает список строк, очищая служебные токены.

    Args:
        model: Объёрнутая модель (TrainWrapper или TrOCR), у которой есть encoder/decoder.
        images: Тензор (B, H, W, 1).
        enc_key_mask: Маска по времени для выхода энкодера (или None — тогда генерируется полностью).
        charset: Для извлечения id токенов и финального декода.
        max_len: Фиксированная длина для декодера.

    Returns:
        List[str]: Тексты для каждого элемента батча.
    """
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


# -------------------- КОНФИГ, ПРОФИЛИ, NAMES --------------------
@dataclass
class TrainConfig:
    """Конфигурация обучения/дообучения и инференса (сохраняется в run_dir/config.json)."""
    data_root: str
    run_dir: str
    img_h: int = 120
    img_w: int = 200
    max_text_len: int = 80
    batch_size: int = 32
    epochs: int = 40
    steps_per_epoch: Optional[int] = None
    val_batches_for_eval: int = 8
    d_model: int = 256
    num_heads: int = 8
    dff: int = 512
    enc_dropout: float = 0.1
    dec_dropout: float = 0.1
    dec_layers: int = 4
    monitor_metric: str = 'cer'
    mixed_precision: bool = False
    seed: int = 42
    label_smoothing: float = 0.10
    learning_rate: float = 2e-4
    init_weights: Optional[str] = None
    lr_decay_rate: float = 0.98
    early_stop_patience: int = 6

def dict_update(dst: dict, src: dict) -> dict:
    """Копия словаря `dst` с обновлением только непустых значений из `src`."""
    out = dict(dst)
    out.update({k: v for k, v in src.items() if v is not None})
    return out

def load_user_config(path: Optional[str]) -> Tuple[dict, dict]:
    """
    Прочитать конфигурацию пользователя:
      • JSON: словарь параметров и опционально PRESETS;
      • .py: ожидать глобальные переменные CONFIG и (опц.) PRESETS.
    """
    if not path:
        return {}, {}
    if path.endswith(".py"):
        import importlib.util, types
        spec = importlib.util.spec_from_file_location("trocr_user_config", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore
        cfg = getattr(mod, "CONFIG", {})
        presets = getattr(mod, "PRESETS", {})
        if not isinstance(cfg, dict):
            raise ValueError("В .py-конфиге должен быть словарь CONFIG")
        if presets and not isinstance(presets, dict):
            raise ValueError("В .py-конфиге PRESETS (если задан) должен быть словарём")
        return cfg, presets
    # JSON
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj, obj.get("PRESETS", {})

def apply_profile(base_cfg: dict, profile: Optional[str], presets: dict) -> dict:
    """Наложить профиль из PRESETS на базовый конфиг (если профиль существует)."""
    if not profile:
        return base_cfg
    if profile not in presets:
        logger.warning(f"--profile '{profile}' не найден в PRESETS, игнорирую.")
        return base_cfg
    return dict_update(base_cfg, presets[profile])

def build_effective_config(args, defaults: TrainConfig) -> Tuple[TrainConfig, dict]:
    """
    Слияние конфигов (приоритет сверху вниз):
      1) дефолты dataclass,
      2) CONFIG из файла,
      3) профиль PRESETS,
      4) CLI-оверайды.
    Возвращает (TrainConfig, «сырой» объединённый словарь для логов/отладки).
    """
    base = asdict(defaults)
    user_cfg, presets = load_user_config(getattr(args, "config", None))
    merged = dict_update(base, user_cfg)
    merged = apply_profile(merged, getattr(args, "profile", None), presets)

    # CLI overrides
    mapping = {
        "data_root": "data_root", "img_h": "img_h", "img_w": "img_w", "max_text_len": "max_text_len",
        "batch_size": "batch_size", "epochs": "epochs", "steps_per_epoch": "steps_per_epoch",
        "learning_rate": "learning_rate", "label_smoothing": "label_smoothing",
        "enc_dropout": "enc_dropout", "dec_dropout": "dec_dropout", "dec_layers": "dec_layers",
        "d_model": "d_model", "num_heads": "num_heads", "dff": "dff",
        "monitor_metric": "monitor_metric", "mixed_precision": "mixed_precision",
        "lr_decay_rate": "lr_decay_rate", "early_stop_patience": "early_stop_patience",
        "val_batches_for_eval": "val_batches_for_eval", "seed": "seed", "init_weights": "init_weights",
    }
    cli_overrides = {}
    for arg_name, cfg_key in mapping.items():
        if hasattr(args, arg_name):
            val = getattr(args, arg_name)
            if val is not None:
                cli_overrides[cfg_key] = val
    merged = dict_update(merged, cli_overrides)

    tc_kwargs = {k: v for k, v in merged.items() if k in TrainConfig.__annotations__}
    tc = TrainConfig(**tc_kwargs)
    return tc, merged

def normalize_subset_name(name: str) -> str:
    """
    Привести имя поднабора к канону:
      'tt'/'posttest'/'post_test' → 'postTest'
      'full' → 'full' (используется для train; для ft/val читаем train/test)
      иначе ошибка
    """
    low = name.lower()
    if low in ("tt", "posttest", "post_test"):
        return "postTest"
    if name == "full":
        return "full"
    if name == "postTest":
        return "postTest"
    raise ValueError("subset must be 'full' or 'postTest'")

# -------------------- СЕРИАЛИЗАЦИЯ РАНА --------------------
def save_config(cfg: TrainConfig, charset: Charset):
    """Сохранить текущий TrainConfig и charset в run_dir."""
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w", encoding="utf-8") as f:
        obj = asdict(cfg)
        obj["vocab_size"] = len(charset.tokens)
        json.dump(obj, f, ensure_ascii=False, indent=2)
    charset.save(os.path.join(cfg.run_dir, "charset.json"))

def build_and_compile(cfg: TrainConfig, vocab_size: int, charset: Charset,
                    freeze_encoder: bool = False) -> tf.keras.Model:
    """
    Построить TrOCR и обёртку для train/test step’ов, с оптимизатором и (опц.) mixed precision.
    Параметр `freeze_encoder` фиксирует веса CNN-энкодера (для финтюнинга).
    """
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
        """
        Обёртка над TrOCR, реализующая `train_step`/`test_step` с masked CE, и
        предоставляющая доступ к encoder/decoder/save_weights для совместимости.
        """
        def __init__(self, inner: TrOCR, vocab_size: int, label_smoothing: float = 0.05, **kwargs):
            super().__init__(**kwargs)
            self.inner = inner
            self.vocab_size = int(vocab_size)
            self.label_smoothing = float(label_smoothing)
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        @property
        def metrics(self): return [self.loss_tracker]
        def call(self, inputs, training=False): return self.inner(inputs, training=training)
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
        def max_text_len(self): return self.inner.max_text_len
        @property
        def encoder(self): return self.inner.encoder
        @property
        def decoder(self): return self.inner.decoder
        def save_weights(self, *args, **kwargs): return self.inner.save_weights(*args, **kwargs)
        def load_weights(self, *args, **kwargs): return self.inner.load_weights(*args, **kwargs)

    wrapper = TrainWrapper(model, vocab_size=vocab_size, label_smoothing=cfg.label_smoothing)
    if freeze_encoder:
        wrapper.encoder.trainable = False

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
    )
    wrapper.compile(optimizer=optimizer, jit_compile=False)
    return wrapper

def prepare_datasets(cfg: TrainConfig, subset: str) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], Charset, List[Sample], List[Sample]]:
    """
    Сконструировать train/val датасеты и Charset. Для 'full' валидируемся на test,
    для 'postTest' используем один и тот же набор (быстрый цикл отладки).
    """
    root = cfg.data_root
    subset = normalize_subset_name(subset)
    if subset == 'full':
        train_json = os.path.join(root, 'train.json')
        test_json  = os.path.join(root, 'test.json')
        train_dir  = os.path.join(root, 'train')
        test_dir   = os.path.join(root, 'test')
        train_samples = read_json_list(train_json, train_dir)
        val_samples   = read_json_list(test_json,  test_dir)
    elif subset == 'postTest':
        pt_json = os.path.join(root, 'postTest.json')
        pt_dir  = os.path.join(root, 'postTest')
        train_samples = read_json_list(pt_json, pt_dir)
        val_samples   = train_samples
    else:
        raise ValueError("subset must be 'full' or 'postTest'")

    charset = Charset(extra_texts=[s.text for s in train_samples + val_samples])
    train_ds = make_dataset(train_samples, charset, cfg.img_h, cfg.img_w, cfg.max_text_len,
                            cfg.batch_size, shuffle=True, repeat=True, shard_desc='train', augment=True)
    val_ds = make_dataset(val_samples, charset, cfg.img_h, cfg.img_w, cfg.max_text_len,
                        cfg.batch_size, shuffle=False, repeat=False, shard_desc='val', include_meta=True, augment=False)
    return train_ds, val_ds, charset, train_samples, val_samples


# -------------------- ТРЕНИРОВКА/ФТ --------------------
def train_entry(cfg: TrainConfig, subset: str):
    """
    Полное обучение с сохранением:
      • чекпойнтов по эпохам,
      • best.weights.h5 (по метрике из EvalCallback),
      • last.weights.h5 (последняя эпоха),
      • history.csv и per-epoch отчётов в run_dir.
    """
    setup_hardware(seed=cfg.seed)
    train_ds, val_ds, charset, train_samples, val_samples = prepare_datasets(cfg, subset)
    save_config(cfg, charset)

    model = build_and_compile(cfg, vocab_size=len(charset.tokens), charset=charset)
    if cfg.init_weights:
        # строим граф
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

    # расписание LR
    def lr_schedule(epoch, lr):
        rate = cfg.lr_decay_rate if cfg.lr_decay_rate is not None else 1.0
        return lr * rate

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_freq='epoch'),
        EvalCallback(val_ds=val_ds, charset=charset, max_batches=val_steps,
                    run_dir=cfg.run_dir, best_ckpt_path=best_path, monitor_metric=cfg.monitor_metric),
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.run_dir, 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.early_stop_patience, restore_best_weights=False),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0) if cfg.lr_decay_rate and cfg.lr_decay_rate != 1.0 else None,
    ]
    callbacks = [c for c in callbacks if c is not None]

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
    """
    Восстановить модель/charset/config из сохранённого запуска:
      • читает `charset.json` и `config.json`,
      • собирает модель и компилирует её (без обучения).
    """
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

def finetune_entry(base_run: str, new_run: str, data_root: str, subset: str,
                weights: str, epochs: int, steps_per_epoch: Optional[int], batch_size: int,
                img_h: Optional[int], img_w: Optional[int], max_text_len: Optional[int],
                lr: float, freeze_encoder: bool, enc_dropout: Optional[float], dec_dropout: Optional[float],
                label_smoothing: Optional[float], mixed_precision: bool,
                monitor_metric: Optional[str] = None, lr_decay_rate: Optional[float] = None,
                early_stop_patience: Optional[float] = None, val_batches_for_eval: Optional[int] = None,
                seed: Optional[int] = None):
    """
    Дообучение от базового run:
      • копируем ключевые гиперпараметры из base_run (кроме явно переопределённых),
      • загружаем base-веса, опционально «замораживаем» энкодер,
      • обучаем и сохраняем новый run в new_run.
    """
    setup_hardware(seed=seed)

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
        monitor_metric=monitor_metric or base_cfg.monitor_metric,
        mixed_precision=mixed_precision if mixed_precision is not None else base_cfg.mixed_precision,
        label_smoothing=label_smoothing if label_smoothing is not None else base_cfg.label_smoothing,
        learning_rate=lr if lr is not None else base_cfg.learning_rate,
        lr_decay_rate=lr_decay_rate if lr_decay_rate is not None else base_cfg.lr_decay_rate,
        early_stop_patience=early_stop_patience if early_stop_patience is not None else base_cfg.early_stop_patience,
        val_batches_for_eval=val_batches_for_eval if val_batches_for_eval is not None else base_cfg.val_batches_for_eval,
        seed=seed if seed is not None else base_cfg.seed,
    )

    subset = normalize_subset_name(subset)
    # Готовим датасеты для нового run'а
    train_ds, val_ds, charset_tmp, train_samples, val_samples = prepare_datasets(cfg, subset)
    charset = base_charset  # фиксируем словарь как у базовой модели
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

    def lr_schedule(epoch, lr):
        rate = cfg.lr_decay_rate if cfg.lr_decay_rate is not None else 1.0
        return lr * rate

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_freq='epoch'),
        EvalCallback(val_ds, charset, val_steps, cfg.run_dir, best_path, monitor_metric=cfg.monitor_metric),
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.run_dir, 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.early_stop_patience, restore_best_weights=False),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0) if cfg.lr_decay_rate and cfg.lr_decay_rate != 1.0 else None,
    ]
    callbacks = [c for c in callbacks if c is not None]

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


# -------------------- CLI --------------------
def parse_args():
    """
    CLI-обёртка для сценариев обучения:
      • train — полное обучение,
      • smoke — быстрый прогон (как train, но можно задать короткие циклы),
      • finetune — дообучение от существующего запуска.
    """
    p = argparse.ArgumentParser(description="TrOCR (TensorFlow/Keras)")
    p.add_argument('--config', type=str, default=None,
                help="Путь к конфигу (JSON или .py с CONFIG/PRESETS)")
    p.add_argument('--profile', type=str, default=None,
                help="Имя профиля из PRESETS (накладывается поверх CONFIG)")

    sub = p.add_subparsers(dest='cmd', required=True)

    # ---- train ----
    p_train = sub.add_parser('train', help="Полное обучение")
    p_train.add_argument('--subset', type=str, default='full',
                        choices=['full', 'postTest', 'tt'],
                        help="Какой поднабор читать из data/handwritten")
    p_train.add_argument('--run-dir', type=str, required=True,
                        help="Каталог для сохранения результатов (runs/...)")

    # ---- smoke ----
    p_smoke = sub.add_parser('smoke', help="Быстрая проверка пайплайна")
    p_smoke.add_argument('--subset', type=str, default='postTest',
                        choices=['full', 'postTest', 'tt'])
    p_smoke.add_argument('--run-dir', type=str, required=True)

    # ---- finetune ----
    p_ft = sub.add_parser('finetune', help="Дообучение от готового ранa")
    p_ft.add_argument('--base-run', type=str, required=True,
                    help="Путь к базовому ранy (где лежат charset.json, config.json)")
    p_ft.add_argument('--new-run', type=str, required=True,
                    help="Новый каталог для результатов дообучения")
    p_ft.add_argument('--subset', type=str, choices=['full', 'postTest', 'tt'],
                    default='postTest', help="Какие данные читать для фт")
    p_ft.add_argument('--weights', type=str, default='best.weights.h5',
                    help="Имя/путь весов для старта фт (относит. к base-run, если имя стандартное)")
    p_ft.add_argument('--freeze-encoder', action='store_true',
                    help="Заморозить CNN-энкодер при дообучении")

    # ---- Общие оверрайды для всех сабкоманд (без дублей!) ----
    for sp in (p_train, p_smoke, p_ft):
        # дата/формат
        sp.add_argument('--data-root', type=str, default=None,
                        help="Корень данных (по умолчанию data/handwritten)")
        sp.add_argument('--img-h', type=int, default=None)
        sp.add_argument('--img-w', type=int, default=None)
        sp.add_argument('--max-text-len', type=int, default=None)

        # батчинг и длительность
        sp.add_argument('--batch-size', type=int, default=None)
        sp.add_argument('--epochs', type=int, default=None)
        sp.add_argument('--steps-per-epoch', type=int, default=None)

        # модель/оптимизатор
        sp.add_argument('--learning-rate', type=float, default=None)
        sp.add_argument('--label-smoothing', type=float, default=None)
        sp.add_argument('--enc-dropout', type=float, default=None)
        sp.add_argument('--dec-dropout', type=float, default=None)
        sp.add_argument('--dec-layers', type=int, default=None)
        sp.add_argument('--d-model', type=int, default=None)
        sp.add_argument('--num-heads', type=int, default=None)
        sp.add_argument('--dff', type=int, default=None)

        # логика валидации/метрик/процедур
        sp.add_argument('--monitor-metric', type=str, default=None, choices=['cer', 'wer'])
        sp.add_argument('--val-batches-for-eval', type=int, default=None,
                        help="Сколько батчей вал. данных использовать в EvalCallback")
        sp.add_argument('--early-stop-patience', type=int, default=None,
                        help="Порог терпения EarlyStopping по val_loss")
        sp.add_argument('--lr-decay-rate', type=float, default=None,
                        help="(опционально) множитель экспон. распада LR, если включён в конфиге")
        sp.add_argument('--mixed-precision', action='store_true',
                        help="Включить float16 mixed precision (если железо позволяет)")
        sp.add_argument('--seed', type=int, default=None)
        sp.add_argument('--init-weights', type=str, default=None,
                        help="Начальные веса (для train/smoke). Для finetune берётся из --weights")

    return p.parse_args()

def main():
    """Основная CLI-ветка: train/smoke/finetune с формированием effective_config.json."""
    args = parse_args()

    if args.cmd in ('train', 'smoke'):
        subset = normalize_subset_name(args.subset)
        # дефолты на случай пустого конфига
        defaults = TrainConfig(
            data_root="data/handwritten",
            run_dir=args.run_dir,
        )
        cfg, raw_cfg = build_effective_config(args, defaults)
        cfg.run_dir = args.run_dir  # run_dir всегда из CLI

        # сохранить «эффективный» конфиг до токенов
        os.makedirs(cfg.run_dir, exist_ok=True)
        with open(os.path.join(cfg.run_dir, "effective_config.json"), "w", encoding="utf-8") as f:
            json.dump(raw_cfg, f, ensure_ascii=False, indent=2)

        train_entry(cfg, subset=subset)

    elif args.cmd == 'finetune':
        subset = normalize_subset_name(args.subset)
        # читаем конфиг/профиль только ради возможных глобальных параметров,
        # но базовые размеры берём из base_run при построении cfg
        defaults = TrainConfig(
            data_root=args.data_root or "data/handwritten",
            run_dir=args.new_run,
        )
        cfg, raw_cfg = build_effective_config(args, defaults)
        cfg.run_dir = args.new_run

        os.makedirs(cfg.run_dir, exist_ok=True)
        with open(os.path.join(cfg.run_dir, "effective_config.json"), "w", encoding="utf-8") as f:
            json.dump(raw_cfg, f, ensure_ascii=False, indent=2)

        finetune_entry(
            base_run=args.base_run,
            new_run=cfg.run_dir,
            data_root=cfg.data_root,
            subset=subset,
            weights=args.weights,
            epochs=cfg.epochs,
            steps_per_epoch=cfg.steps_per_epoch,
            batch_size=cfg.batch_size,
            img_h=cfg.img_h, img_w=cfg.img_w, max_text_len=cfg.max_text_len,
            lr=cfg.learning_rate,
            freeze_encoder=getattr(args, 'freeze_encoder', False),
            enc_dropout=cfg.enc_dropout, dec_dropout=cfg.dec_dropout,
            label_smoothing=cfg.label_smoothing,
            mixed_precision=cfg.mixed_precision,
            monitor_metric=cfg.monitor_metric,
            lr_decay_rate=cfg.lr_decay_rate,
            early_stop_patience=cfg.early_stop_patience,
            val_batches_for_eval=cfg.val_batches_for_eval,
            seed=cfg.seed,
        )
    else:
        raise ValueError("Unknown command")

if __name__ == '__main__':
    main()

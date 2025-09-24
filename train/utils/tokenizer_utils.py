# coding: utf-8
"""
utils/tokenizer_utils.py

Функции по загрузке/созданию/тренировке токенайзера и TrOCRProcessor.
Используем ViTImageProcessor (новый класс) вместо устаревшего ViTFeatureExtractor.
Все сообщения и комментарии — на русском.
"""
import os
import logging
from typing import Optional, List, Tuple, Dict

import config  # type: ignore
from transformers import PreTrainedTokenizerFast, ViTImageProcessor, TrOCRProcessor
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)


def train_bpe_tokenizer(corpus_paths: List[str],
                        vocab_size: int = 8000,
                        special_tokens: Optional[List[str]] = None) -> Tokenizer:
    """Обучить BPE токенайзер (tokenizers.Tokenizer)."""
    if special_tokens is None:
        special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS")
    if not corpus_paths:
        raise ValueError("train_bpe_tokenizer: corpus_paths пуст.")
    tokenizer_obj = Tokenizer(models.BPE(unk_token=special_tokens[1]))
    tokenizer_obj.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer_obj.train(files=corpus_paths, trainer=trainer)
    tokenizer_obj.decoder = decoders.ByteLevel()
    logger.info("BPE tokenizer обучен (vocab_size=%d).", vocab_size)
    return tokenizer_obj


def save_processor_from_tokenizer(tokenizer_obj: Tokenizer,
                                  out_dir: str,
                                  special_tokens: Optional[List[str]] = None,
                                  image_size: Optional[Tuple[int, int]] = None) -> TrOCRProcessor:
    """
    Создать PreTrainedTokenizerFast + ViTImageProcessor и собрать TrOCRProcessor.
    Сохранить в out_dir и вернуть объект.
    """
    if special_tokens is None:
        special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS")
    os.makedirs(out_dir, exist_ok=True)
    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer_obj.save(tok_json_path)
    logger.info("Сохранён tokenizers JSON: %s", tok_json_path)

    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json_path,
                                       unk_token=special_tokens[1], pad_token=special_tokens[0],
                                       bos_token=special_tokens[2], eos_token=special_tokens[3])

    size = image_size or getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))
    # Используем ViTImageProcessor — новый класс, вместо устаревшего ViTFeatureExtractor
    feat = ViTImageProcessor(do_resize=True, size=size, do_normalize=True)
    try:
        feat.save_pretrained(out_dir)
    except Exception:
        logger.debug("Не удалось сохранить ViTImageProcessor (non-fatal).")

    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(out_dir)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal).")
    logger.info("Processor сохранён в %s", out_dir)
    return processor


def build_char_level_tokenizer_and_processor(out_dir: str) -> TrOCRProcessor:
    """
    Собрать простой char-level токенайзер и TrOCRProcessor (fallback).
    """
    os.makedirs(out_dir, exist_ok=True)
    special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])

    modern_upper = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    chars = []
    for ch in modern_upper:
        chars.append(ch)
        chars.append(ch.lower())
    digits = [str(d) for d in range(10)]
    punctuation = list(".,:;!?\"'()[]{}-–—«»/\\@#%&*№•")
    all_chars = list(dict.fromkeys(chars + digits + punctuation + [" "]))

    vocab: Dict[str, int] = {}
    idx = 0
    for t in special_tokens:
        vocab[t] = idx
        idx += 1
    for ch in all_chars:
        if ch in vocab:
            continue
        vocab[ch] = idx
        idx += 1

    model_tok = models.WordLevel(vocab=vocab, unk_token=special_tokens[1])
    tokenizer_obj = Tokenizer(model_tok)
    tokenizer_obj.normalizer = normalizers.Sequence([normalizers.NFC()])
    try:
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Split(pattern=r"", behavior="isolated")
    except Exception:
        tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer_obj.decoder = decoders.WordLevel()
    tokenizer_obj.post_processor = TemplateProcessing(
        single=f"{special_tokens[2]} $A {special_tokens[3]}",
        pair=f"{special_tokens[2]} $A {special_tokens[3]} {special_tokens[2]} $B {special_tokens[3]}",
        special_tokens=[(special_tokens[2], vocab[special_tokens[2]]), (special_tokens[3], vocab[special_tokens[3]])]
    )

    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer_obj.save(tok_json_path)
    logger.info("Сохранён char-level tokenizer.json: %s", tok_json_path)

    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json_path,
                                       unk_token=special_tokens[1], pad_token=special_tokens[0],
                                       bos_token=special_tokens[2], eos_token=special_tokens[3])
    feat = ViTImageProcessor(do_resize=True, size=getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384)), do_normalize=True)
    try:
        feat.save_pretrained(out_dir)
    except Exception:
        logger.debug("Не удалось сохранить ViTImageProcessor (non-fatal).")

    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(out_dir)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal).")
    logger.info("Собран char-level TrOCRProcessor в %s", out_dir)
    return processor


def load_processor_safe(path: str) -> TrOCRProcessor:
    """
    Попытаться безопасно загрузить TrOCRProcessor.
    Используем ViTImageProcessor если возможно (вместо ViTFeatureExtractor).
    """
    # Сначала пробуем напрямую
    try:
        proc = TrOCRProcessor.from_pretrained(path, local_files_only=True)
        logger.info("TrOCRProcessor загружен из '%s' (local).", path)
        return proc
    except Exception as e:
        logger.debug("TrOCRProcessor.from_pretrained(local) failed: %s", e)

    tok_fast = None
    tok_json = os.path.join(path, "tokenizer.json")

    # Попытка загрузить быстрый токенайзер
    try:
        tok_fast = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
        logger.info("PreTrainedTokenizerFast загружен из '%s' (local).", path)
    except Exception:
        logger.debug("PreTrainedTokenizerFast.from_pretrained failed (non-fatal).")
        if os.path.exists(tok_json):
            try:
                special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
                tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json,
                                                   unk_token=special_tokens[1], pad_token=special_tokens[0],
                                                   bos_token=special_tokens[2], eos_token=special_tokens[3])
                logger.info("PreTrainedTokenizerFast создан из tokenizer.json: %s", tok_json)
            except Exception as e2:
                logger.warning("PreTrainedTokenizerFast(tokenizer_file=...) failed: %s", e2)
        else:
            logger.info("tokenizer.json не найден в %s", path)

    # Попытка загрузить современный image processor
    feat = None
    try:
        feat = ViTImageProcessor.from_pretrained(path, local_files_only=True)
        logger.info("ViTImageProcessor загружен из %s", path)
    except Exception:
        logger.debug("ViTImageProcessor.from_pretrained не сработал — создаём вручную.")
        feat = ViTImageProcessor(do_resize=True, size=getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384)), do_normalize=True)
        try:
            feat.save_pretrained(path)
        except Exception:
            logger.debug("Не удалось сохранить ViTImageProcessor (non-fatal).")

    if tok_fast is None:
        logger.warning("Токенайзер не найден в %s — создаём CHAR-LEVEL токенайзер (fallback).", path)
        try:
            processor = build_char_level_tokenizer_and_processor(path)
            return processor
        except Exception as e:
            logger.exception("Автоматическое создание токенайзера не удалось: %s", e)
            raise RuntimeError(f"Не удалось загрузить или создать токенайзер в '{path}'. Причина: {e}")

    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(path)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal).")
    logger.info("TrOCRProcessor собран из токенайзера и ViTImageProcessor.")
    return processor

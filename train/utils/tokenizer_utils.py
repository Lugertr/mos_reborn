"""
tokenizer_utils.py — токенайзер/processor: загрузка, тренировка BPE, char-fallback.
"""

import os
import logging
from typing import List, Optional, Tuple, Dict, Any

import config  # type: ignore
from transformers import TrOCRProcessor, ViTFeatureExtractor, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
    os.makedirs(out_dir, exist_ok=True)
    special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
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
    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json_path,
                                       unk_token=special_tokens[1], pad_token=special_tokens[0],
                                       bos_token=special_tokens[2], eos_token=special_tokens[3])
    try:
        tok_fast.save_pretrained(out_dir)
    except Exception:
        logger.debug("tok_fast.save_pretrained failed (non-fatal)")

    feat = ViTFeatureExtractor(do_resize=True,
                               size=getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384)),
                               do_normalize=True)
    try:
        feat.save_pretrained(out_dir)
    except Exception:
        logger.debug("feat.save_pretrained failed (non-fatal)")

    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(out_dir)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal)")

    logger.info("Собран fallback TrOCRProcessor в %s", out_dir)
    return processor


def load_processor_safe(path: str) -> TrOCRProcessor:
    try:
        proc = TrOCRProcessor.from_pretrained(path, local_files_only=True)
        logger.info("TrOCRProcessor загружен из '%s' (local).", path)
        return proc
    except Exception as e:
        logger.warning("TrOCRProcessor.from_pretrained(local_files_only=True) failed: %s", e)

    tok_fast = None
    tok_json = os.path.join(path, "tokenizer.json")
    try:
        tok_fast = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
        logger.info("PreTrainedTokenizerFast загружен из '%s' (local).", path)
    except Exception as e:
        logger.debug("PreTrainedTokenizerFast.from_pretrained failed: %s", e)
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

    feat = None
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

    if tok_fast is None:
        logger.warning("Токенайзер не найден в %s — создаём CHAR-LEVEL токенайзер (fallback).", path)
        return build_char_level_tokenizer_and_processor(path)

    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(path)
    except Exception:
        logger.debug("processor.save_pretrained failed (non-fatal)")
    logger.info("TrOCRProcessor собран из токенайзера и feature_extractor.")
    return processor


def train_bpe_tokenizer(corpus_paths: List[str],
                        vocab_size: int = 8000,
                        special_tokens: Optional[List[str]] = None) -> Tokenizer:
    if not corpus_paths:
        raise ValueError("train_bpe_tokenizer: corpus_paths пуст.")
    if special_tokens is None:
        special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])

    logger.info("Тренируем BPE токенайзер: vocab_size=%d, files=%s", vocab_size, corpus_paths)
    tokenizer_obj = Tokenizer(models.BPE(unk_token=special_tokens[1]))
    tokenizer_obj.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer_obj.train(files=corpus_paths, trainer=trainer)
    tokenizer_obj.decoder = decoders.ByteLevel()
    logger.info("BPE tokenizer обучен (vocab_size ~ %d)", vocab_size)
    return tokenizer_obj


def save_processor_from_tokenizer(tokenizer_obj: Tokenizer,
                                  out_dir: str,
                                  special_tokens: Optional[List[str]] = None,
                                  image_size: Optional[Tuple[int, int]] = None):
    if special_tokens is None:
        special_tokens = getattr(config, "TOKENIZER_SPECIAL_TOKENS", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    os.makedirs(out_dir, exist_ok=True)
    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer_obj.save(tok_json_path)
    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json_path,
                                       unk_token=special_tokens[1], pad_token=special_tokens[0],
                                       bos_token=special_tokens[2], eos_token=special_tokens[3])
    size = image_size or getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384))
    feat = ViTFeatureExtractor(do_resize=True, size=size, do_normalize=True)
    try:
        feat.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("Не удалось сохранить feature_extractor конфиг: %s", e)
    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)
    try:
        processor.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("processor.save_pretrained failed: %s", e)
    try:
        tok_fast.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("tok_fast.save_pretrained failed: %s", e)
    logger.info("Processor сохранён в %s", out_dir)
    return processor


def preprocess_example_np(example: Dict[str, Any], images_dir: str, processor) -> Dict[str, Any]:
    """
    Подготовка одного примера в numpy: чтение изображения, feature_extractor (processor) -> pixel_values (C,H,W) и labels.
    Возвращает dict с keys pixel_values (np.float32) и labels (list[int]).
    """
    file_name = example.get("file_name") or example.get("image")
    if file_name is None:
        raise KeyError("preprocess_example_np: в примере нет поля file_name или image")
    img_path = file_name if os.path.isabs(file_name) else os.path.join(images_dir, file_name)

    img_path = os.path.normpath(os.path.abspath(img_path))
    images_dir_abs = os.path.normpath(os.path.abspath(images_dir))
    if not img_path.startswith(images_dir_abs):
        raise ValueError("Неверный путь к изображению (вне images_dir)")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"preprocess_example_np: изображение не найдено: {img_path}")

    with Image.open(img_path) as im:
        img = im.convert("RGB")
        out = processor(images=img, return_tensors="np")

    pv = out["pixel_values"][0]
    # Normalize to channel-first (C,H,W)
    if pv.ndim == 3:
        if pv.shape[0] in (1, 3):
            pass
        elif pv.shape[2] in (1, 3):
            pv = np.transpose(pv, (2, 0, 1))
        else:
            logger.debug("preprocess_example_np: unexpected pv shape %s (leaving as-is)", pv.shape)
    else:
        raise ValueError(f"preprocess_example_np: unexpected pixel_values ndim={pv.ndim}")

    labels = processor.tokenizer(example.get("text", ""), truncation=True, max_length=getattr(config, "GENERATION_MAX_LENGTH", 128)).input_ids
    return {"pixel_values": pv.astype(np.float32), "labels": labels}

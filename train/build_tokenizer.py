# build_tokenizer.py

import os
import argparse
from typing import List

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from transformers import PreTrainedTokenizerFast, TrOCRProcessor, ViTFeatureExtractor

def read_corpus_lines(paths: List[str]) -> List[str]:
    lines = []
    for p in paths:
        p = os.path.expanduser(p)
        if not os.path.exists(p):
            print(f"[WARN] не найден файл корпуса: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines

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

    all_chars = []
    for grp in (modern_chars, archaic_chars, digits, roman_upper, roman_lower):
        for ch in grp:
            if ch not in all_chars:
                all_chars.append(ch)
    return all_chars

def build_tokenizer(corpus_paths: List[str], vocab_size: int, out_dir: str):
    lines = read_corpus_lines(corpus_paths)
    if not lines:
        raise ValueError("Корпус пуст. Укажите пути к текстовым файлам.")

    additional_chars = make_additional_chars()
    print(f"Количество обязательных символов: {len(additional_chars)}")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([ normalizers.NFC() ])  # не понижать регистр чтобы сохранить разные буквы
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens + additional_chars,
        show_progress=True
    )

    tokenizer.train_from_iterator(lines, trainer=trainer)
    print("Токенизатор обучен.")

    tok_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]"
    )

    feat = ViTFeatureExtractor(do_resize=True, size=(384,384), do_normalize=True)
    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)

    os.makedirs(out_dir, exist_ok=True)
    processor.save_pretrained(out_dir)
    print(f"Processor сохранён в {out_dir}")

    return processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="+", required=True, help="пути к .txt файлам с корпусом")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--out_dir", type=str, default="./trocr-processor-old")
    args = parser.parse_args()

    build_tokenizer(args.corpora, vocab_size=args.vocab_size, out_dir=args.out_dir)

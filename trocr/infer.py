#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose
-------
Скрипт офлайн-инференса и анализа качества модели TrOCR (TensorFlow).
Два режима:
  • `recognize` — прогон датасета (train/test/postTest), сохранение TSV с (path, pred, ref),
                  логирование средних CER/WER.
  • `analyze`   — пост-анализ ранее сохранённого TSV и вывод сводного отчёта.

Key flow (recognize)
--------------------
1) Загрузка модели/charset/config из каталога run.
2) (Опционально) переопределение размеров/макс. длины/корня данных.
3) Инициализация графа и загрузка весов.
4) Чтение split’а из *.json, сбор tf.data.Dataset.
5) Greedy-декодирование батчами, расчёт CER/WER, сохранение TSV.

CLI
---
Пример:
    python -m trocr.infer recognize --run runs/base --weights best.weights.h5 --split postTest
    python -m trocr.infer analyze   --run runs/base --split postTest
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import logging
from typing import Optional, List, Tuple

import numpy as np
import tensorflow as tf

# ЛОГГЕР
logger = logging.getLogger("trocr.infer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Импортируем ядро из тренера
from trocr.trocr_modified import (
    Charset, TrainConfig, read_json_list, make_dataset,
    load_model_from_run, greedy_decode, cer, wer
)

def normalize_subset_name(name: str) -> str:
    """
    Приводит имя поднабора к каноническому виду.

    Поддержка синонимов:
      "tt" / "posttest" / "post_test" → "postTest";
      "postTest" → "postTest";
      "full" — разрешён только для обучения (не для инференса).

    Raises:
        ValueError: если передано неразрешённое значение для инференса.
    """
    low = name.lower()
    if low in ("tt", "posttest", "post_test"):
        return "postTest"
    if name == "full":
        return "full"
    if name == "postTest":
        return "postTest"
    raise ValueError("split must be 'train'/'test'/'postTest' (или 'full' только для обучения)")

def save_predictions_tsv(out_path: str, rows: List[Tuple[str, str, str]]):
    """
    Сохранить результат инференса в TSV-таблицу.

    Args:
        out_path: Путь к TSV.
        rows: Список кортежей (path, pred, ref) построчно.

    Notes:
        Внутренние табы/переводы строк в pred/ref заменяются, чтобы не ломать формат TSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("path\tpred\tref\n")
        for p, pred, ref in rows:
            pred = pred.replace("\t", " ").replace("\n", " ")
            ref  = ref.replace("\t", " ").replace("\n", " ")
            f.write(f"{p}\t{pred}\t{ref}\n")
    logger.info(f"Saved predictions to {out_path}")

def recognize_entry(run: str, split: str, weights: str, out: Optional[str] = None,
                    batch_size: int = 32,
                    img_h: Optional[int] = None, img_w: Optional[int] = None,
                    max_text_len: Optional[int] = None,
                    data_root: Optional[str] = None):
    """
    Прогон датасета (postTest/test/train), сохранение TSV с (path, pred, ref).

    Args:
        run: Каталог запуска с config.json/charset.json/весами.
        split: Какой поднабор использовать: 'postTest' | 'test' | 'train'.
        weights: Имя/путь файла весов (.h5). Относительный путь считается относительно `run`.
        out: Куда писать TSV (по умолчанию `<run>/predictions.tsv`).
        batch_size: Размер батча на инференсе.
        img_h, img_w, max_text_len, data_root: Переопределения из config, при необходимости.

    Behavior:
        • Сначала грузится модель+конфиг из `run`, при необходимости переопределяются поля.
        • Строится граф (прогон dummy), затем подгружаются веса.
        • Собирается tf.data.Dataset для нужного split и выполняется greedy-декод.
        • Подсчитываем средние CER/WER и логируем.
    """
    model, charset, cfg = load_model_from_run(run)
    # опционально перекрыть размеры
    if img_h: cfg.img_h = img_h
    if img_w: cfg.img_w = img_w
    if max_text_len: cfg.max_text_len = max_text_len
    if data_root: cfg.data_root = data_root

    # построить модель (с теми же размерами)
    model = tf.function(lambda x: x)(model)  # no-op для явности
    dummy = {"image": tf.zeros([1, cfg.img_h, cfg.img_w, 1], tf.float32),
            "decoder_input": tf.fill([1, cfg.max_text_len], tf.cast(charset.sos_id, tf.int32))}
    _ = model(dummy, training=False)

    # веса
    wpath = weights if os.path.isabs(weights) else os.path.join(run, weights)
    model.load_weights(wpath)
    logger.info(f"Loaded weights from {wpath}")

    # выбор split
    split_n = normalize_subset_name(split)
    root = cfg.data_root
    if split_n == 'postTest':
        json_path = os.path.join(root, 'postTest.json'); images_dir = os.path.join(root, 'postTest')
    elif split_n == 'full':
        raise ValueError("Для инференса используйте 'train' или 'test' или 'postTest', не 'full'")
    else:
        # поддержка 'train' / 'test'
        json_path = os.path.join(root, f'{split_n}.json'); images_dir = os.path.join(root, f'{split_n}')
    samples = read_json_list(json_path, images_dir)

    ds = make_dataset(samples, charset, cfg.img_h, cfg.img_w, cfg.max_text_len,
                    batch_size, shuffle=False, repeat=False, shard_desc='infer', include_meta=True, augment=False)

    rows: List[Tuple[str, str, str]] = []
    total_cer = 0.0
    total_wer = 0.0
    n = 0

    for batch in ds:
        preds = greedy_decode(model, batch["image"], batch.get("enc_key_mask"), charset, cfg.max_text_len)
        refs = [x.decode('utf-8') for x in batch['text'].numpy().tolist()]
        paths = [p.decode('utf-8') for p in batch['path'].numpy().tolist()]
        for i in range(len(preds)):
            pred = preds[i]
            ref  = refs[i]
            path = paths[i]
            rows.append((path, pred, ref))
            total_cer += cer(ref, pred)
            total_wer += wer(ref, pred)
            n += 1

    run_dir = run if os.path.isabs(run) else os.path.join("runs", run) if not run.startswith("runs") else run
    if out is None:
        out = os.path.join(run, "predictions.tsv")
    else:
        if not os.path.isabs(out):
            out = os.path.join(run, out)
    save_predictions_tsv(out, rows)

    if n > 0:
        logger.info(f"AVG CER: {total_cer / n:.4f}")
        logger.info(f"AVG WER: {total_wer / n:.4f}")

def analyze_entry(run: str, split: str, in_path: Optional[str], out_prefix: Optional[str]):
    """
    Пост-анализ готового predictions.tsv: построчные CER/WER и сводка.

    Args:
        run: Каталог запуска (куда положим отчёт).
        split: Для имени отчёта (чисто информативно).
        in_path: Путь к TSV; по умолчанию `<run>/predictions.tsv`.
        out_prefix: Префикс имени отчёта; по умолчанию `analysis_<split>`.

    Output:
        `<run>/<out_prefix>.txt` — список строк с (path|pred|ref|cer|wer) и средние значения в конце.
    """
    if not os.path.isabs(run):
        run = run
    if in_path is None:
        in_path = os.path.join(run, "predictions.tsv")
    if out_prefix is None:
        out_prefix = f"analysis_{split}"

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Не найден файл предсказаний: {in_path}")

    lines = []
    total_cer = 0.0
    total_wer = 0.0
    n = 0

    with open(in_path, "r", encoding="utf-8") as f:
        header = f.readline()  # path\tpred\tref
        for line in f:
            line = line.rstrip("\n")
            try:
                path, pred, ref = line.split("\t", 2)
            except ValueError:
                # пропустить некорректные строки
                continue
            _cer = cer(ref, pred)
            _wer = wer(ref, pred)
            total_cer += _cer
            total_wer += _wer
            n += 1
            lines.append(f"{path} | pred: \"{pred}\" | ref: \"{ref}\" | cer: {_cer:.4f} | wer: {_wer:.4f}")

    avg_cer = total_cer / max(1, n)
    avg_wer = total_wer / max(1, n)
    lines.append("")
    lines.append(f"AVG CER: {avg_cer:.4f}")
    lines.append(f"AVG WER: {avg_wer:.4f}")

    out_txt = os.path.join(run, f"{out_prefix}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved analysis to {out_txt}")

def parse_args():
    """Парсер CLI для сабкоманд `recognize` и `analyze`."""
    p = argparse.ArgumentParser(description="TrOCR inference/analyze")
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('recognize')
    r.add_argument('--run', type=str, required=True, help="Каталог ранa (где лежат config.json и веса)")
    r.add_argument('--split', type=str, default='postTest', help="postTest/test/train")
    r.add_argument('--weights', type=str, default='best.weights.h5')
    r.add_argument('--out', type=str, default=None, help="Относительно runs/<run> или абсолютный путь к TSV")
    r.add_argument('--batch-size', type=int, default=32)
    r.add_argument('--img-h', type=int, default=None)
    r.add_argument('--img-w', type=int, default=None)
    r.add_argument('--max-text-len', type=int, default=None)
    r.add_argument('--data-root', type=str, default=None)

    a = sub.add_parser('analyze')
    a.add_argument('--run', type=str, required=True)
    a.add_argument('--split', type=str, default='postTest')
    a.add_argument('--in', dest='in_path', type=str, default=None, help="Путь к predictions.tsv (по умолчанию runs/<run>/predictions.tsv)")
    a.add_argument('--out-prefix', type=str, default=None, help="Префикс имени отчёта (по умолчанию analysis_<split>)")

    return p.parse_args()

def main():
    """Точка входа CLI: делегирует в recognize/analyze по аргументу `cmd`."""
    args = parse_args()
    if args.cmd == 'recognize':
        recognize_entry(
            run=args.run, split=args.split, weights=args.weights, out=args.out,
            batch_size=args.batch_size, img_h=args.img_h, img_w=args.img_w,
            max_text_len=args.max_text_len, data_root=args.data_root
        )
    elif args.cmd == 'analyze':
        analyze_entry(run=args.run, split=args.split, in_path=args.in_path, out_prefix=args.out_prefix)
    else:
        raise ValueError("Unknown command")

if __name__ == '__main__':
    main()

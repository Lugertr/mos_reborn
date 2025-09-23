# train_check.py
"""
Короткие тесты перед обучением:
- проверка путей данных и изображений
- проверка, что processor загружается и обрабатывает пример
- проверка (опционально) загрузки модели, если она сохранена в TOKENIZER_OUT_DIR
"""

import os
import sys
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
import config  # type: ignore

from PIL import Image
from datasets import load_dataset
import tensorflow as tf
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from train_utils import get_device

def test_paths():
    print("== Проверка путей данных ==")
    train_json = os.path.join(config.DATA_DIR, "train.json")
    test_json = os.path.join(config.DATA_DIR, "test.json")
    images_dir = os.path.join(config.DATA_DIR, config.IMAGES_SUBDIR)
    ok = True
    if not os.path.isfile(train_json):
        print("Ошибка: train.json не найден:", train_json)
        ok = False
    if not os.path.isfile(test_json):
        print("Ошибка: test.json не найден:", test_json)
        ok = False
    if not os.path.isdir(images_dir):
        print("Ошибка: images директория не найдена:", images_dir)
        ok = False
    if not ok:
        sys.exit(1)
    print("Пути данных и изображений в порядке.")

def test_processor_and_model():
    print("== Проверка processor и (опционально) модели ==")
    token_dir = config.TOKENIZER_OUT_DIR
    if not os.path.isdir(token_dir):
        print("Предупреждение: TOKENIZER_OUT_DIR не найден:", token_dir)
        print("Запустите build_tokenizer.py чтобы создать processor.")
    try:
        processor = TrOCRProcessor.from_pretrained(token_dir)
        print("Processor загружен:", token_dir)
    except Exception as e:
        print("Не удалось загрузить processor:", e)
        sys.exit(1)

    ds = load_dataset("json", data_files={"test": os.path.join(config.DATA_DIR, "test.json")}, split="test")
    if len(ds) == 0:
        print("test.json пустой.")
        sys.exit(1)
    ex = ds[0]
    fname = ex.get("file_name") or ex.get("image")
    if not fname:
        print("В первом примере нет file_name/image.")
        sys.exit(1)
    img_path = fname if os.path.isabs(fname) else os.path.join(config.DATA_DIR, config.IMAGES_SUBDIR, fname)
    if not os.path.exists(img_path):
        print("Изображение не найдено:", img_path)
        sys.exit(1)
    img = Image.open(img_path).convert("RGB")
    try:
        out = processor(images=img, return_tensors="tf")
        print("Processor обработал изображение; pixel_values shape:", out["pixel_values"].shape)
    except Exception as e:
        print("Ошибка при processor(images=...):", e)
        sys.exit(1)

    try:
        model = VisionEncoderDecoderModel.from_pretrained(token_dir, from_tf=True)
        print("Модель загружена из", token_dir)
        try:
            pv = out["pixel_values"]
            gen = model.generate(pv, max_length=config.GENERATION_MAX_LENGTH, num_beams=1)
            decoded = processor.batch_decode(gen, skip_special_tokens=True)
            print("Пример генерации:", decoded[0] if decoded else "<пусто>")
        except Exception as e:
            print("Генерация из модели не сработала (не fatal):", e)
    except Exception:
        print("TF-модель не найдена в TOKENIZER_OUT_DIR — это нормально, если вы тренируете с нуля.")
    print("Проверка завершена успешно.")

if __name__ == "__main__":
    print("Устройство:", get_device())
    test_paths()
    test_processor_and_model()

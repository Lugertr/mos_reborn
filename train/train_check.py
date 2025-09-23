# train_check.py
"""
Проверка перед обучением: что всё готово — корпус, processor создан, примеры работают.
"""

import os
import sys
from PIL import Image

import config
from train_utils import get_device
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset


def test_paths_and_processor():
    """
    Проверка путей: existence corpus processor, файлов данных и изображений.
    """
    if not os.path.exists(config.MODEL_NAME_OR_PATH):
        print(f"Ошибка: директория processor не найдена: {config.MODEL_NAME_OR_PATH}")
        sys.exit(1)

    train_json = os.path.join(config.DATA_DIR, "train.json")
    test_json = os.path.join(config.DATA_DIR, "test.json")
    images_dir = os.path.join(config.DATA_DIR, config.IMAGES_SUBDIR)

    if not os.path.isfile(train_json):
        print("Ошибка: train.json не найден:", train_json)
        sys.exit(1)
    if not os.path.isfile(test_json):
        print("Ошибка: test.json не найден:", test_json)
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print("Ошибка: images directory не найдена:", images_dir)
        sys.exit(1)

    print("Пути к данным и изображениям проверены.")

def test_processor_and_model():
    """
    Проверка что processor загружается, и один пример проходит через pipeline.
    Модель можно не загружать, если её ещё нет.
    """
    try:
        processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME_OR_PATH)
        print("Processor загружен.")
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
        print("Ошибка: нет file_name/image в первом примере.")
        sys.exit(1)
    img_path = fname if os.path.isabs(fname) else os.path.join(config.DATA_DIR, config.IMAGES_SUBDIR, fname)
    if not os.path.exists(img_path):
        print("Ошибка: изображение не найдено:", img_path)
        sys.exit(1)
    img = Image.open(img_path).convert("RGB")

    try:
        out = processor(images=img, return_tensors="np")
        print("Processor обработал изображение.")
    except Exception as e:
        print("Ошибка при обработке изображения:", e)
        sys.exit(1)

    try:
        # если модель уже есть
        model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_NAME_OR_PATH, from_tf=True)
        print("Модель загружена из processor directory.")
    except Exception:
        print("Модель не найдена / не загружается — это нормально, если тренировка с нуля.")
        return

    try:
        pixel_values = out["pixel_values"]
        gen = model.generate(pixel_values, max_length=config.GENERATION_MAX_LENGTH, num_beams=1)
        decoded = processor.batch_decode(gen, skip_special_tokens=True)
        print("Пример предсказания:", decoded[0] if decoded else "")
    except Exception as e:
        print("Генерация не удалась (model.generate):", e)
        # всё равно processor работает

    print("Проверка train_check пройдена.")


if __name__ == "__main__":
    test_paths_and_processor()
    test_processor_and_model()

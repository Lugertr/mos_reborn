from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import os

checkpoint_path = "./trocr-finetuned/checkpoint-100"
output_dir = "./trocr-finetuned-final"

# Загружаем модель из чекпоинта
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)

# Процессор берём из исходной модели (замени при необходимости!)
processor = TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")

# Сохраняем финальный вариант
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print(f"✅ Финальная модель сохранена в {output_dir}")

from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
import os
from PIL import Image

# 1. Загружаем локальную модель и процессор
model_name = "./trocr-base-handwritten-ru"  
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# 2. Загружаем локальный датасет
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/handwritten/train.json",
        "test": "data/handwritten/test.json"
    }
)

# ⚡ Для отладки можно ограничить подмножество
dataset["train"] = dataset["train"].select(range(200))
dataset["test"] = dataset["test"].select(range(50))

# 3. Препроцессинг
def preprocess(batch):
    images = [Image.open(f"data/handwritten/images/{x}").convert("RGB") for x in batch["file_name"]]
    pixel_values = processor(images=images, return_tensors="pt", padding=True).pixel_values
    labels = processor.tokenizer(batch["text"], padding="max_length", truncation=True).input_ids
    labels = [[l if l != processor.tokenizer.pad_token_id else -100 for l in label] for label in labels]
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# 4. Аргументы тренировки
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",         # 🔹 все промежуточные чекпоинты сюда
    per_device_train_batch_size=1,      # ↓ минимальный батч (экономия VRAM)
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,      # имитация батча = 4
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    num_train_epochs=3,                 # начнём с 3, потом можно поднять
    save_strategy="steps",              # сохраняем каждые N шагов
    save_steps=200,
    save_total_limit=5,                 # храним только последние 5
    eval_strategy="steps",              # проверяем каждые N шагов
    eval_steps=200,
    fp16=True,                          # половинная точность (экономит VRAM)
    resume_from_checkpoint=True         # если процесс упадёт → продолжит
)

# 5. Тренер
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

# 6. Запуск
trainer.train()

# 7. Сохраняем итоговую модель отдельно
final_dir = "./trocr-finetuned"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)

print(f"✅ Обучение завершено. Итоговая модель сохранена в {final_dir}, чекпоинты в ./checkpoints")

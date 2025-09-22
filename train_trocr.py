# train_trocr.py
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from PIL import Image
import torch, os, re

# ========= 0. Утилиты =========

def find_latest_checkpoint(base_dir: str) -> str | None:
    """
    Находит максимальный ./checkpoints/checkpoint-XXXX.
    Вернёт путь к последнему чекпоинту или None, если их нет.
    """
    if not os.path.isdir(base_dir):
        return None
    best_num, best_path = -1, None
    for name in os.listdir(base_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m:
            n = int(m.group(1))
            if n > best_num:
                best_num = n
                best_path = os.path.join(base_dir, name)
    return best_path

def simple_wer(ref: str, hyp: str) -> float:
    """
    Простейший WER без внешних библиотек.
    """
    r = ref.split()
    h = hyp.split()
    # Левенштейн по словам
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return dp[len(r)][len(h)] / len(r)


# ========= 1. Модель/процессор =========

model_name = "./trocr-base-handwritten-ru"   # твоя локальная базовая модель
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# CPU-only? (у тебя torch 2.8.0+cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ========= 2. Датасет =========

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/handwritten/train.json",
        "test":  "data/handwritten/test.json"
    }
)

# Для отладки используем небольшой поднабор
dataset["train"] = dataset["train"].select(range(200))
dataset["test"]  = dataset["test"].select(range(50))

# ========= 3. Препроцессинг =========

def preprocess(batch):
    images = [Image.open(f"data/handwritten/images/{x}").convert("RGB") for x in batch["file_name"]]
    pixel_values = processor(images=images, return_tensors="pt", padding=True).pixel_values
    # labels как ids; пад-ид заменяем на -100, чтобы не учитывался в loss
    labels = processor.tokenizer(batch["text"], padding="max_length", truncation=True).input_ids
    labels = [[(l if l != processor.tokenizer.pad_token_id else -100) for l in lab] for lab in labels]
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# ========= 4. Аргументы тренировки =========

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,   # имитация батча = 4
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=20,                # почаще логируем лосс
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=5,
    eval_strategy="steps",
    eval_steps=200,
    fp16=False,                      # у тебя CPU, fp16 отключаем
    remove_unused_columns=False,     # важно для seq2seq + vision
)

# ========= 5. Метрики =========

def postprocess_text(pred_ids, label_ids):
    # распаковываем, убираем токены
    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = [[l for l in ids if l != -100] for ids in label_ids]
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return pred_str, label_str

def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    pred_str, label_str = postprocess_text(pred_ids, label_ids)
    wers = [simple_wer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    # средний WER
    return {"wer": sum(wers)/len(wers) if wers else 0.0}

# ========= 6. Тренер =========

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=processor,      # вместо tokenizer (устаревший)
    compute_metrics=compute_metrics,
)

# ========= 7. Запуск тренировки (возобновление, если есть чекпоинт) =========

latest_ckpt = find_latest_checkpoint("./checkpoints")
if latest_ckpt:
    print(f"🔄 Возобновляем обучение с {latest_ckpt}")
    trainer.train(resume_from_checkpoint=latest_ckpt)
else:
    print("🚀 Старт обучения с нуля")
    trainer.train()

# ========= 8. Итоговая сохранённая модель =========

final_dir = "./trocr-finetuned"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)
print(f"✅ Обучение завершено. Итоговая модель сохранена в {final_dir}, чекпоинты в ./checkpoints")

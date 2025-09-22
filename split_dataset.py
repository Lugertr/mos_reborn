# split_dataset.py
import json, random, os

labels_file = "data/handwritten/labels.json"

with open(labels_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# превращаем словарь в список словарей
dataset = [{"file_name": k, "text": v} for k, v in data.items()]

print(f"Всего записей: {len(dataset)}")

# перемешиваем
random.shuffle(dataset)

# делим 95% train / 5% test
split_idx = int(len(dataset) * 0.95)
train, test = dataset[:split_idx], dataset[split_idx:]

# сохраняем
with open("data/handwritten/train.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open("data/handwritten/test.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train)}, Test: {len(test)}")

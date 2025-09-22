# test_preproc.py
from preproc import preprocess_for_print, preprocess_for_hand
from PIL import Image
import os

# Пути к тестовым картинкам
test_images = [
    "/root/ocr_project/test.png",       # печатный текст
    "/root/ocr_project/test_hand4.png"   # рукописный текст
]

os.makedirs("preproc_results", exist_ok=True)

for img_path in test_images:
    img = Image.open(img_path).convert("RGB")

    # Печатный
    processed_print = preprocess_for_print(img)
    processed_print.save(f"preproc_results/{os.path.basename(img_path)}_print.png")

    # Рукописный
    processed_hand = preprocess_for_hand(img)
    processed_hand.save(f"preproc_results/{os.path.basename(img_path)}_hand.png")

print("✅ Предобработанные изображения сохранены в папке preproc_results")

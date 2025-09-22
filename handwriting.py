from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance
import torch

# Загружаем процессор и модель (локальная папка с весами)
processor = TrOCRProcessor.from_pretrained("./trocr-base-handwritten-ru")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-base-handwritten-ru")

def preprocess_handwritten(img: Image.Image) -> Image.Image:
    """
    Предобработка рукописного изображения:
    - ч/б
    - увеличение контраста
    - бинаризация
    - ресайз до  384x384
    """
    # Ч/б
    img = img.convert("L")

    # Усиление контраста
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Бинаризация (порог 128)
    img = img.point(lambda x: 0 if x < 128 else 255, "1")

    # Обратно в RGB для модели
    img = img.convert("RGB")

    # Ресайз под модель
    img = ImageOps.pad(img, (384, 384), color="white")

    return img

def recognize_handwriting(pil_img: Image.Image) -> str:
    """
    Распознавание рукописи через TrOCR.
    """
    try:
        img = preprocess_handwritten(pil_img)

        pixel_values = processor(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        return f"Ошибка распознавания: {str(e)}"

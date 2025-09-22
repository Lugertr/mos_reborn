# preproc.py
import cv2
import numpy as np
from PIL import Image


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Выравнивание перекоса текста с защитой от  переворота."""
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ограничим угол: не больше ±max_angle
    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return image  # считаем, что перекоса нет

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_for_print(pil_img: Image.Image) -> Image.Image:
    """Предобработка для печатного текста (Tesseract)."""
    img = np.array(pil_img.convert("L"))  # серое изображение

    # Усиление локального контраста (лучше, чем equalizeHist на пожелтевших страницах)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Убираем шумы
    img = cv2.medianBlur(img, 3)

    # Выравнивание перекоса
    img = deskew(img)

    # Жёсткая бинаризация (чёткие буквы для Tesseract)
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return Image.fromarray(img)


def preprocess_for_hand(pil_img: Image.Image) -> Image.Image:
    """Предобработка для рукописного текста (TrOCR)."""
    img = np.array(pil_img.convert("L"))

    # Сглаживаем шумы, но оставляем полутона
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Локальное выравнивание контраста (важно для слабых чернил)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # НЕ бинаризуем → TrOCR лучше работает с оттенками
    return Image.fromarray(img)

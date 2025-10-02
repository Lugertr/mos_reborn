"""
Purpose
-------
Предобработка изображений для печатного текста (ветка Tesseract).

Two presets
-----------
- `preprocess_for_print_soft`: мягкая нормализация (CLAHE + median blur + deskew).
- `preprocess_for_print_hard`: «жёсткая» версия (как soft, но с бинаризацией Otsu).

Why these steps
---------------
- CLAHE выравнивает освещённость/контраст старых сканов.
- Median blur (ядро 3) сглаживает точечный шум, почти не размывая границы.
- Deskew устраняет общий наклон страницы.
- Otsu (в hard) помогает на «грязных» фонах, делая текст контрастнее.
"""

import cv2
import numpy as np
from PIL import Image
from .deskew import deskew

def preprocess_for_print_soft(pil_img: Image.Image) -> Image.Image:
    """
    Мягкая предобработка без жёсткой пороговой бинаризации.

    Pipeline:
      1) Grayscale (L)
      2) CLAHE (clipLimit=3.0, tileGridSize=8×8)
      3) Median blur (k=3)
      4) Deskew

    Args:
        pil_img: Входное изображение (PIL.Image).

    Returns:
        PIL.Image (градации серого) после нормализации.
    """
    img = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    img = deskew(img)
    return Image.fromarray(img)

def preprocess_for_print_hard(pil_img: Image.Image) -> Image.Image:
    """
    Жёсткая предобработка: как soft, но с бинаризацией по Отцу.

    Pipeline:
      1) Grayscale (L)
      2) CLAHE (clipLimit=3.0, tileGridSize=8×8)
      3) Median blur (k=3)
      4) Deskew
      5) Otsu threshold (cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Args:
        pil_img: Входное изображение (PIL.Image).

    Returns:
        PIL.Image (бинарная) — под задачу печатного OCR на сложных фонах.
    """
    img = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    img = deskew(img)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(img)

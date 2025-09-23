import cv2
import numpy as np
from PIL import Image
import os

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Выравнивание перекоса текста с защитой от переворота."""
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_for_print_soft(pil_img: Image.Image, debug_path: str | None = None) -> Image.Image:
    """Щадящая предобработка для печатного текста (без жёсткой бинаризации)."""
    img = np.array(pil_img.convert("L"))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    img = deskew(img)

    result = Image.fromarray(img)
    if debug_path:
        result.save(debug_path)
    return result


def preprocess_for_print_hard(pil_img: Image.Image, debug_path: str | None = None) -> Image.Image:
    """Жёсткая предобработка для печатного текста (с бинаризацией)."""
    img = np.array(pil_img.convert("L"))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    img = deskew(img)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    result = Image.fromarray(img)
    if debug_path:
        result.save(debug_path)
    return result


def preprocess_for_hand(pil_img: Image.Image, debug_path: str | None = None) -> Image.Image:
    """Предобработка для рукописного текста (TrOCR)."""
    img = np.array(pil_img.convert("L"))

    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    result = Image.fromarray(img)
    if debug_path:
        result.save(debug_path)
    return result

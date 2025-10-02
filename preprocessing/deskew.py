"""
Purpose
-------
Автовыравнивание (deskew) сканов: оценка угла наклона текста и поворот изображения
в допустимых пределах.

How it works
------------
- Берём координаты ненулевых пикселей (`image > 0`) — подходит для серых/бинарных изображений.
- Через `cv2.minAreaRect` получаем угол наклона ограничивающего прямоугольника.
- Нормализуем угол в удобный диапазон, фильтруем «почти ноль» (<1°) и «слишком много» (>max_angle).
- Поворачиваем с интерполяцией `INTER_CUBIC` и заполнением краёв режимом `BORDER_REPLICATE`.

Notes
-----
Ожидается, что на вход подаётся одно каналное изображение (GRAY/BINARY) с текстом
на светлом фоне. Если нет значимых пикселей — возвращаем исходное изображение.
"""

import cv2
import numpy as np

def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Выравнивает наклон текста, если угол в разумных пределах.

    Args:
        image: Изображение в виде numpy-массива (серое/бинарное).
        max_angle: Максимально допустимый модуль угла поворота (в градусах).

    Returns:
        Повернутое изображение той же формы, либо исходное, если поворот не нужен/неуместен.
    """
    # Находим координаты «значимых» пикселей (текст/объекты)
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:
        return image

    # Оцениваем угол минимального прямоугольника, охватывающего точки
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV возвращает угол в диапазоне (-90, 0]; нормализуем:
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Игнорируем совсем маленький наклон и слишком большой (за пределами max_angle)
    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return image

    # Поворот вокруг центра изображения
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

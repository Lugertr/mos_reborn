"""
Purpose
-------
Автоматический поворот страницы по оценке ориентации (OSD) от Tesseract.

Details
-------
- Управляется флагом `TESS_OSD_ENABLE` из config.
- `pytesseract.image_to_osd` возвращает текстовый отчёт, из которого извлекаем "Rotate: <deg>".
- Если угол 90/180/270 (по модулю 360) — поворачиваем изображение.
- В отчёте Tesseract угол задаётся по часовой стрелке (CW), а PIL ожидает против (CCW),
  поэтому используем `-rot` и `expand=True`.

Fail-safe
---------
Любые ошибки OSD (нет tesseract, нераспознанный формат и т.д.) тихо игнорируются —
возвращаем исходное изображение.
"""

from typing import Tuple
from PIL import Image
from config import TESS_OSD_ENABLE
import pytesseract

def apply_osd_rotation(pil: Image.Image) -> Image.Image:
    """
    Применить поворот страницы на основе OSD Tesseract, если это включено.

    Args:
        pil: Входное изображение (PIL.Image) RGB/GRAY.

    Returns:
        PIL.Image: либо повернутое, либо исходное изображение (если поворот не требуется/ошибка).

    Notes:
        Tesseract сообщает угол поворота по часовой стрелке, PIL — против,
        поэтому фактически вращаем на `-rot`.
    """
    if not TESS_OSD_ENABLE:
        return pil
    try:
        osd = pytesseract.image_to_osd(pil)
        # Парсим строку 'Rotate: <deg>'
        rot = 0
        for line in osd.splitlines():
            if line.lower().startswith("rotate:"):
                rot = int(line.split(":")[1].strip())
                break
        if rot % 360 in (90, 180, 270):
            return pil.rotate(-rot, expand=True)  # tesseract даёт CW, PIL ждёт CCW
        return pil
    except Exception:
        # Любая ошибка OSD — не критично, возвращаем оригинал
        return pil

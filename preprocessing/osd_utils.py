# preprocessing/osd_utils.py
from typing import Tuple
from PIL import Image
import pytesseract

def _parse_osd_angle(osd_text: str) -> int:
    # Ищем строку вида "Rotate: 90"
    for line in osd_text.splitlines():
        line = line.strip()
        if line.lower().startswith("rotate:"):
            try:
                return int(line.split(":")[1].strip())
            except Exception:
                pass
    return 0

def apply_osd_rotation(pil_img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(pil_img)
        angle = _parse_osd_angle(osd)
        angle = angle % 360
        if angle in (0, 90, 180, 270):
            return pil_img.rotate(-angle, expand=True) if angle else pil_img
        return pil_img
    except Exception:
        # Если OSD не отработал — возвращаем как есть
        return pil_img

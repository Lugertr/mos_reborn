# preprocessing/osd_utils.py
from typing import Tuple
from PIL import Image
from config import TESS_OSD_ENABLE
import pytesseract

def apply_osd_rotation(pil: Image.Image) -> Image.Image:
    if not TESS_OSD_ENABLE:
        return pil
    try:
        osd = pytesseract.image_to_osd(pil)
        # парсим строку Rotate: <deg>
        rot = 0
        for line in osd.splitlines():
            if line.lower().startswith("rotate:"):
                rot = int(line.split(":")[1].strip())
                break
        if rot % 360 in (90, 180, 270):
            return pil.rotate(-rot, expand=True)  # tesseract даёт CW, PIL ждёт CCW
        return pil
    except Exception:
        return pil
# tesseract/segment_lines.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from PIL import Image
import pytesseract

def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

def segment(
    pil_img: Image.Image,
    langs: str = "rus+eng",
    psm: int = 6,
    level: int = 4,     # 4=line, 5=word
    oem: int = 1,
) -> List[Dict[str, Any]]:
    """
    Универсальный сегментатор через pytesseract.image_to_data.
    level==4: конструирует строки, агрегируя слова (level=5) по (block,par,line).
    level==5: возвращает слова как отдельные сегменты.
    """
    cfg = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(pil_img, lang=langs, output_type=pytesseract.Output.DICT, config=cfg)

    n = len(data["level"])
    segments: List[Dict[str, Any]] = []

    if level == 5:
        # Слова напрямую
        for i in range(n):
            if data["level"][i] != 5:
                continue
            text = (data["text"][i] or "").strip()
            conf = data["conf"][i]
            if not text or conf == "-1":
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            segments.append({
                "bbox": (x, y, x + w, y + h),
                "text": text,
                "avg_conf": float(conf),
                "words": 1,
            })
        # стабильно упорядочим сверху-вниз, слева-направо
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        return segments

    # level == 4: агрегируем слова по строкам
    # Соберём все words по ключу (block, par, line)
    groups: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    # Первым проходом соберём «каркасы» строк (bbox из level=4)
    for i in range(n):
        if data["level"][i] == 4:
            b = data["block_num"][i]
            p = data["par_num"][i]
            l = data["line_num"][i]
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            groups[(b, p, l)] = {
                "bbox": (x, y, x + w, y + h),
                "words": [],
                "confs": [],
            }

    # Вторым проходом добавим слова (level=5) в соответствующие группы
    for i in range(n):
        if data["level"][i] != 5:
            continue
        text = (data["text"][i] or "").strip()
        conf = data["conf"][i]
        if not text or conf == "-1":
            continue
        b = data["block_num"][i]
        p = data["par_num"][i]
        l = data["line_num"][i]
        key = (b, p, l)
        if key not in groups:
            # на некоторых psm/оem комбинациях line-строка может не попасть — создадим
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            groups[key] = {"bbox": (x, y, x + w, y + h), "words": [], "confs": []}
        groups[key]["words"].append(text)
        groups[key]["confs"].append(float(conf))

    # Сформируем сегменты
    for key, g in groups.items():
        words = g["words"]
        confs = g["confs"]
        if not words:
            # иногда строка без слов — пропустим
            continue
        segments.append({
            "bbox": g["bbox"],
            "text": " ".join(words),
            "avg_conf": _avg(confs),
            "words": len(words),
        })

    # упорядочим сверху-вниз, слева-направо
    segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    return segments


# Для совместимости со старым именем (строки)
def segment_lines(pil_img: Image.Image, langs: str = "rus+eng", psm: int = 6, oem: int = 1) -> List[Dict[str, Any]]:
    return segment(pil_img, langs=langs, psm=psm, level=4, oem=oem)

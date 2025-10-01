# tesseract/segment_lines.py
from typing import Dict, List, Tuple
from PIL import Image
import pytesseract

def segment_lines(pil_img: Image.Image, langs: str = "rus+eng", psm: int = 6) -> List[Dict]:
    """
    Возвращает список строк: {"bbox":(x1,y1,x2,y2), "text":str, "avg_conf":float, "words":int}
    На базе pytesseract.image_to_data c level==4.
    """
    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(pil_img, lang=langs, output_type=pytesseract.Output.DICT, config=config)

    lines: Dict[Tuple[int, int, int], Dict] = {}
    # Уровни: 1=page,2=block,3=para,4=line,5=word
    for i, level in enumerate(data["level"]):
        if level == 4:  # line
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            left = int(data["left"][i]); top = int(data["top"][i])
            width = int(data["width"][i]); height = int(data["height"][i])
            bbox = (left, top, left + width, top + height)
            lines[key] = {"bbox": bbox, "texts": [], "confs": []}

    for i, level in enumerate(data["level"]):
        if level == 5:  # word
            text = (data["text"][i] or "").strip()
            conf = data["conf"][i]
            if text and conf != "-1":
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                if key in lines:
                    lines[key]["texts"].append(text)
                    try:
                        lines[key]["confs"].append(float(conf))
                    except Exception:
                        pass

    out: List[Dict] = []
    for key in sorted(lines.keys()):
        obj = lines[key]
        words = obj["texts"]
        text = " ".join(words).strip()
        confs = obj["confs"]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        out.append({
            "bbox": obj["bbox"],
            "text": text,
            "avg_conf": avg_conf,
            "words": len(words),
        })
    return out

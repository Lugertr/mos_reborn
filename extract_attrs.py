# extract_attrs.py
import re
from typing import List, Dict, Any

MONTHS_OLD = r"(январ[ья]|феврал[ья]|март[ае]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[ае]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])"
DATE_RE = re.compile(rf"(\d{{1,2}}\s+{MONTHS_OLD}\s+\d{{3,4}})", re.IGNORECASE)
ARCHIVE_CODE_RE = re.compile(r"\b(\d+)\.(\d+)\.(\d+)\b")
# очень грубо: два-три слова с заглавной
NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.[А-ЯЁ]\.|(?:\s+[А-ЯЁ][а-яё]+){1,2})")

def extract_attributes_from_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    На вход блоки [{'text':..., 'bbox':[...], 'engine':...}, ...]
    Возвращает список атрибутов с привязкой к bbox.
    """
    attrs = []
    for b in blocks:
        t = b.get("text", "")
        bbox = b.get("bbox")

        for m in DATE_RE.finditer(t):
            attrs.append({"type": "date", "value": m.group(1), "bbox": bbox})

        for m in ARCHIVE_CODE_RE.finditer(t):
            attrs.append({"type": "archive_code", "value": m.group(0), "bbox": bbox})

        for m in NAME_RE.finditer(t):
            attrs.append({"type": "name", "value": m.group(0), "bbox": bbox})

    return attrs

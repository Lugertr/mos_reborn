"""
Purpose
-------
Сегментация распознанного Tesseract'ом содержимого на **строки** (level=4)
или **слова** (level=5) на основе вывода `pytesseract.image_to_data`.

How it works
------------
`pytesseract.image_to_data` возвращает таблицу (DICT) по уровням иерархии
страницы (page → block → par → line → word). Мы либо:
  • при `level=5` — берём каждое слово как отдельный сегмент,
  • при `level=4` — агрегируем слова в строки по ключу (block, par, line).

Output schema (для каждого сегмента)
------------------------------------
{
    "bbox": (x1, y1, x2, y2),  # прямоугольник в координатах исходного изображения
    "text": "<склеенный текст>",
    "avg_conf": <float>,       # средняя уверенность по словам (0..100)
    "words": <int>             # число слов внутри сегмента
}

Params
------
- langs: языки tesseract (например, "rus+eng")
- psm: page segmentation mode tesseract (сильно влияет на разметку)
- level: 4 = строки, 5 = слова
- oem: движок tesseract (legacy/lstm/best); передаётся через config
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from PIL import Image
import pytesseract

def _avg(xs):
    """
    Среднее значение по списку (безопасно обрабатывает пустой список).

    Args:
        xs: последовательность чисел (float/int)

    Returns:
        Среднее xs, либо 0.0 если список пуст.
    """
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

    Behavior:
        - level==4: собирает строки, агрегируя слова (ключ группировки: (block, par, line))
        - level==5: возвращает слова как отдельные сегменты

    Notes:
        - psm/oem передаются в tesseract через config-строку.
        - Поля `text`, `conf`, `left/top/width/height` берём из таблицы data.
        - Конфиденс '-1' указывает на «нет данных» — такие элементы пропускаются.
    """
    # Формируем конфиг для tesseract. Здесь задаются движок и режим сегментации.
    cfg = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(
        pil_img,
        lang=langs,
        output_type=pytesseract.Output.DICT,
        config=cfg
    )

    n = len(data["level"])
    segments: List[Dict[str, Any]] = []

    if level == 5:
        # --- Режим слов: каждое слово — отдельный сегмент ---
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
        # Стабильный порядок: сверху-вниз, слева-направо
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        return segments

    # --- Режим строк (level==4): агрегируем слова по (block, par, line) ---
    # Соберём каркасы строк из записей level=4 (их bbox используем как «рамку» строки)
    groups: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

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

    # Добавляем слова (level=5) в соответствующие группы
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
            # На некоторых комбинациях psm/oem line-запись может отсутствовать — создаём каркас по первому слову
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            groups[key] = {"bbox": (x, y, x + w, y + h), "words": [], "confs": []}
        groups[key]["words"].append(text)
        groups[key]["confs"].append(float(conf))

    # Превращаем группы в финальные сегменты строк
    for key, g in groups.items():
        words = g["words"]
        confs = g["confs"]
        if not words:
            # Иногда встречаются пустые line-записи без слов — пропускаем
            continue
        segments.append({
            "bbox": g["bbox"],
            "text": " ".join(words),
            "avg_conf": _avg(confs),
            "words": len(words),
        })

    # Единый порядок для стабильности: сверху-вниз, слева-направо
    segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    return segments


# Для совместимости со старым именем (строки)
def segment_lines(pil_img: Image.Image, langs: str = "rus+eng", psm: int = 6, oem: int = 1) -> List[Dict[str, Any]]:
    """
    Обёртка для обратной совместимости: сегментация **строк** (level=4).

    Args:
        pil_img: Изображение страницы (PIL.Image)
        langs: Языки tesseract (например, "rus+eng")
        psm: Page Segmentation Mode
        oem: Движок tesseract

    Returns:
        Список сегментов-строк в формате, описанном в шапке модуля.
    """
    return segment(pil_img, langs=langs, psm=psm, level=4, oem=oem)

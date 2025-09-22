# line_ocr.py
from PIL import Image
import pytesseract
from handwriting import recognize_handwriting
from typing import List, Dict, Any, Tuple


def recognize_handwriting_by_lines(pil_img: Image.Image, langs: str = "rus+eng") -> Dict[str, Any]:
    """
    Делим изображение на строки через Tesseract (level=4),
    каждую строку обрезаем и отправляем в TrOCR.
    Возвращает dict с plain_text и списком строк [{text, bbox}].
    """
    # pytesseract уровни: 1=page, 2=block, 3=paragraph, 4=line, 5=word
    data = pytesseract.image_to_data(
        pil_img, lang=langs, output_type=pytesseract.Output.DICT
    )

    lines: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for i in range(len(data["level"])):
        if data["level"][i] == 4:  # строки
            block_num = data["block_num"][i]
            par_num = data["par_num"][i]
            line_num = data["line_num"][i]

            # координаты строки
            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])
            bbox = (left, top, left + width, top + height)

            # вырезаем строку
            crop = pil_img.crop(bbox)

            # распознаём через TrOCR
            text = recognize_handwriting(crop)

            key = (block_num, par_num, line_num)
            lines[key] = {"text": text, "bbox": bbox}

    # сортируем по порядку (сверху вниз)
    sorted_lines = [lines[k] for k in sorted(lines.keys())]

    # собираем весь текст
    plain_text = "\n".join([line["text"] for line in sorted_lines])

    return {
        "engine": "trocr_line",
        "plain_text": plain_text,
        "lines": sorted_lines
    }

import os, shutil, uuid
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from typing import List, Dict, Any, Tuple

from handwriting import recognize_handwriting
from preproc import (
    preprocess_for_print_soft,
    preprocess_for_print_hard,
    preprocess_for_hand,
)
from line_ocr import recognize_handwriting_by_lines

app = FastAPI(title="OCR Service (Tesseract + TrOCR, job-API)")

UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Вспомогательные функции ----------

def tesseract_blocks(pil_img: Image.Image, langs: str = "rus+eng") -> Tuple[List[Dict[str, Any]], str, float]:
    """Гоним изображение через Tesseract и собираем блоки + полный текст."""

    # Получаем "сырые" слова с координатами
    data = pytesseract.image_to_data(pil_img, lang=langs, output_type=pytesseract.Output.DICT)
    blocks, confs = [], []
    for i, txt in enumerate(data["text"]):
        text = (txt or "").strip()
        conf = data["conf"][i]
        if text and conf != "-1":
            left = int(data["left"][i]); top = int(data["top"][i])
            width = int(data["width"][i]); height = int(data["height"][i])
            conf_f = float(conf)
            blocks.append({
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "conf": conf_f,
                "engine": "tesseract"
            })
            confs.append(conf_f)

    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # Получаем красивый plain_text (как в CLI)
    plain_text = pytesseract.image_to_string(pil_img, lang=langs, config="--oem 1 --psm 6")

    return blocks, plain_text.strip(), avg_conf


def trocr_blocks(pil_img: Image.Image) -> Tuple[List[Dict[str, Any]], str]:
    text = recognize_handwriting(pil_img) or ""
    blocks, y_offset = [], 0
    for line in text.split("\n"):
        x_offset = 0
        for w in line.split():
            blocks.append({
                "text": w,
                "bbox": [x_offset, y_offset, x_offset + 50, y_offset + 20],
                "conf": None,
                "engine": "trocr"
            })
            x_offset += 60
        y_offset += 30
    return blocks, text


# ---------- Синхронный OCR ----------

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    mode: str = Query("auto", enum=["auto", "print", "hand"]),
    langs: str = Query("rus+eng"),
    preproc: str = Query("soft", enum=["soft", "hard", "raw"])
):
    """
    mode=print → только Tesseract
    mode=hand  → только TrOCR (рукопись)
    mode=auto  → Tesseract → fallback на TrOCR
    preproc=soft|hard|raw → тип предобработки печатного текста
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        pil_img = Image.open(file_path).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # -------- PRINT ----------
    if mode == "print":
        debug_path = os.path.join(DEBUG_DIR, f"{file.filename}_print_{preproc}.png")
        if preproc == "soft":
            prep = preprocess_for_print_soft(pil_img, debug_path=debug_path)
        elif preproc == "hard":
            prep = preprocess_for_print_hard(pil_img, debug_path=debug_path)
        else:  # raw
            prep = pil_img
            prep.save(debug_path)

        blocks, text, avg_conf = tesseract_blocks(prep, langs=langs)
        return {"engine": "tesseract", "avg_conf": avg_conf, "plain_text": text, "blocks": blocks}

    # -------- HAND ----------
    if mode == "hand":
        debug_path = os.path.join(DEBUG_DIR, f"{file.filename}_hand.png")
        prep = preprocess_for_hand(pil_img, debug_path=debug_path)
        blocks, text = trocr_blocks(prep)
        return {"engine": "trocr", "plain_text": text, "blocks": blocks}

    # -------- AUTO ----------
    debug_path = os.path.join(DEBUG_DIR, f"{file.filename}_auto_{preproc}.png")
    if preproc == "soft":
        prep = preprocess_for_print_soft(pil_img, debug_path=debug_path)
    elif preproc == "hard":
        prep = preprocess_for_print_hard(pil_img, debug_path=debug_path)
    else:
        prep = pil_img
        prep.save(debug_path)

    blocks, text, avg_conf = tesseract_blocks(prep, langs=langs)
    words_count = len(blocks)

    if (avg_conf < 70.0) or (words_count < 3 and len(text) < 12):
        debug_path = os.path.join(DEBUG_DIR, f"{file.filename}_auto_fallback.png")
        prep = preprocess_for_hand(pil_img, debug_path=debug_path)
        blocks, text = trocr_blocks(prep)
        return {"engine": "trocr", "fallback_from": {"avg_conf": avg_conf, "words": words_count}, "plain_text": text, "blocks": blocks}
    else:
        return {"engine": "tesseract", "avg_conf": avg_conf, "plain_text": text, "blocks": blocks}

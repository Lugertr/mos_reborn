# main.py
import os, shutil, uuid
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from typing import List, Dict, Any, Tuple

from handwriting import recognize_handwriting
from preproc import preprocess_for_print, preprocess_for_hand
from line_ocr import recognize_handwriting_by_lines

app = FastAPI(title="OCR Service (Tesseract + TrOCR, job-API)")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Простое in-memory хранилище джобов
JOBS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Вспомогательные функции ----------

def tesseract_blocks(pil_img: Image.Image, langs: str = "rus+eng") -> Tuple[List[Dict[str, Any]], str, float]:
    """Гоним изображение через Tesseract"""
    data = pytesseract.image_to_data(pil_img, lang=langs, output_type=pytesseract.Output.DICT)

    blocks = []
    confs = []
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

    blocks_sorted = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    plain_text = " ".join([b["text"] for b in blocks_sorted])
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return blocks_sorted, plain_text, avg_conf


def trocr_blocks(pil_img: Image.Image) -> Tuple[List[Dict[str, Any]], str]:
    """
    TrOCR возвращает сплошной текст.
    Здесь мы разбиваем его на строки/слова, чтобы структура была похожа на Tesseract.
    """
    text = recognize_handwriting(pil_img) or ""
    blocks = []
    y_offset = 0

    for line in text.split("\n"):
        words = line.split()
        x_offset = 0
        for w in words:
            blocks.append({
                "text": w,
                "bbox": [x_offset, y_offset, x_offset + 50, y_offset + 20],
                "conf": None,
                "engine": "trocr"
            })
            x_offset += 60
        y_offset += 30

    return blocks, text


# ---------- Старый "синхронный" OCR (для тестов) ----------

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    mode: str = Query("auto", enum=["auto", "print", "hand"]),
    langs: str = Query("rus+eng")
):
    """
    mode=print → только Tesseract
    mode=hand  → только TrOCR (рукопись)
    mode=auto  → сначала Tesseract, если низкая уверенность → TrOCR
    """
    # Сохраняем файл
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        pil_img = Image.open(file_path).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"can't open image: {e}"}, status_code=400)

    if mode == "print":
        prep = preprocess_for_print(pil_img)
        blocks, text, avg_conf = tesseract_blocks(prep, langs=langs)
        return {"engine": "tesseract", "avg_conf": avg_conf, "plain_text": text, "blocks": blocks}

    if mode == "hand":
        prep = preprocess_for_hand(pil_img)
        blocks, text = trocr_blocks(prep)
        return {"engine": "trocr", "plain_text": text, "blocks": blocks}

    # AUTO
    prep = preprocess_for_print(pil_img)
    blocks, text, avg_conf = tesseract_blocks(prep, langs=langs)
    words_count = len(blocks)

    use_trocr = (avg_conf < 70.0) or (words_count < 3 and len(text) < 12)

    if use_trocr:
        prep = preprocess_for_hand(pil_img)
        blocks, text = trocr_blocks(prep)
        return {
            "engine": "trocr",
            "fallback_from": {"avg_conf": avg_conf, "words": words_count},
            "plain_text": text,
            "blocks": blocks
        }
    else:
        return {"engine": "tesseract", "avg_conf": avg_conf, "plain_text": text, "blocks": blocks}


# main.py в /ocr_lines
@app.post("/ocr_lines")
async def ocr_lines(file: UploadFile = File(...), langs: str = Query("rus+eng")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        pil_img = Image.open(file_path).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"can't open image: {e}"}, status_code=400)

    prep = preprocess_for_hand(pil_img)
    lines = recognize_handwriting_by_lines(prep, langs=langs)   # вернёт list[dict]
    plain = "\n".join([ln["text"] for ln in lines if (ln["text"] or "").strip()])

    return {"engine": "trocr_line", "plain_text": plain, "lines": lines}




# ---------- Новый job-based API ----------

@app.post("/ocr_job/start")
async def ocr_job_start(
    file: UploadFile = File(...),
    mode: str = Query("auto", enum=["auto", "print", "hand"]),
    langs: str = Query("rus+eng")
):
    """Стартуем асинхронное задание OCR"""
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Пока упрощённо: сразу кладём в очередь на обработку
    JOBS[job_id] = {"status": "queued", "result": None, "mode": mode, "langs": langs, "file_path": file_path}

    # Для MVP можно запустить обработку синхронно, но без падения основного API
    try:
        pil_img = Image.open(file_path).convert("RGB")
        if mode == "print":
            prep = preprocess_for_print(pil_img)
            blocks, text, avg_conf = tesseract_blocks(prep, langs=langs)
            JOBS[job_id] = {"status": "done", "result": {"engine": "tesseract", "avg_conf": avg_conf, "plain_text": text, "blocks": blocks}}

        elif mode == "hand":
            prep = preprocess_for_hand(pil_img)
            blocks, text = trocr_blocks(prep)
            JOBS[job_id] = {"status": "done", "result": {"engine": "trocr", "plain_text": text, "blocks": blocks}}

        else:  # auto
            prep = preprocess_for_print(pil_img)
            blocks, text, avg_conf = tesseract_blocks(prep, langs=langs)
            words_count = len(blocks)
            if (avg_conf < 70.0) or (words_count < 3 and len(text) < 12):
                prep = preprocess_for_hand(pil_img)
                blocks, text = trocr_blocks(prep)
                JOBS[job_id] = {"status": "done", "result": {"engine": "trocr", "fallback_from": {"avg_conf": avg_conf, "words": words_count}, "plain_text": text, "blocks": blocks}}
            else:
                JOBS[job_id] = {"status": "done", "result": {"engine": "tesseract", "avg_conf": avg_conf, "plain_text": text, "blocks": blocks}}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}

    return {"job_id": job_id}


@app.get("/ocr_job/status/{job_id}")
async def ocr_job_status(job_id: str):
    """Проверяем статус OCR-задачи"""
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job

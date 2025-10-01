# api/routes.py
"""
API routes for the OCR service.

POST /ocr_segments:
- Принимает multipart/form-data:
    file: UploadFile (изображение)
    img_width: int
    img_height: int
    langs: str = "rus+eng"
    psm: int = 6
    preproc: str = "soft"  # "soft" | "hard"
    conf_threshold: float = 70.0
    wer_mode: str = "proxy"  # "proxy" | "exact"
    ref_text: Optional[str] = None  # для WER exact (построчно)

- Логика:
    1) OSD — корректируем ориентацию (0/90/180/270)
    2) Предобработка печатного (soft/hard)
    3) Сегментация строк (pytesseract.image_to_data, level==4)
    4) Для "слабых" строк — опциональный fallback в TrOCR
    5) Нормализация NFC, вычисление WER:
        - exact: по ref_text (построчно)
        - proxy: из доверия Tesseract или несогласия Tess vs TrOCR
    6) Ответ: массив SegmentOut с полями:
        coords: (x, y) верхний-левый
        preview_value: текст Tesseract (для превью)
        value: финальный текст (TrOCR при фоллбэке, иначе Tesseract)
        wer: float
        width, height: размеры bbox
    7) stats — краткая сводка
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from PIL import Image

from schemas import SegmentsResponse, SegmentOut
from config import MAX_UPLOAD_BYTES, TROCR_ENABLED
from logging_utils import setup_logger, timeblock
from preprocessing.osd_utils import apply_osd_rotation
from preprocessing.preproc import preprocess_for_print_soft, preprocess_for_print_hard
from postprocessing.text_norm import clean_spaces
from postprocessing.wer import wer_exact, wer_proxy_from_conf, wer_proxy_between
from tesseract.segment_lines import segment_lines

logger = setup_logger("ocr")
router = APIRouter()


@router.post("/ocr_segments", response_model=SegmentsResponse)
async def ocr_segments(
    request: Request,
    file: UploadFile = File(...),
    img_width: int = Form(...),
    img_height: int = Form(...),
    langs: str = Form("rus+eng"),
    psm: int = Form(6),
    preproc: str = Form("soft"),                   # "soft" | "hard"
    conf_threshold: float = Form(70.0),
    wer_mode: str = Form("proxy"),                 # "proxy" | "exact"
    ref_text: Optional[str] = Form(None),
):
    # Проверка размера по заголовку (жёсткое ограничение)
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    with timeblock(logger, "ocr_segments_total"):
        # 1) Чтение изображения
        try:
            pil = Image.open(file.file).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

        # 2) OSD — исправление поворота (кратно 90°)
        with timeblock(logger, "osd_rotation"):
            pil = apply_osd_rotation(pil)

        # 3) Предобработка для печатного текста
        with timeblock(logger, "preprocess", mode=preproc):
            if preproc == "hard":
                prep = preprocess_for_print_hard(pil)
            else:
                prep = preprocess_for_print_soft(pil)

        # 4) Сегментация строк (level==4)
        with timeblock(logger, "tesseract_segment", langs=langs, psm=psm):
            lines = segment_lines(prep, langs=langs, psm=psm)
            # lines: [{"bbox": (x1,y1,x2,y2), "text": str, "avg_conf": float, "words": int}, ...]

        # 5) Отбор слабых строк для TrOCR
        low = [
            ln for ln in lines
            if (ln["avg_conf"] < conf_threshold) or (ln["words"] < 2 and len(ln["text"]) < 10)
        ]
        low_ids = {id(ln) for ln in low}  # помечаем по id для стабильной идентификации

        # 6) Референсы построчно (для exact WER)
        ref_lines: List[str] = []
        if wer_mode == "exact" and ref_text is not None:
            ref_lines = [clean_spaces(s) for s in ref_text.splitlines()]

        # 7) Fallback в TrOCR (опционально)
        trocr_texts: List[str] = []   # всегда объявляем, чтобы избежать NameError
        if TROCR_ENABLED and low:
            from trocr.runtime import recognize_batch  # ленивый импорт
            # Кропим из оригинального изображения (без бинаризации)
            crops = [pil.crop(ln["bbox"]) for ln in low]
            with timeblock(logger, "trocr_batch", count=len(crops)):
                trocr_texts = await asyncio.to_thread(recognize_batch, crops)

        # 8) Сбор ответа
        segments: List[SegmentOut] = []
        trocr_i = 0
        used_trocr = 0

        for idx, ln in enumerate(lines):
            x1, y1, x2, y2 = ln["bbox"]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            # Tesseract-гипотеза для превью
            tess_text = clean_spaces(ln["text"])

            if TROCR_ENABLED and id(ln) in low_ids:
                # Используем TrOCR как финальную гипотезу (если ответ есть)
                trocr_text = clean_spaces(trocr_texts[trocr_i]) if trocr_i < len(trocr_texts) else ""
                trocr_i += 1
                used_trocr += 1

                final_value = trocr_text or tess_text  # если TrOCR вернул пусто — подстрахуемся Tess'ом

                if wer_mode == "exact" and idx < len(ref_lines):
                    wer_val = wer_exact(ref_lines[idx], final_value)
                else:
                    # proxy: если TrOCR пуст — по conf Tess; иначе — несогласие Tess vs TrOCR
                    wer_val = wer_proxy_from_conf(ln["avg_conf"]) if not trocr_text else wer_proxy_between(tess_text, trocr_text)
            else:
                # Остаёмся на Tesseract
                final_value = tess_text
                if wer_mode == "exact" and idx < len(ref_lines):
                    wer_val = wer_exact(ref_lines[idx], final_value)
                else:
                    wer_val = wer_proxy_from_conf(ln["avg_conf"])

            segments.append(SegmentOut(
                coords=(x1, y1),
                preview_value=tess_text,   # всегда Tesseract
                value=final_value,         # финальное значение (TrOCR при фоллбэке)
                wer=wer_val,
                width=w,
                height=h
            ))

        stats = {
            "total_lines": len(lines),
            "fallback_to_trocr": used_trocr if TROCR_ENABLED else 0,
            "conf_threshold": conf_threshold,
            "wer_mode": wer_mode,
            "trocr_enabled": TROCR_ENABLED,
            "img_width": img_width,
            "img_height": img_height,
        }

        return SegmentsResponse(segments=segments, stats=stats)

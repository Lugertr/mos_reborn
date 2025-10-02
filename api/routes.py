# api/routes.py
from __future__ import annotations
import asyncio
import time
from io import BytesIO
from typing import List, Optional, Callable, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image

from schemas import SegmentsResponse, SegmentOut
from config import (
    MAX_UPLOAD_BYTES,
    TROCR_ENABLED,
    TESS_LANGS, TESS_PSM, TESS_OEM, TESS_LEVEL,
    TESS_CONF_THRESHOLD, TESS_MIN_WORDS, TESS_MIN_TEXT_LEN,
    PREPROC_MODE_DEFAULT,
)
from logging_utils import setup_logger
from preprocessing.osd_utils import apply_osd_rotation
from preprocessing.preproc import preprocess_for_print_soft, preprocess_for_print_hard
from postprocessing.text_norm import clean_spaces
from postprocessing.wer import wer_exact, wer_proxy_from_conf, wer_proxy_between
from tesseract.segment_lines import segment
from api.progress import sse

logger = setup_logger("ocr")
router = APIRouter()


# ---------- небольшая утилита для шагов/таймингов ----------
class StepTrace:
    def __init__(self):
        self.trace: List[Dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def mark(self, name: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        now = time.perf_counter()
        item = {
            "step": name,
            "t_ms": int((now - self._t0) * 1000),
        }
        if extra:
            item.update(extra)
        self.trace.append(item)
        return item


# ---------- безопасное чтение UploadFile с лимитом ----------
async def read_upload_limited(file: UploadFile, limit: int) -> bytes:
    """
    Читает не более (limit+1) байт из UploadFile.
    Если превышение — 413. Возвращает байты файла.
    """
    # NB: UploadFile.read() уже асинхронно читает содержимое из temp/spooled файла
    chunk = await file.read(limit + 1)
    if len(chunk) > limit:
        raise HTTPException(status_code=413, detail="Payload too large")
    if not chunk:
        raise HTTPException(status_code=400, detail="Empty upload")
    return chunk


# ---------- общий пайплайн распознавания ----------
async def run_pipeline(
    request: Request,
    file: UploadFile,
    img_w: int,
    img_h: int,
    langs: str,
    tess_level: int,
    psm: int,
    preproc_mode: str,
    conf_threshold: float,
    wer_mode: str,
    ref_text: Optional[str],
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Выполняет весь /ocr_segments и:
      - если emit=None → просто возвращает dict (подходит для JSON ответа);
      - если emit задан → дергает emit("progress", {...}) на каждом этапе и в конце возвращает итоговый dict (для SSE).
    """
    def progress(phase: str, pct: int, **kw):
        payload = {"phase": phase, "progress": max(0, min(100, pct))}
        payload.update(kw)
        if emit:
            emit("progress", payload)

    # лимит размера по заголовку, если есть
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Payload too large")
        except ValueError:
            # игнорируем некорректный заголовок — отработаем лимитом ниже
            pass

    trace = StepTrace()
    progress("received", 3, note="upload received")
    trace.mark("received", {"filename": file.filename})

    # читаем файл с жёстким лимитом, открываем PIL из байтов
    t0 = time.perf_counter()
    blob = await read_upload_limited(file, MAX_UPLOAD_BYTES)
    trace.mark("read_limited_done", {"duration_ms": int((time.perf_counter() - t0) * 1000), "bytes": len(blob)})

    try:
        pil = Image.open(BytesIO(blob)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")
    trace.mark("image_opened", {"size": pil.size})
    progress("osd_rotation", 8)

    # OSD-поворот
    t0 = time.perf_counter()
    pil = apply_osd_rotation(pil)
    trace.mark("osd_rotation_done", {"duration_ms": int((time.perf_counter() - t0) * 1000)})
    progress("preprocess", 18, mode=preproc_mode)

    # предобработка
    t0 = time.perf_counter()
    if preproc_mode == "hard":
        prep = preprocess_for_print_hard(pil)
    else:
        prep = preprocess_for_print_soft(pil)
    trace.mark("preprocess_done", {"duration_ms": int((time.perf_counter() - t0) * 1000)})
    progress("segment_lines", 35, langs=langs, psm=psm)

    # сегментация строк (level==4)
    t0 = time.perf_counter()
    lines = segment(prep, langs=langs, psm=psm, level=tess_level, oem=TESS_OEM)
    seg_ms = int((time.perf_counter() - t0) * 1000)
    trace.mark("segment_lines_done", {"duration_ms": seg_ms, "lines": len(lines)})
    progress("segment_lines_done", 55, lines=len(lines))

    # отберём сомнительные для fallback
    low = [
        ln for ln in lines
        if (ln["avg_conf"] < conf_threshold) or (ln["words"] < TESS_MIN_WORDS and len(ln["text"]) < TESS_MIN_TEXT_LEN)
    ]
    trace.mark("low_conf_selected", {"count": len(low), "threshold": conf_threshold})

    # подготовим ссылки на референс построчно (для точного WER)
    ref_lines: List[str] = []
    if wer_mode == "exact" and ref_text is not None:
        ref_lines = [clean_spaces(s) for s in ref_text.splitlines()]

    # возможный fallback: TrOCR на сомнительных
    trocr_texts: List[str] = []
    if TROCR_ENABLED and low:
        progress("trocr_fallback", 65, count=len(low))
        try:
            from trocr.runtime import recognize_batch  # ленивый импорт
            crops = [pil.crop(ln["bbox"]) for ln in low]
            t0 = time.perf_counter()
            # В отдельном потоке, чтобы не блокировать event loop
            trocr_texts = await asyncio.to_thread(recognize_batch, crops)
            trace.mark("trocr_done", {
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "count": len(crops)
            })
        except Exception as e:
            logger.exception("TrOCR fallback failed: %s", e)
            trace.mark("trocr_failed", {"error": str(e)})
    else:
        trace.mark("trocr_skipped", {"enabled": bool(TROCR_ENABLED), "low_conf": len(low)})

    # собрать сегменты
    segments: List[SegmentOut] = []
    trocr_i = 0
    for idx, ln in enumerate(lines):
        x1, y1, x2, y2 = ln["bbox"]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        tesser_text = clean_spaces(ln["text"])
        final_text = tesser_text

        if TROCR_ENABLED and ln in low and trocr_i < len(trocr_texts):
            final_text = clean_spaces(trocr_texts[trocr_i])
            trocr_i += 1

        # WER
        if wer_mode == "exact" and idx < len(ref_lines):
            wer_val = wer_exact(ref_lines[idx], final_text)
        else:
            if TROCR_ENABLED and final_text != tesser_text:
                # прокси через расхождение предсказаний
                wer_val = wer_proxy_between(tesser_text, final_text)
            else:
                wer_val = wer_proxy_from_conf(ln["avg_conf"])

        segments.append(SegmentOut(
            coords=(x1, y1),
            value=final_text,            # финальное (TrOCR при фолбэке, иначе Tesseract)
            preview_value=tesser_text,   # предпросмотр (Tesseract)
            wer=wer_val,
            width=w,
            height=h
        ))

    progress("assembling", 88, segments=len(segments))

    stats = {
        "total_lines": len(lines),
        "fallback_to_trocr": len(low) if TROCR_ENABLED else 0,
        "conf_threshold": conf_threshold,
        "wer_mode": wer_mode,
        "trocr_enabled": TROCR_ENABLED,
        "img_width": img_w,
        "img_height": img_h,
        "trace": trace.trace,  # массив этапов с таймингами
    }
    result = SegmentsResponse(segments=segments, stats=stats).model_dump()
    progress("done", 100)

    return result


# ---------- основной эндпоинт ----------
@router.post("/ocr_segments")
async def ocr_segments(
    request: Request,
    file: UploadFile = File(...),
    img_width: int = Form(...),
    img_height: int = Form(...),
    langs: str = Form(TESS_LANGS),
    psm: int = Form(TESS_PSM),
    tess_level: int = Form(TESS_LEVEL),          # <— НОВОЕ: 4|5
    preproc: str = Form(PREPROC_MODE_DEFAULT),   # "soft"|"hard"
    conf_threshold: float = Form(TESS_CONF_THRESHOLD),
    wer_mode: str = Form("proxy"),               # "proxy"|"exact"
    ref_text: Optional[str] = Form(None),
    stream: bool = Form(False),
):
    """
    По умолчанию возвращает JSON с результатом и stats.trace (тайминги шагов).
    Если stream=true — возвращает SSE-поток (text/event-stream) с событиями 'progress' и финальным 'result'.
    """

    if not stream:
        # Обычный JSON-ответ (без потоковой передачи).
        payload = await run_pipeline(
            request, file, img_width, img_height, langs,tess_level, psm, preproc,
            conf_threshold, wer_mode, ref_text, emit=None
        )
        return payload

    # --- Потоковый SSE-режим ---
    def _make_emitter(queue: asyncio.Queue[bytes]):
        def emit(event_name: str, data: Dict[str, Any]):
            queue.put_nowait(sse(event_name, data))
        return emit

    async def gen():
        q: asyncio.Queue[bytes] = asyncio.Queue()
        emit_cb = _make_emitter(q)

        # стартовое событие
        emit_cb("hello", {"phase": "start", "progress": 0})

        async def _run():
            try:
                result = await run_pipeline(
                    request, file, img_width, img_height, langs,tess_level, psm, preproc,
                    conf_threshold, wer_mode, ref_text, emit=emit_cb
                )
                q.put_nowait(sse("result", result))
            except HTTPException as he:
                q.put_nowait(sse("error", {"status": he.status_code, "detail": he.detail}))
            except Exception as e:
                logger.exception("ocr_segments stream failed: %s", e)
                q.put_nowait(sse("error", {"status": 500, "detail": str(e)}))
            finally:
                # маркер завершения
                q.put_nowait(b"")

        # запускаем пайплайн в фоне
        task = asyncio.create_task(_run())

        # отдаём из очереди по мере появления
        while True:
            chunk = await q.get()
            if chunk == b"":  # завершение
                break
            yield chunk

        await task

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # важно для nginx, чтобы не буферил
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

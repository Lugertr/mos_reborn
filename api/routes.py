# api/routes.py
"""
Purpose
-------
HTTP-эндпоинт распознавания и поток прогресса (SSE). В этом модуле собран
весь «сквозной» пайплайн: приём файла → проверка лимитов → автоповорот (OSD) →
предобработка → сегментация на строки Tesseract'ом → выбор «сомнительных» строк →
опциональный fallback в TrOCR → расчёт метрики WER → сборка ответа.

Key concepts
------------
- Порог уверенности Tesseract (`conf_threshold`) решает, какие строки уйдут в TrOCR.
- `wer_mode`: "proxy" (приближённая оценка) или "exact" (по референсу `ref_text`).
- SSE-режим: отдаём события `hello` → `progress` ... → `result` по `text/event-stream`.

Endpoints
---------
POST /ocr_segments  — обычный JSON-ответ или потоковый SSE (по флагу `stream`).
"""

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
    """
    Накопитель этапов пайплайна с относительными таймингами.

    Использование:
        trace = StepTrace()
        trace.mark("preprocess_done", {"duration_ms": 123})

    Хранит список словарей вида:
        {"step": <str>, "t_ms": <int>, **extra}
    где t_ms — миллисекунды от момента создания `StepTrace`.
    """
    def __init__(self):
        self.trace: List[Dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def mark(self, name: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Добавить отметку в журнал этапов.

        Args:
            name: Человекочитаемое имя шага (например, "segment_lines_done").
            extra: Любые дополнительные поля (кол-во строк, длительность и т.п.).

        Returns:
            Добавленный элемент журнала (dict).
        """
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
    Прочитать из UploadFile не более `limit` байт (жёсткий контроль размера).

    Поведение:
    - если фактический размер превышает лимит → HTTP 413 Payload Too Large;
    - если файл пустой → HTTP 400;
    - иначе вернуть байтовое содержимое.

    Note:
        UploadFile.read() читает из временного файла асинхронно — блокировки event loop
        нет, но мы всё равно контролируем объём для надёжности и диагностики.
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
    Выполнить полный OCR-конвейер для одной страницы.

    Args:
        request: FastAPI Request (используется для чтения заголовков, напр. content-length).
        file: Загруженный файл изображения (страница документа).
        img_w, img_h: Размеры исходного изображения, переданные с фронта (в статистику).
        langs: Языки Tesseract (например, "rus+eng").
        tess_level: Уровень сегментации Tesseract (обычно 4 — строки; 5 — слова).
        psm: Page Segmentation Mode Tesseract.
        preproc_mode: "soft" | "hard" — режим предобработки печатного текста.
        conf_threshold: Порог средней уверенности строки для fallback в TrOCR.
        wer_mode: "proxy" | "exact" — способ оценки WER.
        ref_text: Эталонный текст для "exact" WER, разбивается по строкам.
        emit: Колбэк для эмита событий прогресса в SSE. Если None — прогресс не шлём.

    Returns:
        Dict, совместимый со схемой `SegmentsResponse` (result + stats + trace).

    Notes:
        - Прогресс по этапам отправляется через emit("progress", {...}) при наличии emit.
        - Fallback в TrOCR применяется только к «сомнительным» строкам и только если он включён.
    """
    def progress(phase: str, pct: int, **kw):
        payload = {"phase": phase, "progress": max(0, min(100, pct))}
        payload.update(kw)
        if emit:
            emit("progress", payload)

    # Лимит на размер тела по заголовку (если клиент его прислал).
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Payload too large")
        except ValueError:
            # Некорректный заголовок — игнорируем, реальный лимит будет ниже.
            pass

    trace = StepTrace()
    progress("received", 3, note="upload received")
    trace.mark("received", {"filename": file.filename})

    # Читаем файл в память с жёстким лимитом, дальше открываем через PIL.
    t0 = time.perf_counter()
    blob = await read_upload_limited(file, MAX_UPLOAD_BYTES)
    trace.mark("read_limited_done", {"duration_ms": int((time.perf_counter() - t0) * 1000), "bytes": len(blob)})

    try:
        pil = Image.open(BytesIO(blob)).convert("RGB")
    except Exception as e:
        # Ошибки чтения/декодирования изображения → 400 Bad Request
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")
    trace.mark("image_opened", {"size": pil.size})
    progress("osd_rotation", 8)

    # OSD-поворот (распознаём ориентацию и разворот страницы).
    t0 = time.perf_counter()
    pil = apply_osd_rotation(pil)
    trace.mark("osd_rotation_done", {"duration_ms": int((time.perf_counter() - t0) * 1000)})
    progress("preprocess", 18, mode=preproc_mode)

    # Лёгкая/жёсткая предобработка печатного текста.
    t0 = time.perf_counter()
    if preproc_mode == "hard":
        prep = preprocess_for_print_hard(pil)
    else:
        prep = preprocess_for_print_soft(pil)
    trace.mark("preprocess_done", {"duration_ms": int((time.perf_counter() - t0) * 1000)})
    progress("segment_lines", 35, langs=langs, psm=psm)

    # Сегментация на строки через Tesseract (level==4). Возвращает список словарей со
    # строками: bbox, распознанный текст, ср. уверенность, кол-во слов.
    t0 = time.perf_counter()
    lines = segment(prep, langs=langs, psm=psm, level=tess_level, oem=TESS_OEM)
    seg_ms = int((time.perf_counter() - t0) * 1000)
    trace.mark("segment_lines_done", {"duration_ms": seg_ms, "lines": len(lines)})
    progress("segment_lines_done", 55, lines=len(lines))

    # Отбор «сомнительных» строк: низкая уверенность или слишком мало слов/символов.
    low = [
        ln for ln in lines
        if (ln["avg_conf"] < conf_threshold) or (ln["words"] < TESS_MIN_WORDS and len(ln["text"]) < TESS_MIN_TEXT_LEN)
    ]
    trace.mark("low_conf_selected", {"count": len(low), "threshold": conf_threshold})

    # Референс для exact-WER (линейно по индексам строк).
    ref_lines: List[str] = []
    if wer_mode == "exact" and ref_text is not None:
        ref_lines = [clean_spaces(s) for s in ref_text.splitlines()]

    # Опциональный fallback в TrOCR для «сомнительных» строк.
    trocr_texts: List[str] = []
    if TROCR_ENABLED and low:
        progress("trocr_fallback", 65, count=len(low))
        try:
            from trocr.runtime import recognize_batch  # ленивый импорт, тяжёлые зависимости
            crops = [pil.crop(ln["bbox"]) for ln in low]
            t0 = time.perf_counter()
            # Запуск в отдельном потоке (CPU/GPU работа модели), чтобы не блокировать event loop.
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

    # Сборка финальных сегментов (bbox → ширина/высота; финальный текст с учётом fallback).
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

        # Расчёт WER:
        # - "exact": сравнение с соответствующей строкой `ref_text` (если есть).
        # - "proxy": оценка на основе conf Tesseract или расхождения Tesseract/TrOCR.
        if wer_mode == "exact" and idx < len(ref_lines):
            wer_val = wer_exact(ref_lines[idx], final_text)
        else:
            if TROCR_ENABLED and final_text != tesser_text:
                wer_val = wer_proxy_between(tesser_text, final_text)
            else:
                wer_val = wer_proxy_from_conf(ln["avg_conf"])

        segments.append(SegmentOut(
            coords=(x1, y1),
            value=final_text,            # финальный текст (TrOCR при фолбэке, иначе Tesseract)
            preview_value=tesser_text,   # предпросмотр: версия от Tesseract
            wer=wer_val,
            width=w,
            height=h
        ))

    progress("assembling", 88, segments=len(segments))

    # Статистика по задаче + трассировка этапов.
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
    tess_level: int = Form(TESS_LEVEL),          # уровень детализации распознавания Tesseract (4: строки, 5: слова)
    preproc: str = Form(PREPROC_MODE_DEFAULT),   # "soft"|"hard" — режим предобработки печати
    conf_threshold: float = Form(TESS_CONF_THRESHOLD),
    wer_mode: str = Form("proxy"),               # "proxy"|"exact"
    ref_text: Optional[str] = Form(None),
    stream: bool = Form(False),
):
    """
    Универсальный эндпоинт распознавания страницы.

    Режимы:
      - `stream=false` (по умолчанию): вернуть единый JSON (`SegmentsResponse`) после завершения пайплайна.
      - `stream=true`: вернуть Server-Sent Events (SSE) — события `hello`, последовательность `progress`,
        и финальное `result` с тем же содержимым, что и обычный JSON.

    Ошибки:
      - 400: некорректное изображение / некорректные параметры.
      - 413: превышен лимит загрузки.
      - 500: непредвиденная ошибка во время пайплайна.
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
        """
        Обёртка, превращающая emit(event, data) → байтовый чанκ формата SSE,
        который кладём в очередь для StreamingResponse.
        """
        def emit(event_name: str, data: Dict[str, Any]):
            queue.put_nowait(sse(event_name, data))
        return emit

    async def gen():
        """
        Асинхронный генератор — читает байты событий из очереди `q` и отдаёт их клиенту.
        Пустой байтовый блок `b""` — маркер завершения стрима.
        """
        q: asyncio.Queue[bytes] = asyncio.Queue()
        emit_cb = _make_emitter(q)

        # стартовое событие для инициализации со стороны клиента
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

        # запускаем пайплайн в фоне (чтобы генератор мог отдавать прогресс по мере готовности)
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
        "X-Accel-Buffering": "no",  # важно для nginx, чтобы не буферил стрим
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

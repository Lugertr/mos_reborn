# api/progress.py
"""
Purpose
-------
Мини-утилита для формирования строк протокола Server-Sent Events (SSE).
Возвращает «сырые» байты, готовые к отправке в StreamingResponse.

Формат одного события:
    id: <event_id>         (опционально)
    event: <event_name>    (опционально)
    data: <строка или JSON>
    data: <продолжение...> (каждая строка данных выводится отдельной data:)
    retry: <миллисекунды>  (опционально)
    <пустая строка>        (разделитель событий)
"""

from __future__ import annotations
import json
from typing import Any, Optional

def sse(event: Optional[str], data: Any, event_id: Optional[str] = None, retry_ms: Optional[int] = None) -> bytes:
    """
    Сформировать событие SSE в виде байтовой строки.

    Args:
        event: Имя события (например, "progress" | "result" | "error"). Если None — поле не добавляется.
        data: Данные события. Если это не строка — сериализуется в JSON (UTF-8, ensure_ascii=False).
        event_id: Идентификатор события (по желанию клиента).
        retry_ms: Рекомендованный интервал переподключения для клиента (миллисекунды).

    Returns:
        bytes: Готовый для отправки блок SSE, заканчивающийся пустой строкой.

    Notes:
        - Поддерживаются многострочные payload: каждая строка будет передана отдельной `data:`.
        - SSE — односторонний текстовый поток; заголовок ответа должен быть `text/event-stream`.
    """
    if not isinstance(data, str):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = data

    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    for line in (payload.splitlines() or [""]):
        lines.append(f"data: {line}")
    if retry_ms is not None:
        lines.append(f"retry: {int(retry_ms)}")
    lines.append("")  # пустая строка = разделитель событий
    return ("\n".join(lines) + "\n").encode("utf-8")

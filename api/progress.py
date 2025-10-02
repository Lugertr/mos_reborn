# api/progress.py
from __future__ import annotations
import json
from typing import Any, Optional

def sse(event: Optional[str], data: Any, event_id: Optional[str] = None, retry_ms: Optional[int] = None) -> bytes:
    """
    Формирует Server-Sent Event:
      event: <name>
      id: <id>
      data: <JSON-или-текст>   (многострочные data: поддерживаются)
      retry: <ms>
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

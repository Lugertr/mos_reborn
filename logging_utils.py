# logging_utils.py
"""
Purpose
-------
Утилиты логирования для сервиса:
- `setup_logger(name)`: создаёт/возвращает именованный логгер с единым форматом,
  не «проливающийся» в корневой логгер (uvicorn), чтобы сообщения не дублировались.
- `timeblock(logger, event, **extra)`: контекст-менеджер для измерения длительности
  произвольного блока кода и логирования результата как структурированного сообщения.

Почему так
----------
- Uvicorn уже настраивает свой root-логгер. Мы отключаем propagate у наших логгеров,
  чтобы записи не уходили в root и не появлялись дважды.
- `timeblock` логирует даже при исключении: блок находится в `finally`.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict

def setup_logger(name: str) -> logging.Logger:
    """
    Создать (или вернуть уже созданный) именованный логгер с базовой настройкой.

    Настройки:
      - Уровень INFO (можно поднять/опустить снаружи при необходимости).
      - `propagate=False`, чтобы не отправлять записи в root (uvicorn) и не дублировать вывод.
      - Один StreamHandler с форматом: "ts [LEVEL] logger_name: message".

    Пример:
        logger = setup_logger("ocr")
        logger.info("service started")

    Args:
        name: Имя логгера (например, "ocr", "ocr.train", "ocr.trocr").

    Returns:
        Готовый к использованию экземпляр logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Не даём сообщениям всплывать в root-логгер Uvicorn'а (избегаем дублей в консоли).
    logger.propagate = False

    # Идёмпотентность: если уже добавляли handler — второй раз не добавляем.
    if not logger.handlers:
        h = logging.StreamHandler()
        # Добавим имя логгера в формат, чтобы сразу видеть источник записи.
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger

@contextmanager
def timeblock(logger: logging.Logger, event: str, **extra: Dict):
    """
    Измерить время выполнения блока кода и залогировать факт с меткой события.

    Использование:
        with timeblock(logger, "trocr_load", run="runs/default"):
            heavy_init()

    Поведение:
      - В логи уходит dict вида:
            {"event": <event>, "duration_ms": <float>, ...extra}
      - Сообщение пишется в любом случае (в ветке finally), даже если внутри блока было исключение.

    Args:
        logger: Логгер, в который писать.
        event: Короткий идентификатор события (например, "trocr_load", "trocr_infer").
        **extra: Любые дополнительные пары ключ-значение, попадут в лог (например, run, batch, path).

    Note:
        Время округляется до сотых миллисекунды.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        payload = {"event": event, "duration_ms": round(dt * 1000, 2)}
        if extra:
            payload.update(extra)
        logger.info(payload)

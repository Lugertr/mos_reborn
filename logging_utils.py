# logging_utils.py
import logging
import time
from contextlib import contextmanager
from typing import Dict

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # НЕ даём сообщениям всплывать в root-логгер Uvicorn'а
    logger.propagate = False

    if not logger.handlers:
        h = logging.StreamHandler()
        # Добавим имя логгера, чтобы быстро понимать, откуда запись
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger

@contextmanager
def timeblock(logger: logging.Logger, event: str, **extra: Dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        payload = {"event": event, "duration_ms": round(dt * 1000, 2)}
        if extra:
            payload.update(extra)
        logger.info(payload)

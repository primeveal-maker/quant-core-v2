from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO


def configure_logging(log_dir: Path, level: int = DEFAULT_LOG_LEVEL) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    for name in ("app.log", "exec.log", "signals.log"):
        handler = RotatingFileHandler(log_dir / name, maxBytes=1_000_000, backupCount=3)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root.addHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)

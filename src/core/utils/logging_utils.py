"""Lightweight logging helpers with optional JSON output and correlation IDs."""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)


def set_correlation(*, run_id: str | None = None, thread_id: str | None = None) -> None:
    """Attach run/thread ids to subsequent log records in this context."""
    if run_id is not None:
        _run_id.set(run_id)
    if thread_id is not None:
        _thread_id.set(thread_id)


def clear_correlation() -> None:
    """Clear correlation contextvars."""
    _run_id.set(None)
    _thread_id.set(None)


def get_correlation() -> dict[str, str]:
    """Return non-empty correlation fields for the current context."""
    out: dict[str, str] = {}
    rid = _run_id.get()
    tid = _thread_id.get()
    if rid:
        out["run_id"] = rid
    if tid:
        out["thread_id"] = tid
    return out


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(get_correlation())
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Human-readable format with optional correlation suffixes."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        corr = get_correlation()
        if not corr:
            return base
        suffix = " ".join(f"{k}={v}" for k, v in corr.items())
        return f"{base} {suffix}"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured from LOG_LEVEL / LOG_FORMAT."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = (os.getenv("LOG_FORMAT") or "text").strip().lower()
        if fmt == "json":
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(
                _TextFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
        logger.addHandler(handler)
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger

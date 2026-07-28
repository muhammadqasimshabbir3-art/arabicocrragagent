"""Structured audit logging for the read-only database security pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

_LOGGER = logging.getLogger("agent.db_security")


def log_db_security_event(
    *,
    event: str,
    user_prompt: str = "",
    intent: str = "",
    confidence: float | None = None,
    layer: str = "",
    blocked: bool | None = None,
    generated_sql: str = "",
    validator_ok: bool | None = None,
    validator_error: str = "",
    tool_success: bool | None = None,
    tool_error: str = "",
    final_response: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Log one security pipeline event for debugging and audits."""
    payload: dict[str, Any] = {
        "event": event,
        "user_prompt": (user_prompt or "")[:2000],
        "intent": intent,
        "confidence": confidence,
        "layer": layer,
        "blocked": blocked,
        "generated_sql": (generated_sql or "")[:2000],
        "validator_ok": validator_ok,
        "validator_error": (validator_error or "")[:500],
        "tool_success": tool_success,
        "tool_error": (tool_error or "")[:500],
        "final_response": (final_response or "")[:2000],
    }
    if extra:
        payload["extra"] = extra
    # Drop empty noise for cleaner logs
    compact = {k: v for k, v in payload.items() if v not in (None, "", {})}
    _LOGGER.info("db_security %s", json.dumps(compact, default=str))


__all__ = ["log_db_security_event"]

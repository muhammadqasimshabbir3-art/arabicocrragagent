"""Gmail API inbox tools: OAuth unread reading and in-thread replies."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from langchain.tools import tool

from agent.async_utils import run_in_thread

# gmail_agent lives at src/gmail_agent.py (sibling of the agent package).
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from gmail_agent import (  # noqa: E402
    GmailAgentConfig,
    configure_logging,
    generate_reply_with_ollama,
    get_gmail_service,
    get_message,
    list_unread_messages,
    mark_as_read,
    process_unread_and_reply,
    send_reply,
)


def extract_gmail_inbox_limit(text: str) -> Optional[int]:
    """Parse an optional message limit from natural-language inbox requests."""
    lowered = text.lower()
    for pattern in (
        r"(?:process|reply to|handle)\s+(?:up to\s+)?(\d+)\s+unread",
        r"(\d+)\s+unread\s+(?:email|message|mail)",
        r"limit\s+(\d+)",
    ):
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def process_gmail_inbox_sync(limit: Optional[int] = None) -> str:
    """Run the Gmail OAuth inbox auto-reply workflow synchronously."""
    config = GmailAgentConfig.from_env()
    configure_logging(config.log_level)

    if not Path(config.client_secrets).exists():
        return (
            "Gmail inbox error: OAuth credentials not found at "
            f"{config.client_secrets}. Download a Desktop OAuth client JSON "
            "from Google Cloud Console and set GOOGLE_CLIENT_SECRETS in .env."
        )

    try:
        summary = process_unread_and_reply(limit=limit, config=config)
    except FileNotFoundError as exc:
        return f"Gmail inbox error: {exc}"
    except Exception as exc:
        return f"Gmail inbox error: {exc}"

    processed = summary.get("processed", 0)
    succeeded = summary.get("succeeded", 0)
    failed = summary.get("failed", 0)

    if processed == 0:
        return (
            "Gmail inbox: no unread messages found in the inbox. "
            "Nothing to reply to."
        )

    lines = [
        "Gmail inbox auto-reply complete.",
        f"Processed: {processed}",
        f"Replied successfully: {succeeded}",
        f"Failed: {failed}",
    ]
    if failed:
        lines.append(
            "Check that Gmail API + OAuth are configured and Ollama is running."
        )
    return "\n".join(lines)


def read_unread_gmail_sync(limit: int = 5) -> str:
    """Read unread Gmail messages and return a concise content summary."""
    config = GmailAgentConfig.from_env()
    configure_logging(config.log_level)

    if not Path(config.client_secrets).exists():
        return (
            "Gmail inbox error: OAuth credentials not found at "
            f"{config.client_secrets}. Set GOOGLE_CLIENT_SECRETS in .env."
        )

    try:
        service = get_gmail_service(config)
        refs = list_unread_messages(service, config)
    except Exception as exc:
        return f"Gmail inbox error: {exc}"

    if not refs:
        return "No unread emails found."

    lines: list[str] = [f"Unread emails found: {len(refs)}", ""]
    for index, ref in enumerate(refs[: max(1, limit)], start=1):
        msg_id = ref.get("id", "")
        if not msg_id:
            continue
        try:
            msg = get_message(service, config, msg_id)
        except Exception as exc:
            lines.append(f"{index}. Failed to read message {msg_id}: {exc}")
            lines.append("")
            continue

        body = " ".join(msg["body"].split())
        body_preview = body[:300] + ("..." if len(body) > 300 else "")
        lines.append(f"{index}. Message ID: {msg['id']}")
        lines.append(f"   Thread ID: {msg['thread_id']}")
        lines.append(f"   From: {msg['sender']}")
        lines.append(f"   Subject: {msg['subject']}")
        lines.append(f"   Body: {body_preview or '[empty body]'}")
        lines.append("")
    return "\n".join(lines).strip()


def reply_to_gmail_message_sync(message_id: str, reply_text: str = "") -> str:
    """Reply to a specific Gmail message ID within same thread."""
    config = GmailAgentConfig.from_env()
    configure_logging(config.log_level)
    if not message_id.strip():
        return "Gmail inbox error: message_id is required."

    try:
        service = get_gmail_service(config)
        original = get_message(service, config, message_id.strip())
        final_reply = reply_text.strip() or generate_reply_with_ollama(original, config)
        sent = send_reply(service, config, original, final_reply)
        if not sent:
            return f"Failed to send reply for message {message_id}."
        mark_as_read(service, config, message_id.strip())
    except Exception as exc:
        return f"Gmail inbox error: {exc}"

    return (
        "Reply sent successfully.\n"
        f"Message ID: {message_id}\n"
        f"Thread ID: {original['thread_id']}"
    )


@tool
async def process_gmail_inbox(limit: int = 0) -> str:
    """Read unread Gmail inbox messages and send AI-generated in-thread replies.

    Use when the user asks to process, read, or auto-reply to unread Gmail inbox
    messages via the Gmail API (OAuth), NOT for sending a new SMTP email report.

    Requires .env:
    - GOOGLE_CLIENT_SECRETS, GMAIL_TOKEN_FILE (Gmail API OAuth)
    - OLLAMA_URL, OLLAMA_MODEL (reply generation)

    Args:
        limit: Max unread messages to process. Use 0 to process all unread.

    Returns:
        Summary of processed, succeeded, and failed inbox replies.
    """
    effective_limit = limit if limit > 0 else None
    return await run_in_thread(process_gmail_inbox_sync, effective_limit)


@tool
async def read_unread_gmail(limit: int = 5) -> str:
    """Read unread Gmail emails and return sender/subject/body summaries."""
    return await run_in_thread(read_unread_gmail_sync, max(1, limit))


@tool
async def reply_to_gmail_message(message_id: str, reply_text: str = "") -> str:
    """Reply to one Gmail message by message_id in the same email thread."""
    return await run_in_thread(reply_to_gmail_message_sync, message_id, reply_text)


__all__ = [
    "extract_gmail_inbox_limit",
    "process_gmail_inbox",
    "process_gmail_inbox_sync",
    "read_unread_gmail",
    "read_unread_gmail_sync",
    "reply_to_gmail_message",
    "reply_to_gmail_message_sync",
]

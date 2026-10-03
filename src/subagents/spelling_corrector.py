"""Spelling Guardian — conservative Arabic/English spelling corrector subagent.

Corrects only when confident. If unsure, leaves the user's text unchanged.
Never invents new meaning, never translates, never rewrites style.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.config.settings import get_settings
from core.utils.logging_utils import get_logger

logger = get_logger("subagents.spelling_corrector")

SPELLING_SYSTEM = """You are Spelling Guardian, a conservative spelling corrector for Arabic and English.

Rules (strict):
1. Fix ONLY clear spelling mistakes / typos you are highly confident about.
2. If you are unsure about a word, leave it EXACTLY as written.
3. Do NOT translate. Do NOT change meaning. Do NOT add or remove ideas.
4. Do NOT fix grammar, style, punctuation, or dialect preference unless it is clearly a typo.
5. Keep names as-is unless the misspelling is obvious (e.g. Scheherazd → Scheherazade, شهرزاذ → شهرزاد).
6. Preserve user language mix (Arabic + English) and original word order.
7. If the whole text looks already correct, return it unchanged with confidence 1.0 and empty changes.

Return JSON only:
{
  "corrected": "text after safe spelling fixes only",
  "confidence": 0.0 to 1.0,
  "changes": [{"from": "typo", "to": "fixed"}],
  "sure": true
}
Set sure=true only when every change is a clear spelling fix.
If unsure about any proposed change, omit that change and keep the original token.
"""


@dataclass
class SpellingResult:
    """Result of a conservative spelling pass."""

    original: str
    corrected: str
    confidence: float = 1.0
    sure: bool = True
    changes: list[dict[str, str]] = field(default_factory=list)
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "confidence": self.confidence,
            "sure": self.sure,
            "changes": list(self.changes),
            "applied": self.applied,
        }


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


# High-confidence common misspellings only (Arabic + English).
# Applied only as whole-word replacements.
_SURE_FIXES: dict[str, str] = {
    # English Alf Layla / common retrieval typos
    "scheherazd": "scheherazade",
    "scheherazad": "scheherazade",
    "sheherazade": "scheherazade",
    "shahryar": "shahryar",  # already common spelling — kept
    "shahriyar": "shahryar",
    "dunyazade": "dunyazad",
    "alf layla wa lyla": "alf layla wa layla",
    "1001 night": "1001 nights",
    # Arabic common entity typos (hamza / dots / OCR-ish user typos)
    "شهرزاذ": "شهرزاد",
    "شهرزاة": "شهرزاد",
    "شهريارر": "شهريار",
    "دنيازاة": "دنيازاد",
    "قمرالزمان": "قمر الزمان",
    "الف ليله": "الف ليلة",
    "ألف ليله": "ألف ليلة",
}


def _apply_sure_dictionary(text: str) -> tuple[str, list[dict[str, str]]]:
    """Apply only whole-word / phrase fixes from the sure dictionary."""
    if not text:
        return text, []
    out = text
    changes: list[dict[str, str]] = []
    # Longer phrases first
    items = sorted(_SURE_FIXES.items(), key=lambda kv: len(kv[0]), reverse=True)
    for src, dst in items:
        if src == dst:
            continue
        # Case-insensitive for Latin; exact for Arabic
        if re.search(r"[A-Za-z]", src):
            pattern = re.compile(rf"(?i)(?<!\w){re.escape(src)}(?!\w)")
            if pattern.search(out):
                out2 = pattern.sub(dst, out)
                if out2 != out:
                    changes.append({"from": src, "to": dst})
                    out = out2
        else:
            if src in out:
                out2 = out.replace(src, dst)
                if out2 != out:
                    changes.append({"from": src, "to": dst})
                    out = out2
    return out, changes


def _is_safe_edit(original: str, corrected: str, changes: list[dict[str, str]]) -> bool:
    """Reject edits that look like rewrites rather than spelling fixes."""
    if not corrected.strip():
        return False
    if corrected.strip() == original.strip():
        return True
    # Too much length drift → likely rewrite
    o, c = original.strip(), corrected.strip()
    if abs(len(c) - len(o)) > max(12, int(0.25 * len(o)) + 1):
        return False
    # Too many token changes
    if len(changes) > 6:
        return False
    for ch in changes:
        src = (ch.get("from") or "").strip()
        dst = (ch.get("to") or "").strip()
        if not src or not dst:
            return False
        # Single-token fixes should stay similar length
        if " " not in src and " " not in dst and abs(len(src) - len(dst)) > 4:
            return False
    return True


def correct_spelling(text: str, *, use_llm: bool | None = None) -> SpellingResult:
    """Correct clear spelling mistakes only; leave uncertain tokens unchanged.

    Args:
        text: User query or message fragment.
        use_llm: Override settings; default uses ENABLE_SPELLING_CORRECTOR + LLM.
    """
    original = text or ""
    stripped = original.strip()
    if not stripped:
        return SpellingResult(original=original, corrected=original, applied=False)

    settings = get_settings()
    enabled = getattr(settings, "enable_spelling_corrector", True)
    if not enabled:
        return SpellingResult(
            original=original,
            corrected=stripped,
            confidence=1.0,
            sure=True,
            changes=[],
            applied=False,
        )

    use_llm_flag = getattr(settings, "spelling_use_llm", True) if use_llm is None else use_llm
    if use_llm_flag is False:
        fixed, changes = _apply_sure_dictionary(stripped)
        applied = fixed != stripped and bool(changes)
        return SpellingResult(
            original=original,
            corrected=fixed if applied else stripped,
            confidence=1.0,
            sure=True,
            changes=changes,
            applied=applied,
        )

    # Always apply sure dictionary first (no LLM needed).
    base, dict_changes = _apply_sure_dictionary(stripped)

    min_conf = float(getattr(settings, "spelling_min_confidence", 0.85))
    corrected = base
    confidence = 1.0
    sure = True
    llm_changes: list[dict[str, str]] = []

    try:
        from core.llm.factory import invoke_plain

        raw = invoke_plain(
            [
                {"role": "system", "content": SPELLING_SYSTEM},
                {"role": "user", "content": base},
            ]
        )
        data = _extract_json(str(raw)) or {}
        candidate = str(data.get("corrected") or base).strip() or base
        confidence = float(data.get("confidence") or 0.0)
        sure = bool(data.get("sure"))
        raw_changes = data.get("changes") or []
        if isinstance(raw_changes, list):
            for item in raw_changes:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("from") or "").strip()
                dst = str(item.get("to") or "").strip()
                if src and dst and src != dst:
                    llm_changes.append({"from": src, "to": dst})

        # Conservative gate: require sure + high confidence + safe edit shape
        if (
            sure
            and confidence >= min_conf
            and _is_safe_edit(base, candidate, llm_changes)
            and candidate != base
        ):
            corrected = candidate
        else:
            # Unsure → keep dictionary-only (or original) result
            corrected = base
            llm_changes = []
            sure = True
            confidence = 1.0 if dict_changes else confidence
    except Exception as exc:  # noqa: BLE001
        logger.warning("Spelling Guardian LLM skipped: %s", exc)
        corrected = base
        llm_changes = []

    all_changes = dict_changes + [c for c in llm_changes if c not in dict_changes]
    applied = corrected.strip() != stripped
    if applied:
        logger.info(
            "Spelling Guardian applied %s change(s): %s",
            len(all_changes),
            all_changes[:5],
        )
    return SpellingResult(
        original=original,
        corrected=corrected,
        confidence=confidence,
        sure=sure,
        changes=all_changes,
        applied=applied,
    )


def correct_query_for_retrieval(text: str) -> str:
    """Convenience: return corrected text for retrieval, or original if unchanged."""
    result = correct_spelling(text)
    return result.corrected if result.corrected.strip() else (text or "")


__all__ = [
    "SpellingResult",
    "correct_spelling",
    "correct_query_for_retrieval",
]

"""Read-only database query planner for knowledge-base retrieval.

This agent decides whether a vector-database lookup is needed and writes a
READ-only retrieval query plan. For Arabic knowledge corpora it converts the
retrieval query to Arabic before embedding/search, while the original user
question is kept for answering in the user's language.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.config.settings import get_settings
from core.llm.factory import invoke_plain
from core.prompts.planner import PLANNER_SYSTEM, TRANSLATE_SYSTEM
from core.routing.language import detect_language, has_arabic


@dataclass
class QueryPlan:
    """READ-only retrieval plan produced by the query planner agent."""

    needs_database: bool
    mode: str = "read"
    search_query: str = ""
    keywords: list[str] = field(default_factory=list)
    query_notes: list[str] = field(default_factory=list)
    boost_applied: bool = False
    reason: str = ""
    original_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan fields for logging / UI."""
        return {
            "needs_database": self.needs_database,
            "mode": "read",
            "search_query": self.search_query,
            "keywords": list(self.keywords),
            "query_notes": list(self.query_notes),
            "boost_applied": self.boost_applied,
            "reason": self.reason,
            "original_query": self.original_query,
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


def _tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text or "", flags=re.UNICODE)
    stop = {
        "a", "an", "the", "is", "are", "what", "who", "how", "does", "did", "please",
        "ما", "هل", "من", "في", "عن", "هذا", "هذه", "التي", "الذي",
    }
    out: list[str] = []
    for token in tokens:
        if token.lower() in stop or len(token) < 2:
            continue
        if token not in out:
            out.append(token)
        if len(out) >= 8:
            break
    return out


BOOSTER_SYSTEM = """You improve ambiguous retrieval queries for Arabic Alf Layla knowledge search.

Return JSON only:
{
  "should_boost": true/false,
  "notes": ["line 1", "line 2"]
}

Rules:
- Use exactly 2 short lines only when the query is short/ambiguous.
- Add disambiguating context terms and likely canonical names/entities.
- Keep meaning unchanged; do not answer the question.
- Prefer Arabic entity forms when possible (e.g., شهرزاد، شهريار، دنيازاد، مرزوان، قمر الزمان).
- If already clear, set should_boost=false and notes=[].
"""


def _needs_query_boost(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    settings = get_settings()
    min_words = max(2, int(getattr(settings, "query_booster_min_words", 6)))
    words = [w for w in re.split(r"\s+", q) if w]
    if len(words) <= min_words:
        return True
    generic = {
        "who", "what", "why", "where", "tell", "about", "story", "king", "prince",
        "من", "ما", "لماذا", "اين", "احك", "قصة", "ملك", "امير", "الوزير",
    }
    low = sum(1 for w in words if w.strip("?.!,،؟").lower() in generic)
    return low >= max(2, len(words) // 2)


def _fallback_query_notes(text: str) -> list[str]:
    t = (text or "").lower()
    if "qamar" in t or "قمر" in t or "prince" in t or "أمير" in t:
        return [
            "السياق: قصة قمر الزمان ضمن ألف ليلة وليلة.",
            "مرتبط غالبا بالوزير مرزوان ورحلة الأمير.",
        ]
    if "king" in t or "ملك" in t or "عرائس" in t or "bride" in t:
        return [
            "السياق: الحكاية الإطارية عن الملك شهريار وشهرزاد.",
            "يتعلق بالقتل بعد ليلة ثم السرد لتأجيل التنفيذ.",
        ]
    if "sister" in t or "أخت" in t or "دنيا" in t:
        return [
            "السياق: دنيازاد أخت شهرزاد في مطلع الليالي.",
            "دورها طلب متابعة الحكاية كل ليلة.",
        ]
    if "jar" in t or "jinni" in t or "عفريت" in t or "جرة" in t:
        return [
            "السياق: حكاية الصياد والعفريت المحبوس في الجرة.",
            "يرتبط بالخاتم وسرد النجاة بالحيلة.",
        ]
    return [
        "السياق: السؤال غالبا عن شخصيات ألف ليلة وليلة.",
        "استخدم أسماء الشخصيات وصيغ الحكاية الإطارية في البحث.",
    ]


def expand_ambiguous_query(text: str, *, use_llm: bool | None = None) -> tuple[str, list[str], bool]:
    """Append two short context notes for ambiguous user queries."""
    q = (text or "").strip()
    if not q:
        return "", [], False
    settings = get_settings()
    if not getattr(settings, "enable_query_booster", True):
        return q, [], False
    if not _needs_query_boost(q):
        return q, [], False

    use_llm_flag = getattr(settings, "query_booster_use_llm", True) if use_llm is None else use_llm
    notes: list[str] = []
    if use_llm_flag:
        try:
            raw = invoke_plain(
                [
                    {"role": "system", "content": BOOSTER_SYSTEM},
                    {"role": "user", "content": q},
                ]
            )
            data = _extract_json(str(raw)) or {}
            if bool(data.get("should_boost")):
                raw_notes = data.get("notes") or []
                if isinstance(raw_notes, list):
                    notes = [str(n).strip() for n in raw_notes if str(n).strip()][:2]
        except Exception:  # noqa: BLE001
            notes = []
    if not notes:
        notes = _fallback_query_notes(q)

    notes = [n.replace("\n", " ").strip()[:160] for n in notes if n.strip()][:2]
    if len(notes) < 2:
        return q, [], False
    expanded = f"{q}\n{notes[0]}\n{notes[1]}"
    return expanded, notes, True


def to_arabic_retrieval_query(question: str) -> str:
    """Convert a user question into an Arabic retrieval query for embedding/search."""
    text = (question or "").strip()
    if not text:
        return ""
    if detect_language(text) == "ar" and has_arabic(text):
        return text
    try:
        translated = str(
            invoke_plain(
                [
                    {"role": "system", "content": TRANSLATE_SYSTEM},
                    {"role": "user", "content": text},
                ]
            )
        ).strip()
        # Keep first line only; strip wrapping quotes.
        translated = translated.splitlines()[0].strip().strip('"').strip("'")
        if has_arabic(translated):
            return translated
    except Exception:  # noqa: BLE001
        pass
    return text


def _heuristic_plan(question: str, *, notes: list[str] | None = None, boosted: bool = False) -> QueryPlan:
    text = (question or "").strip()
    lowered = text.lower()
    small_talk = (
        "hello", "hi", "hey", "thanks", "thank you", "who are you", "what can you do",
        "مرحبا", "السلام", "شكرا", "من أنت", "من انت", "ماذا تستطيع",
    )
    if not text or any(marker in lowered or marker in text for marker in small_talk):
        if len(text.split()) <= 6:
            return QueryPlan(
                needs_database=False,
                search_query="",
                reason="Small talk / greeting — no database read needed",
                original_query=text,
            )
    arabic_query = to_arabic_retrieval_query(text)
    if notes:
        arabic_query = f"{arabic_query}\n" + "\n".join(notes[:2])
    return QueryPlan(
        needs_database=True,
        search_query=arabic_query,
        keywords=_tokenize_keywords(arabic_query),
        query_notes=list(notes or []),
        boost_applied=boosted,
        reason="Content question — Arabic retrieval query for embedding/search",
        original_query=text,
    )


def plan_database_query(
    question: str,
    *,
    skip_spelling: bool = False,
    skip_boost: bool = False,
) -> QueryPlan:
    """Ask the planner LLM for a READ-only Arabic retrieval plan."""
    text = (question or "").strip()
    if not text:
        return QueryPlan(needs_database=False, reason="Empty question", original_query="")

    # Spelling Guardian: only high-confidence typos; unsure tokens left as-is.
    if not skip_spelling:
        from subagents.spelling_corrector import correct_spelling

        spell = correct_spelling(text)
        text = spell.corrected.strip() or text

    notes: list[str] = []
    boosted = False
    if not skip_boost:
        boosted_text, notes, boosted = expand_ambiguous_query(text)
        text = boosted_text

    try:
        raw = invoke_plain(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": text},
            ]
        )
        data = _extract_json(str(raw))
        if not data:
            return _heuristic_plan(text, notes=notes, boosted=boosted)
        needs = bool(data.get("needs_database"))
        search_query = str(data.get("search_query") or "").strip()
        if needs and not has_arabic(search_query):
            search_query = to_arabic_retrieval_query(text)
        if needs and notes:
            search_query = f"{search_query}\n" + "\n".join(notes[:2])
        keywords_raw = data.get("keywords") or []
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()][:8]
        if needs and keywords and not any(has_arabic(k) for k in keywords):
            keywords = _tokenize_keywords(search_query)
        if needs and notes:
            for token in _tokenize_keywords(" ".join(notes)):
                if token not in keywords:
                    keywords.append(token)
                if len(keywords) >= 12:
                    break
        reason = str(data.get("reason") or "").strip()
        if boosted and notes:
            boost_reason = " + query booster notes for ambiguity"
        else:
            boost_reason = ""
        return QueryPlan(
            needs_database=needs,
            mode="read",
            search_query=search_query if needs else "",
            keywords=keywords if needs else [],
            query_notes=list(notes if needs else []),
            boost_applied=bool(boosted and needs),
            reason=reason
            or (
                f"READ-only Arabic retrieval plan{boost_reason}"
                if needs
                else "No DB needed"
            ),
            original_query=text,
        )
    except Exception:  # noqa: BLE001
        return _heuristic_plan(text, notes=notes, boosted=boosted)

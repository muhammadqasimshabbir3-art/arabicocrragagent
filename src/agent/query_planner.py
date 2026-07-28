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

from llm.factory import invoke_plain
from routing.language import detect_language, has_arabic

PLANNER_SYSTEM = """You are a READ-ONLY database query planner for an Arabic document knowledge base.

Your job:
1. Decide if answering the user requires retrieving evidence from the vector database.
2. If yes, write a READ-only search query plan.
3. Never invent write/update/delete/insert operations. Queries are ALWAYS read-only.

IMPORTANT — the knowledge corpus is Arabic:
- Convert the retrieval query to clear Modern Standard Arabic for embedding + search.
- Translate proper names carefully and keep their meaning accurate.
- keywords must be Arabic terms useful for BM25 lexical matching.
- Keep the meaning of the user's question (English or Arabic).
- Do not specialize for any particular book, character, or domain — stay general.

Return ONLY valid JSON with this schema:
{
  "needs_database": true|false,
  "mode": "read",
  "search_query": "Arabic retrieval query for embedding/search",
  "keywords": ["كلمة1", "كلمة2"],
  "reason": "short reason"
}

Rules:
- mode MUST always be "read"
- If greeting / thanks / who are you / small talk: needs_database=false
- If question is about book/document content, characters, topics, authors, plots, facts: needs_database=true
- search_query MUST be Arabic when needs_database=true
- keywords: 2-8 Arabic terms
"""

TRANSLATE_SYSTEM = """Translate the user question into a concise Modern Standard Arabic
search query for retrieving passages from an Arabic document corpus.
Preserve proper names accurately (do not confuse similar names).
Return ONLY the Arabic query text, no quotes, no explanation."""


@dataclass
class QueryPlan:
    """READ-only retrieval plan produced by the query planner agent."""

    needs_database: bool
    mode: str = "read"
    search_query: str = ""
    keywords: list[str] = field(default_factory=list)
    reason: str = ""
    original_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_database": self.needs_database,
            "mode": "read",
            "search_query": self.search_query,
            "keywords": list(self.keywords),
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


def _heuristic_plan(question: str) -> QueryPlan:
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
    return QueryPlan(
        needs_database=True,
        search_query=arabic_query,
        keywords=_tokenize_keywords(arabic_query),
        reason="Content question — Arabic retrieval query for embedding/search",
        original_query=text,
    )


def plan_database_query(question: str) -> QueryPlan:
    """Ask the planner LLM for a READ-only Arabic retrieval plan."""
    text = (question or "").strip()
    if not text:
        return QueryPlan(needs_database=False, reason="Empty question", original_query="")

    try:
        raw = invoke_plain(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": text},
            ]
        )
        data = _extract_json(str(raw))
        if not data:
            return _heuristic_plan(text)
        needs = bool(data.get("needs_database"))
        search_query = str(data.get("search_query") or "").strip()
        if needs and not has_arabic(search_query):
            search_query = to_arabic_retrieval_query(text)
        keywords_raw = data.get("keywords") or []
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()][:8]
        if needs and keywords and not any(has_arabic(k) for k in keywords):
            keywords = _tokenize_keywords(search_query)
        reason = str(data.get("reason") or "").strip()
        return QueryPlan(
            needs_database=needs,
            mode="read",
            search_query=search_query if needs else "",
            keywords=keywords if needs else [],
            reason=reason
            or (
                "READ-only Arabic retrieval plan"
                if needs
                else "No DB needed"
            ),
            original_query=text,
        )
    except Exception:  # noqa: BLE001
        return _heuristic_plan(text)

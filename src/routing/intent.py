"""Route selection for the document intelligence graph."""

from __future__ import annotations

import re

from langchain_core.messages import AnyMessage, HumanMessage

from routing.language import detect_language
from state.schema import AgentRoute


def get_latest_user_text(messages: list[AnyMessage]) -> str:
    """Return the most recent human message text."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def is_fresh_user_turn(messages: list[AnyMessage]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, HumanMessage) or getattr(last, "type", None) == "human"


def wants_summary(text: str) -> bool:
    lowered = (text or "").lower()
    arabic_markers = ("لخص", "ملخص", "اختصر", "خلاصة")
    english_markers = ("summarize", "summary", "sum up", "tl;dr", "tldr")
    return any(marker in lowered for marker in english_markers) or any(
        marker in text for marker in arabic_markers
    )


def looks_like_small_talk(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    markers = (
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
        "who are you",
        "what can you do",
        "مرحبا",
        "السلام عليكم",
        "أهلا",
        "اهلا",
        "شكرا",
        "من أنت",
        "من انت",
        "كيف حالك",
    )
    if any(marker in lowered or marker in raw for marker in markers):
        return len(raw.split()) <= 8
    return False


def wants_knowledge_lookup(text: str) -> bool:
    """True when the user asks a content question that may need the knowledge DB."""
    raw = (text or "").strip()
    if not raw or looks_like_small_talk(raw):
        return False
    if "?" in raw or "؟" in raw:
        return True
    question_starters = (
        "what", "who", "where", "when", "why", "how", "tell", "explain",
        "ما", "من", "أين", "اين", "متى", "لماذا", "كيف", "هل", "أخبر", "اشرح",
        "عن", "موضوع",
    )
    lowered = raw.lower()
    if any(lowered.startswith(s) or raw.startswith(s) for s in question_starters):
        return True
    return len(raw.split()) >= 4


def wants_web_lookup(text: str) -> bool:
    """True when user likely needs external/public web information.

    Keep this generic: do not hardcode platform names.
    """
    raw = (text or "").strip()
    if not raw or looks_like_small_talk(raw):
        return False
    lowered = raw.lower()
    # Strong direct web verbs/signals.
    direct_markers = (
        "search the web",
        "search online",
        "look up online",
        "browse",
        "google",
        "internet",
        "online",
        "ابحث في الويب",
        "ابحث على الإنترنت",
        "ابحث اونلاين",
        "تصفح",
        "الانترنت",
        "أونلاين",
    )
    if any(marker in lowered or marker in raw for marker in direct_markers):
        return True

    # Generic public-profile/public-work intent (platform-agnostic).
    profile_terms_en = (
        "profile",
        "account",
        "handle",
        "username",
        "portfolio",
        "page",
        "channel",
        "public repos",
        "public repository",
        "public repositories",
        "public work",
    )
    profile_terms_ar = (
        "الملف الشخصي",
        "ملف شخصي",
        "حساب",
        "اسم المستخدم",
        "معرف",
        "رابط",
        "اعماله",
        "أعماله",
        "المشاريع العامة",
        "المستودعات العامة",
        "صفحته",
        "القناة",
    )
    ask_verbs_en = ("find", "get", "show", "provide", "give", "fetch", "locate", "summarize")
    ask_verbs_ar = ("هات", "اعطني", "أعطني", "زودني", "زوّدني", "اعرض", "لخص", "لخّص")
    has_profile_term = any(t in lowered for t in profile_terms_en) or any(
        t in raw for t in profile_terms_ar
    )
    has_ask_verb = any(v in lowered for v in ask_verbs_en) or any(v in raw for v in ask_verbs_ar)
    if has_profile_term and (has_ask_verb or "?" in raw or "؟" in raw):
        return True

    # Domain/link patterns usually imply external browsing.
    if re.search(r"(?i)\b[a-z0-9-]+\.[a-z]{2,}\b", raw):
        return True

    # Explicit "search/browse for X" phrasing.
    if re.search(r"(?i)\b(search|browse|look up)\b.+\b(web|online|internet)\b", lowered):
        return True
    if re.search(r"(?i)\b(latest|current|today)\b.+\b(news|price|score|weather)\b", lowered):
        return True
    if re.search(r"(?i)\b(link|url)\b.+\b(profile|account|page)\b", lowered):
        return True
    return False


def pick_route(
    user_text: str,
    messages: list[AnyMessage],
    *,
    has_document: bool,
    summarize_only: bool = False,
    knowledge_available: bool = False,
) -> AgentRoute:
    """Choose ingest / query / summarize / knowledge / web / chat.

    PDF upload path and knowledge-base chat are separate:
    - If a document is attached, ALWAYS use the PDF path
      (ingest / query_documents / summarize_document).
    - Knowledge DB (query_knowledge_base) is used only when no file is attached.
    - Web search is used when the question needs live/online info.
    """
    if not is_fresh_user_turn(messages) and not has_document:
        return "call_model"

    # Path A — uploaded PDF/image (independent from knowledge DB).
    if has_document:
        if wants_web_lookup(user_text):
            return "web_search"
        if summarize_only or wants_summary(user_text) or not (user_text or "").strip():
            return "summarize_document"
        return "query_documents"

    # Path B — live web / browser lookup when explicitly needed.
    if wants_web_lookup(user_text):
        return "web_search"

    # Path C — no upload: READ-only knowledge DB vs general chat.
    _ = detect_language(user_text)
    if knowledge_available and wants_knowledge_lookup(user_text):
        return "query_knowledge_base"
    return "call_model"

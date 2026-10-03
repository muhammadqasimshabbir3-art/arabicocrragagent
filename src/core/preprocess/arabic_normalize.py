"""Arabic / Unicode text normalization for RAG."""

from __future__ import annotations

import re
import unicodedata

# Arabic presentation forms / common ligature-ish cleanups.
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_WHITESPACE = re.compile(r"[ \t\f\v]+")
_NEWLINES = re.compile(r"\n{3,}")
_INVISIBLE_MARKS = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
# OCR/PDF sometimes emits Arabic letters split by spaces: "ا ل ت ع ل ي م".
_SPACED_ARABIC_WORD = re.compile(r"(?:[\u0600-\u06FF]\s+){2,}[\u0600-\u06FF]")
_ARABIC_TOKEN = re.compile(r"^[\u0600-\u06FF]+$")
_HAS_ARABIC = re.compile(r"[\u0600-\u06FF]")
_COMMON_PREFIXES = ("ال", "وال", "بال", "لل", "كال", "فال")
_COMMON_SUFFIXES = (
    "ة",
    "ات",
    "ون",
    "ين",
    "ان",
    "ية",
    "ه",
    "ها",
    "هم",
    "كما",
    "نا",
    "ي",
)
_COMMON_FRAGMENTS = ("ال", "لل", "ون", "ات", "ية", "في", "من", "ان")
_COMMON_BIGRAMS = {
    "ال",
    "لل",
    "من",
    "ان",
    "في",
    "ية",
    "ات",
    "ون",
    "ين",
    "ما",
    "ري",
    "را",
    "با",
    "يا",
    "ية",
    "تح",
    "عي",
    "ذك",
    "اص",
    "وا",
    "مه",
    "ند",
    "عل",
    "تع",
}


def _collapse_spaced_arabic_letters(text: str) -> str:
    """Join Arabic letters when OCR inserted spaces between every character."""

    def _join(match: re.Match[str]) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return _SPACED_ARABIC_WORD.sub(_join, text)


def _reverse_arabic_token_if_needed(token: str) -> str:
    if _ARABIC_TOKEN.fullmatch(token):
        return token[::-1]
    return token


def _arabic_shape_score(text: str) -> int:
    """Score how naturally Arabic tokens are shaped (language-agnostic heuristic)."""
    score = 0
    for token in re.findall(r"[\u0600-\u06FF]+", text):
        if len(token) < 2:
            continue
        if token.startswith(_COMMON_PREFIXES):
            score += 2
        if token.endswith(_COMMON_SUFFIXES):
            score += 2
        score += sum(1 for frag in _COMMON_FRAGMENTS if frag in token)
        bigrams = [token[i : i + 2] for i in range(len(token) - 1)]
        score += sum(1 for bg in bigrams if bg in _COMMON_BIGRAMS)
        # Penalize very odd letter runs common in visually reversed extraction.
        if token.startswith(("ة", "ى", "ؤ", "ئ")):
            score -= 2
        if token.endswith(("ا", "و")) and len(token) > 3:
            score -= 1
    return score


def _repair_visual_order_arabic(text: str) -> str:
    """Fix PDFs where Arabic letters are extracted in visual (reversed) order."""
    if not text or not _HAS_ARABIC.search(text):
        return text
    parts = re.split(r"(\s+)", text)
    reversed_parts = [_reverse_arabic_token_if_needed(part) for part in parts]
    candidate = "".join(reversed_parts)
    # Apply only when it clearly improves readability.
    before = _arabic_shape_score(text)
    after = _arabic_shape_score(candidate)
    if after >= 4 and after >= before + 3:
        return candidate
    return text


def normalize_arabic(
    text: str,
    *,
    remove_diacritics: bool = False,
    normalize_alef: bool = True,
    normalize_ya: bool = True,
    normalize_teh_marbuta: bool = True,
    remove_tatweel: bool = True,
) -> str:
    """Normalize Arabic text for retrieval while preserving meaning."""
    if not text:
        return ""

    # NFKC collapses Arabic presentation forms into canonical Arabic letters.
    value = unicodedata.normalize("NFKC", text)

    if remove_tatweel:
        value = value.replace(_TATWEEL, "")

    if normalize_alef:
        # أ إ آ ٱ → ا
        value = re.sub("[إأآٱ]", "ا", value)

    if normalize_ya:
        # ى → ي
        value = value.replace("ى", "ي")

    if normalize_teh_marbuta:
        # ة → ه (retrieval-oriented; original forms kept in OCR metadata path)
        value = value.replace("ة", "ه")

    if remove_diacritics:
        value = _ARABIC_DIACRITICS.sub("", value)

    value = _INVISIBLE_MARKS.sub("", value)
    value = _collapse_spaced_arabic_letters(value)
    value = _repair_visual_order_arabic(value)
    value = _WHITESPACE.sub(" ", value)
    value = _NEWLINES.sub("\n\n", value)
    return value.strip()


def clean_ocr_text(text: str, *, for_retrieval: bool = True) -> str:
    """Post-process OCR output for indexing."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _INVISIBLE_MARKS.sub("", value)
    value = _collapse_spaced_arabic_letters(value)
    value = _repair_visual_order_arabic(value)
    value = _WHITESPACE.sub(" ", value)
    value = _NEWLINES.sub("\n\n", value)
    value = value.strip()
    if for_retrieval:
        # Keep diacritics in displayed text; normalize letters for consistency.
        return normalize_arabic(
            value,
            remove_diacritics=False,
            normalize_alef=True,
            normalize_ya=True,
            normalize_teh_marbuta=False,
            remove_tatweel=True,
        )
    return value


def looks_unreliable_arabic_extraction(text: str) -> bool:
    """Heuristic for low-quality Arabic extraction that should trigger OCR fallback."""
    value = unicodedata.normalize("NFKC", text or "")
    if not _HAS_ARABIC.search(value):
        return False
    # Many spaced-letter runs usually means broken extraction order/layout.
    spaced_runs = len(_SPACED_ARABIC_WORD.findall(value))
    normalized = clean_ocr_text(value, for_retrieval=True)
    tokens = re.findall(r"[\u0600-\u06FF]+", normalized)
    if not tokens:
        return True
    single_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    return spaced_runs >= 1 or single_ratio > 0.35 or avg_len < 2.2


def dedupe_repeated_lines(text: str) -> str:
    """Remove consecutive duplicate lines (common OCR header/footer noise)."""
    lines = (text or "").splitlines()
    out: list[str] = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            continue
        out.append(line)
        prev = stripped or prev
    return "\n".join(out).strip()

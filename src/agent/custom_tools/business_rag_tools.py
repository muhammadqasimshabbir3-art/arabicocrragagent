"""Business knowledge RAG over semi-structured Neon documents.

Source of truth: Neon tables `business_documents` + `business_chunks`.
Retrieval is hybrid:
  1) Traditional keyword IR — BM25, exact phrases, title/tag matching
  2) Semantic similarity — local BGE embeddings (BAAI/bge-small-en-v1.5)

Pipeline (reliable answers + citations):
  1) Hybrid retrieve top 5 passages
  2) Trust layer — LLM reads EACH of the 5, marks RELEVANT + quotes
  3) Answer only from RELEVANT quotes
  4) Append all top-5 Sources (USED vs reviewed) for the user

Large docs (2–3 MB) are chunked before embedding; the embedder scores chunks,
not whole files.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from langchain.tools import tool

from agent.async_utils import run_in_thread
from agent.custom_tools.database_tools import _build_database_url
from agent.embeddings import embed_query

load_dotenv()

TOP_K = 5
# Larger chunks suit multi-MB policy / SOP documents while staying under BGE's window.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Hybrid fusion weights (must sum ~1.0). Semantic weight raised for neural BGE.
WEIGHT_BM25 = 0.35
WEIGHT_SEMANTIC = 0.40
WEIGHT_PHRASE = 0.15
WEIGHT_FIELD = 0.10

BM25_K1 = 1.5
BM25_B = 0.75

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "our",
        "we",
        "you",
        "your",
        "what",
        "which",
        "who",
        "how",
        "do",
        "does",
        "did",
        "can",
        "with",
        "from",
        "that",
        "this",
        "it",
        "as",
        "at",
        "by",
        "if",
        "about",
        "me",
        "my",
        "please",
        "tell",
    }
)

BUSINESS_RAG_KEYWORDS = (
    "policy",
    "policies",
    "return policy",
    "refund",
    "warranty",
    "shipping policy",
    "how do we",
    "what is our",
    "sop",
    "procedure",
    "handbook",
    "faq",
    "knowledge base",
    "business document",
    "company policy",
    "exchange",
    "damaged product",
    "delivery time",
    "opening hours",
    "store hours",
    "customer support",
    "code of conduct",
    "sales brief",
    "product care",
    "rag",
    "semi-structured",
    # Customer how-to / after-sales (must NOT go to store SQL)
    "replace a product",
    "replace the product",
    "replace product",
    "want to replace",
    "return a product",
    "return the product",
    "want to return",
    "want to exchange",
    "exchange a product",
    "what should i do",
    "how do i return",
    "how do i replace",
    "how can i return",
    "how can i replace",
    "gift card",
    "store credit",
    "price match",
)

_SOURCE_ANALYSIS_PROMPT = """You are a careful Solar Store document analyst (trust layer).
You are given exactly the top retrieved sources (up to 5). Read EVERY numbered
source [1]…[5] carefully. Do NOT invent facts. Do NOT skip any source.

For EACH source write exactly this block:
### [n] <title>
- Verdict: RELEVANT or NOT_RELEVANT
- Quotes: copy exact phrases from that source that help answer the question (or "none")
- Why: one short sentence

Rules:
- Prefer RELEVANT if any sentence could answer part of the question.
- Quotes must be copied from the source text (light truncation OK).
- Do not write the final user-facing answer yet.
"""

_ANSWER_FROM_ANALYSIS_PROMPT = """You answer Solar Store business-knowledge questions.
You already have a trust-layer analysis of the top retrieved sources (up to 5).
Use ONLY sources marked RELEVANT and their Quotes.
If nothing is RELEVANT, say the retrieved documents do not contain a clear answer.

Write a clear answer for the user:
- Cite sources inline as [1], [2], … matching the analysis numbers.
- Prefer quoting or paraphrasing the Quotes (do not invent policy details).
- If several RELEVANT sources conflict, say so and present both.
- Do NOT write a Sources section or a Source analysis section — those are appended
  automatically so the user can see all top-5 retrieved documents.
"""


def needs_business_rag(text: str) -> bool:
    """Detect knowledge / policy questions that should use business RAG."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    return any(keyword in lowered for keyword in BUSINESS_RAG_KEYWORDS)


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = max(
                normalized.rfind("\n\n", start, end),
                normalized.rfind(". ", start, end),
                normalized.rfind(" ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed (classic IR)."""
    tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _extract_phrases(query: str) -> list[str]:
    """Build useful multi-word phrases from the query for exact matching."""
    lowered = (query or "").lower().strip()
    phrases: list[str] = []
    phrases.extend(re.findall(r'"([^"]{2,80})"', lowered))
    raw = re.findall(r"[a-z0-9_]+", lowered)
    for size in (3, 2):
        for index in range(len(raw) - size + 1):
            phrase = " ".join(raw[index : index + size])
            if phrase not in _STOPWORDS:
                phrases.append(phrase)
    for known in (
        "return policy",
        "refund policy",
        "shipping policy",
        "warranty coverage",
        "opening hours",
        "store hours",
        "product care",
        "code of conduct",
        "sales brief",
        "customer support",
    ):
        if known in lowered:
            phrases.append(known)
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase in seen or len(phrase) < 3:
            continue
        seen.add(phrase)
        unique.append(phrase)
    return unique


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _parse_embedding(raw: Any) -> list[float]:
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        return [float(x) for x in json.loads(raw)]
    if isinstance(raw, memoryview):
        return [float(x) for x in json.loads(raw.tobytes().decode("utf-8"))]
    return [float(x) for x in list(raw)]


def _normalize_scores(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]."""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _bm25_scores(
    query_tokens: list[str],
    documents_tokens: list[list[str]],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Okapi BM25 scores for each document given a query."""
    doc_count = len(documents_tokens)
    if doc_count == 0 or not query_tokens:
        return [0.0] * doc_count

    doc_lens = [len(tokens) or 1 for tokens in documents_tokens]
    avgdl = sum(doc_lens) / doc_count
    df: Counter[str] = Counter()
    for tokens in documents_tokens:
        df.update(set(tokens))

    idf: dict[str, float] = {}
    for term in set(query_tokens):
        freq = df.get(term, 0)
        idf[term] = math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))

    scores: list[float] = []
    for tokens, doc_len in zip(documents_tokens, doc_lens, strict=True):
        tf = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            term_freq = tf[term]
            denom = term_freq + k1 * (1.0 - b + b * doc_len / avgdl)
            score += idf.get(term, 0.0) * (term_freq * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


def _phrase_score(query: str, title: str, content: str) -> float:
    """Exact / near-exact phrase matches (traditional string matching)."""
    haystack = f"{title}\n{content}".lower()
    phrases = _extract_phrases(query)
    if not phrases:
        return 0.0
    hits = 0.0
    for phrase in phrases:
        if phrase in haystack:
            hits += 1.0 + 0.25 * max(0, len(phrase.split()) - 1)
    return hits


def _field_keyword_score(
    query_tokens: list[str],
    *,
    title: str,
    doc_type: str,
    tags: list[str] | None,
    content: str,
) -> float:
    """Traditional fielded keyword matching (title > tags > type > body)."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(title))
    type_tokens = set(_tokenize(doc_type))
    tag_tokens = set(_tokenize(" ".join(tags or [])))
    body_tokens = set(_tokenize(content))

    score = 0.0
    for term in query_tokens:
        if term in title_tokens:
            score += 3.0
        if term in tag_tokens:
            score += 2.0
        if term in type_tokens:
            score += 1.5
        if term in body_tokens:
            score += 1.0
    return score


def score_chunk_candidates(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score in-memory chunk candidates with hybrid BM25 + semantic + keywords."""
    if not candidates:
        return []

    query_tokens = _tokenize(query)
    query_vec = embed_query(query)
    docs_tokens = [
        _tokenize(
            f"{c.get('title', '')} {c.get('doc_type', '')} "
            f"{' '.join(c.get('tags') or [])} {c.get('content', '')}"
        )
        for c in candidates
    ]
    bm25_raw = _bm25_scores(query_tokens, docs_tokens)
    semantic_raw = [
        _cosine_similarity(query_vec, c.get("embedding") or []) for c in candidates
    ]
    phrase_raw = [
        _phrase_score(query, str(c.get("title", "")), str(c.get("content", "")))
        for c in candidates
    ]
    field_raw = [
        _field_keyword_score(
            query_tokens,
            title=str(c.get("title", "")),
            doc_type=str(c.get("doc_type", "")),
            tags=list(c.get("tags") or []),
            content=str(c.get("content", "")),
        )
        for c in candidates
    ]

    bm25_n = _normalize_scores(bm25_raw)
    semantic_n = _normalize_scores(semantic_raw)
    phrase_n = _normalize_scores(phrase_raw)
    field_n = _normalize_scores(field_raw)

    scored: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        total = (
            WEIGHT_BM25 * bm25_n[index]
            + WEIGHT_SEMANTIC * semantic_n[index]
            + WEIGHT_PHRASE * phrase_n[index]
            + WEIGHT_FIELD * field_n[index]
        )
        scored.append(
            {
                **{k: v for k, v in candidate.items() if k != "embedding"},
                "score": round(total, 4),
                "score_breakdown": {
                    "bm25": round(bm25_n[index], 4),
                    "semantic": round(semantic_n[index], 4),
                    "phrase": round(phrase_n[index], 4),
                    "field_keywords": round(field_n[index], 4),
                },
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def retrieve_business_chunks(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Retrieve top-k chunks using hybrid BM25 + keyword + semantic ranking."""
    import psycopg

    url = _build_database_url()
    with psycopg.connect(url, connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  c.id,
                  c.content,
                  c.embedding,
                  c.metadata,
                  d.title,
                  d.doc_type,
                  d.metadata AS doc_metadata,
                  d.tags
                FROM business_chunks c
                JOIN business_documents d ON d.id = c.document_id
                """
            )
            rows = cur.fetchall()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        (
            chunk_id,
            content,
            embedding_raw,
            chunk_meta,
            title,
            doc_type,
            doc_meta,
            tags,
        ) = row
        candidates.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "embedding": _parse_embedding(embedding_raw),
                "title": title,
                "doc_type": doc_type,
                "tags": list(tags or []),
                "chunk_metadata": chunk_meta if isinstance(chunk_meta, dict) else {},
                "doc_metadata": doc_meta if isinstance(doc_meta, dict) else {},
            }
        )

    ranked = score_chunk_candidates(query, candidates)
    return ranked[: max(1, top_k)]


def format_retrieved_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the LLM."""
    if not chunks:
        return "No matching business documents were found."
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.get("doc_metadata") or {}
        meta_bits = ", ".join(f"{k}={v}" for k, v in list(meta.items())[:6])
        breakdown = chunk.get("score_breakdown") or {}
        breakdown_bits = ", ".join(f"{k}={v}" for k, v in breakdown.items())
        tags = chunk.get("tags") or []
        tag_bits = f" tags={','.join(tags)}" if tags else ""
        parts.append(
            f"[{index}] {chunk.get('title')} ({chunk.get('doc_type')})"
            f" score={chunk.get('score')}"
            + (f" [{breakdown_bits}]" if breakdown_bits else "")
            + (f" | {meta_bits}" if meta_bits else "")
            + tag_bits
            + f"\n{chunk.get('content')}"
        )
    return "\n\n".join(parts)


def format_rag_sources(
    chunks: list[dict[str, Any]],
    *,
    relevant_indexes: set[int] | None = None,
) -> str:
    """Build a mandatory Sources footer listing every top retrieved passage.

    Always lists all retrieved chunks (typically top 5) so the user can see
    where the answer came from. Marks which ones the trust layer used.
    """
    if not chunks:
        return (
            "Sources:\n"
            "- Neon database `neondb` → tables `business_documents`, `business_chunks`\n"
            "- No matching document passages were retrieved."
        )

    lines = [
        f"Sources (top {len(chunks)} retrieved — model read each before answering):",
        "- Neon database `neondb` → tables `business_documents`, `business_chunks`",
    ]
    for index, chunk in enumerate(chunks, start=1):
        title = str(chunk.get("title") or "Untitled").strip()
        doc_type = str(chunk.get("doc_type") or "document").strip()
        meta = chunk.get("doc_metadata") or {}
        extras: list[str] = [f"type={doc_type}"]
        for field in ("department", "version", "effective_date"):
            if meta.get(field) is not None:
                extras.append(f"{field}={meta[field]}")
        score = chunk.get("score")
        if score is not None:
            extras.append(f"relevance={score}")
        if relevant_indexes is not None:
            extras.append("USED" if index in relevant_indexes else "reviewed")
        lines.append(f"- [{index}] {title} ({', '.join(extras)})")
    return "\n".join(lines)


def _strip_existing_sources_footer(text: str) -> str:
    """Remove trailing Sources / Source analysis blocks so we can re-append cleanly."""
    cleaned = (text or "").strip()
    cleaned = re.sub(
        r"\n*-{0,3}\s*\n?(?:Source analysis \(trust layer\):|Sources:|_RAG sources:_).*\Z",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def _parse_relevant_source_indexes(analysis: str, chunk_count: int) -> set[int]:
    """Best-effort: which [n] blocks were marked RELEVANT in the trust-layer analysis."""
    relevant: set[int] = set()
    if not analysis or chunk_count <= 0:
        return relevant
    pattern = re.compile(
        r"###\s*\[(\d+)\][\s\S]*?Verdict:\s*(RELEVANT|NOT_RELEVANT)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(analysis):
        index = int(match.group(1))
        if 1 <= index <= chunk_count and match.group(2).upper() == "RELEVANT":
            relevant.add(index)
    return relevant


def analyze_retrieved_sources_sync(
    user_text: str,
    chunks: list[dict[str, Any]],
    llm_invoke,
) -> str:
    """Pass 1 — force the LLM to read each retrieved source before answering."""
    if not chunks:
        return "No sources retrieved."
    context = format_retrieved_context(chunks)
    return str(
        llm_invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_SOURCE_ANALYSIS_PROMPT}\n\n"
                        f"User question:\n{user_text}\n\n"
                        f"Retrieved sources:\n{context}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze every source for this question. "
                        "Mark RELEVANT or NOT_RELEVANT and quote evidence."
                    ),
                },
            ]
        )
    ).strip()


def answer_from_source_analysis_sync(
    user_text: str,
    analysis: str,
    chunks: list[dict[str, Any]],
    llm_invoke,
) -> str:
    """Pass 2 — answer only from the trust-layer analysis of retrieved sources."""
    titles = ", ".join(
        f"[{i}] {c.get('title')}" for i, c in enumerate(chunks, start=1)
    )
    return str(
        llm_invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_ANSWER_FROM_ANALYSIS_PROMPT}\n\n"
                        f"Source index map: {titles}\n\n"
                        f"Source analysis:\n{analysis}"
                    ),
                },
                {"role": "user", "content": user_text},
            ]
        )
    ).strip()


def format_source_analysis_section(
    analysis: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Compact trust-layer section shown to the user under the answer."""
    relevant = _parse_relevant_source_indexes(analysis, len(chunks))
    if relevant:
        used = ", ".join(
            f"[{i}] {chunks[i - 1].get('title')}" for i in sorted(relevant)
        )
        header = f"Source analysis (trust layer) — used: {used}"
    else:
        header = "Source analysis (trust layer)"
    body = (analysis or "").strip()
    if len(body) > 3500:
        body = body[:3500].rstrip() + "\n… (analysis truncated)"
    return f"{header}:\n{body}"


def answer_business_rag_sync(user_text: str, llm_invoke) -> str:
    """Retrieve top-5 → read/judge each → answer from RELEVANT only + list Sources."""
    chunks = retrieve_business_chunks(user_text, top_k=TOP_K)
    if not chunks:
        return (
            "I could not find matching business documents in Neon for that question.\n\n"
            f"{format_rag_sources([])}"
        )

    analysis = analyze_retrieved_sources_sync(user_text, chunks, llm_invoke)
    answer = answer_from_source_analysis_sync(user_text, analysis, chunks, llm_invoke)
    body = _strip_existing_sources_footer(answer)
    relevant = _parse_relevant_source_indexes(analysis, len(chunks))
    return (
        f"{body}\n\n"
        f"---\n{format_source_analysis_section(analysis, chunks)}\n\n"
        f"{format_rag_sources(chunks, relevant_indexes=relevant)}"
    )


@tool
async def business_knowledge_rag(query: str) -> str:
    """Retrieve semi-structured business documents from Neon and return RAG context.

    Uses hybrid retrieval: BM25 + exact phrase/title/tag keyword matching +
    BGE semantic embeddings (BAAI/bge-small-en-v1.5). Use for policies, warranties,
    FAQs, SOPs, product care, shipping rules — NOT for live stock/order totals
    (use query_store_database).

    Args:
        query: The user's natural-language business knowledge question.

    Returns:
        Top matching document passages with titles and scores.
    """

    def _run() -> str:
        chunks = retrieve_business_chunks(query, top_k=TOP_K)
        return format_retrieved_context(chunks)

    return await run_in_thread(_run)


__all__ = [
    "BUSINESS_RAG_KEYWORDS",
    "analyze_retrieved_sources_sync",
    "answer_business_rag_sync",
    "answer_from_source_analysis_sync",
    "business_knowledge_rag",
    "format_rag_sources",
    "format_retrieved_context",
    "format_source_analysis_section",
    "needs_business_rag",
    "retrieve_business_chunks",
    "score_chunk_candidates",
    "_bm25_scores",
    "_extract_phrases",
    "_parse_relevant_source_indexes",
    "_split_text",
    "_tokenize",
]

"""Grounded document QA with trust-layer analysis and citations."""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm.factory import get_chat_model_async, invoke_plain
from prompts.qa import (
    TRUST_ANALYSIS_SYSTEM,
    answer_from_analysis_system,
    effective_response_language,
    summary_system,
)
from retriever.semantic import DocumentIndex, format_context, retrieve
from routing.language import detect_language
from vectorstore.base import RetrievedChunk

LLMInvoke = Callable[[list[dict[str, str]]], str]


def _strip_sources_footer(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(
        r"\n*-{0,3}\s*\n?(?:Source analysis \(trust layer\):|Sources:|_Sources:_).*\Z",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def _parse_relevant_indexes(analysis: str, chunk_count: int) -> set[int]:
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


def format_sources(
    chunks: list[RetrievedChunk],
    *,
    relevant_indexes: set[int] | None = None,
) -> str:
    """Build a Sources footer with page references."""
    if not chunks:
        return (
            "Sources:\n"
            "- No matching passages were retrieved from the uploaded documents."
        )
    lines = [
        f"Sources (top {len(chunks)} retrieved — model read each before answering):",
    ]
    for index, chunk in enumerate(chunks, start=1):
        if chunk.page_start and chunk.page_end and chunk.page_start != chunk.page_end:
            page = f"pages {chunk.page_start}-{chunk.page_end}"
        elif chunk.page_start:
            page = f"page {chunk.page_start}"
        else:
            page = "page n/a"
        flags = [f"score={chunk.score:.3f}", page]
        if relevant_indexes is not None:
            flags.append("USED" if index in relevant_indexes else "reviewed")
        lines.append(f"- [{index}] {chunk.filename} ({', '.join(flags)})")
    return "\n".join(lines)


def analyze_sources_sync(
    question: str,
    chunks: list[RetrievedChunk],
    llm_invoke: LLMInvoke | None = None,
) -> str:
    """Trust layer pass 1 — judge each retrieved passage."""
    invoke = llm_invoke or invoke_plain
    context = format_context(chunks)
    return str(
        invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{TRUST_ANALYSIS_SYSTEM}\n\n"
                        f"User question:\n{question}\n\n"
                        f"Retrieved sources:\n{context}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze every source. Mark RELEVANT or NOT_RELEVANT "
                        "and quote evidence."
                    ),
                },
            ]
        )
    ).strip()


def answer_from_analysis_sync(
    question: str,
    analysis: str,
    chunks: list[RetrievedChunk],
    llm_invoke: LLMInvoke | None = None,
    *,
    response_language: str | None = None,
) -> str:
    """Trust layer pass 2 — answer only from RELEVANT evidence."""
    invoke = llm_invoke or invoke_plain
    index_map = ", ".join(
        f"[{i}] {chunk.filename} p.{chunk.page_start}"
        for i, chunk in enumerate(chunks, start=1)
    )
    detected = detect_language(question)
    lang = effective_response_language(response_language, detected)
    preferred = response_language if response_language in ("ar", "en") else None
    return str(
        invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{answer_from_analysis_system(preferred)}\n\n"
                        f"Preferred answer language: {lang}\n"
                        f"Source index map: {index_map}\n\n"
                        f"Source analysis:\n{analysis}"
                    ),
                },
                {"role": "user", "content": question},
            ]
        )
    ).strip()


def answer_document_question_sync(
    index: DocumentIndex,
    question: str,
    llm_invoke: LLMInvoke | None = None,
    *,
    response_language: str | None = None,
) -> str:
    """Retrieve → trust-layer analyze → grounded answer + Sources."""
    chunks = retrieve(index, question)
    if not chunks:
        lang = effective_response_language(
            response_language,
            detect_language(question),
        )
        msg = {
            "ar": "تعذر تحديد الإجابة من المستندات المرفوعة.",
            "en": "The answer cannot be determined from the uploaded documents.",
            "mixed": (
                "The answer cannot be determined from the uploaded documents. / "
                "تعذر تحديد الإجابة من المستندات المرفوعة."
            ),
        }.get(lang, "The answer cannot be determined from the uploaded documents.")
        return f"{msg}\n\n{format_sources([])}"

    analysis = analyze_sources_sync(question, chunks, llm_invoke)
    answer = answer_from_analysis_sync(
        question,
        analysis,
        chunks,
        llm_invoke,
        response_language=response_language,
    )
    body = _strip_sources_footer(answer)
    relevant = _parse_relevant_indexes(analysis, len(chunks))
    analysis_header = "Source analysis (trust layer)"
    if relevant:
        used = ", ".join(f"[{i}]" for i in sorted(relevant))
        analysis_header = f"{analysis_header} — used: {used}"
    analysis_body = analysis if len(analysis) <= 3500 else analysis[:3500] + "\n…"
    return (
        f"{body}\n\n"
        f"---\n{analysis_header}:\n{analysis_body}\n\n"
        f"{format_sources(chunks, relevant_indexes=relevant)}"
    )


async def summarize_document(
    index: DocumentIndex,
    *,
    response_language: str | None = None,
) -> str:
    """Summarize the uploaded document with the chat model."""
    excerpt = index.full_text[:12000]
    detected = detect_language(excerpt[:500])
    lang = effective_response_language(response_language, detected)
    preferred = response_language if response_language in ("ar", "en") else None
    messages = [
        SystemMessage(
            content=(
                f"{summary_system(preferred)}\n"
                f"Preferred answer language: {lang}"
            )
        ),
        HumanMessage(
            content=(
                f"Filename: {index.filename}\n\n"
                f"Document text:\n{excerpt}\n\n"
                "Write a concise summary of the main points."
            )
        ),
    ]
    model = await get_chat_model_async()
    response = await model.ainvoke(messages)
    return str(response.content).strip() or "I could not generate a document summary."


def answer_knowledge_question_sync(
    question: str,
    *,
    search_query: str | None = None,
    keywords: list[str] | None = None,
    llm_invoke: LLMInvoke | None = None,
    response_language: str | None = None,
) -> str:
    """Retrieve from the seeded knowledge DB (hybrid BM25) → grounded answer."""
    from retriever.knowledge import retrieve_from_knowledge

    query = (search_query or question or "").strip()
    chunks = retrieve_from_knowledge(query, keywords=keywords)
    if not chunks:
        lang = effective_response_language(
            response_language,
            detect_language(question),
        )
        msg = {
            "ar": "تعذر العثور على معلومات كافية في قاعدة المعرفة.",
            "en": "Not enough information was found in the knowledge database.",
            "mixed": (
                "Not enough information was found in the knowledge database. / "
                "تعذر العثور على معلومات كافية في قاعدة المعرفة."
            ),
        }.get(lang, "Not enough information was found in the knowledge database.")
        return f"{msg}\n\n{format_sources([])}"

    analysis = analyze_sources_sync(question, chunks, llm_invoke)
    answer = answer_from_analysis_sync(
        question,
        analysis,
        chunks,
        llm_invoke,
        response_language=response_language,
    )
    body = _strip_sources_footer(answer)
    relevant = _parse_relevant_indexes(analysis, len(chunks))
    analysis_header = "Source analysis (trust layer)"
    if relevant:
        used = ", ".join(f"[{i}]" for i in sorted(relevant))
        analysis_header = f"{analysis_header} — used: {used}"
    analysis_body = analysis if len(analysis) <= 3500 else analysis[:3500] + "\n…"
    return (
        f"{body}\n\n"
        f"---\n{analysis_header}:\n{analysis_body}\n\n"
        f"{format_sources(chunks, relevant_indexes=relevant)}"
    )


async def knowledge_analysis_response(
    *,
    question: str,
    search_query: str | None = None,
    keywords: list[str] | None = None,
    response_language: str | None = None,
) -> AIMessage:
    """Answer from the seeded knowledge collection without requiring a PDF upload."""
    from utils.async_utils import run_in_thread

    preferred = response_language if response_language in ("ar", "en") else None
    try:
        answer = await run_in_thread(
            answer_knowledge_question_sync,
            question.strip(),
            search_query=search_query,
            keywords=keywords,
            llm_invoke=None,
            response_language=preferred,
        )
        return AIMessage(content=answer)
    except Exception as exc:  # noqa: BLE001
        return AIMessage(content=f"Knowledge retrieval error: {exc}")


async def document_analysis_response(
    *,
    question: str,
    data_base64: str,
    filename: str = "uploaded.pdf",
    mime_type: str | None = None,
    summarize_only: bool = False,
    response_language: str | None = None,
) -> AIMessage:
    """Ingest (if needed) and answer or summarize — graph-compatible AIMessage."""
    from utils.async_utils import run_in_thread
    from retriever.semantic import get_or_build_index

    preferred = response_language if response_language in ("ar", "en") else None

    try:
        index = await run_in_thread(
            get_or_build_index,
            data_base64,
            filename,
            mime_type,
        )
        if summarize_only or not (question or "").strip():
            summary = await summarize_document(
                index,
                response_language=preferred,
            )
            if preferred == "ar":
                wrapper = (
                    f"تمت فهرسة المستند: **{index.filename}** "
                    f"({index.metadata.get('chunk_count', len(index.chunks))} مقطع، "
                    f"OCR={index.ocr.engine})\n\n"
                    f"**الملخص**\n{summary}\n\n"
                    "يمكنك طرح أسئلة متابعة حول هذا المستند."
                )
            else:
                wrapper = (
                    f"Document indexed: **{index.filename}** "
                    f"({index.metadata.get('chunk_count', len(index.chunks))} chunks, "
                    f"OCR={index.ocr.engine})\n\n"
                    f"**Summary**\n{summary}\n\n"
                    "You can ask follow-up questions about this document."
                )
            return AIMessage(content=wrapper)

        answer = await run_in_thread(
            answer_document_question_sync,
            index,
            question.strip(),
            None,
            response_language=preferred,
        )
        return AIMessage(content=answer)
    except Exception as exc:  # noqa: BLE001
        return AIMessage(content=f"Document analysis error: {exc}")

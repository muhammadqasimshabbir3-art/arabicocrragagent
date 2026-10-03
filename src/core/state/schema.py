"""Typed state for the Arabic Document Intelligence graph."""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

AgentRoute = Literal[
    "query_documents",
    "summarize_document",
    "query_knowledge_base",
    "web_search",
    "call_model",
]


class AgentState(TypedDict):
    """State for the Arabic Document Intelligence agent graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_input: NotRequired[str]
    agent_route: NotRequired[AgentRoute]
    task_plan_summary: NotRequired[str]
    # Document upload (PDF or image); kept as pdf_* for frontend compatibility.
    pdf_data_base64: NotRequired[str]
    pdf_filename: NotRequired[str]
    pdf_summarize_only: NotRequired[bool]
    document_data_base64: NotRequired[str]
    document_filename: NotRequired[str]
    document_mime_type: NotRequired[str]
    summarize_only: NotRequired[bool]
    # Ingest / retrieval metadata surfaced to the UI.
    document_fingerprint: NotRequired[str]
    indexed_chunk_count: NotRequired[int]
    detected_language: NotRequired[str]
    # Preferred answer language from the UI ("ar" | "en"); overrides detection.
    response_language: NotRequired[str]
    # Explicit UI source toggles (ChatGPT-style). When set, beat heuristics.
    use_web_search: NotRequired[bool]
    use_knowledge_base: NotRequired[bool]
    # Spelling Guardian output (conservative ar/en typo fixes).
    spelling_original: NotRequired[str]
    spelling_corrected: NotRequired[str]
    spelling_applied: NotRequired[bool]
    query_boost_applied: NotRequired[bool]
    query_boost_notes: NotRequired[list[str]]
    # READ-only knowledge-base query plan written by the query planner agent.
    query_plan: NotRequired[dict]
    retrieval_query: NotRequired[str]
    retrieval_keywords: NotRequired[list[str]]
    # Structured error channel (user-safe message; details stay in logs).
    error_code: NotRequired[str]
    error_message: NotRequired[str]

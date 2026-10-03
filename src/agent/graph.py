"""Arabic Document Intelligence Agent — LangGraph workflow."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from subagents.document_qa import document_analysis_response, knowledge_analysis_response
from subagents.query_planner import plan_database_query
from core.llm.factory import get_chat_model_async
from core.prompts.qa import SYSTEM_IDENTITY, effective_response_language, language_rules_for
from core.routing.intent import get_latest_user_text, pick_route
from core.routing.language import detect_language
from core.state.schema import AgentRoute, AgentState

load_dotenv()

# Optional fail-fast for production (tests omit GROQ — keep off by default).
if os.getenv("VALIDATE_SETTINGS_ON_STARTUP", "").lower() in {"1", "true", "yes", "on"}:
    try:
        from core.config.settings import validate_startup_settings

        validate_startup_settings()
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("Startup settings validation failed: %s", exc)

GRAPH_RUN_CONFIG = {"recursion_limit": 50}

# Alias used by LangGraph / older imports.
State = AgentState


async def get_model():
    """Return shared chat model without blocking event loop."""
    return await get_chat_model_async()


def _preferred_response_language(state: AgentState) -> str | None:
    preferred = state.get("response_language")
    return preferred if preferred in ("ar", "en") else None


def _prepare_messages(state: AgentState) -> tuple[list, list]:
    existing = list(state.get("messages") or [])
    updates: list = []
    if existing:
        return existing, updates
    user_input = state.get("user_input", "")
    if not user_input:
        raise ValueError(
            "No input provided. Pass messages and/or user_input in graph state."
        )
    human = HumanMessage(content=user_input)
    updates.append(human)
    return [human], updates


def _document_payload(state: AgentState) -> tuple[str, str, str | None, bool]:
    """Resolve upload fields (new + legacy pdf_* keys)."""
    data = (
        state.get("document_data_base64")
        or state.get("pdf_data_base64")
        or ""
    )
    filename = (
        state.get("document_filename")
        or state.get("pdf_filename")
        or "uploaded.pdf"
    )
    mime = state.get("document_mime_type")
    summarize = bool(
        state.get("summarize_only") or state.get("pdf_summarize_only")
    )
    return data, filename, mime, summarize


async def prepare_input(state: AgentState) -> dict[str, Any]:
    """Normalize input into conversation messages."""
    _, updates = _prepare_messages(state)
    if updates:
        return {"messages": updates}
    return {}


async def decision_agent(state: AgentState) -> dict[str, Any]:
    """Classify the turn: ingest/query/summarize/knowledge/chat."""
    from core.retriever.knowledge import knowledge_available
    from core.utils.async_utils import run_in_thread

    messages, _ = _prepare_messages(state)
    user_text = get_latest_user_text(messages)
    data, _, _, summarize = _document_payload(state)
    has_document = bool(data)
    lang = detect_language(user_text)
    kb_ready = await run_in_thread(knowledge_available)
    use_web = bool(state.get("use_web_search"))
    use_kb = bool(state.get("use_knowledge_base"))
    route = pick_route(
        user_text,
        messages,
        has_document=has_document,
        summarize_only=summarize,
        knowledge_available=kb_ready,
        use_web_search=use_web,
        use_knowledge_base=use_kb,
    )

    summaries = {
        "query_documents": "Retrieve → trust-layer read → grounded answer + citations",
        "summarize_document": "Ingest (if needed) → summarize document",
        "query_knowledge_base": (
            "Database → READ-only query plan → hybrid BM25+vector retrieve → answer"
        ),
        "web_search": "Search Internet → live web lookup → grounded answer",
        "call_model": "General conversation",
    }
    if use_kb and route == "call_model" and not has_document and not kb_ready:
        plan = "Database selected but knowledge base is empty — general chat"
    elif use_web:
        plan = summaries.get(route, "Search Internet")
    elif use_kb:
        plan = summaries.get(route, "Database lookup")
    else:
        plan = summaries.get(route, "General conversation")
    return {
        "agent_route": route,
        "task_plan_summary": plan,
        "detected_language": lang,
        "use_web_search": use_web,
        "use_knowledge_base": use_kb,
    }


async def query_documents(state: AgentState) -> dict[str, Any]:
    """Answer a question grounded in the uploaded document."""
    from core.utils.async_utils import run_in_thread
    from subagents.spelling_corrector import correct_spelling

    messages, _ = _prepare_messages(state)
    user_text = get_latest_user_text(messages)
    spell = await run_in_thread(correct_spelling, user_text)
    cleaned = spell.corrected.strip() or user_text
    data, filename, mime, _ = _document_payload(state)
    response = await document_analysis_response(
        question=cleaned,
        data_base64=data,
        filename=filename,
        mime_type=mime,
        summarize_only=False,
        response_language=_preferred_response_language(state),
    )
    out: dict[str, Any] = {
        "messages": [response],
        "document_filename": filename,
        "spelling_original": spell.original,
        "spelling_corrected": spell.corrected,
        "spelling_applied": spell.applied,
    }
    return out


async def summarize_document_node(state: AgentState) -> dict[str, Any]:
    """Summarize the uploaded document."""
    data, filename, mime, _ = _document_payload(state)
    response = await document_analysis_response(
        question="",
        data_base64=data,
        filename=filename,
        mime_type=mime,
        summarize_only=True,
        response_language=_preferred_response_language(state),
    )
    return {"messages": [response], "document_filename": filename}


async def query_planner(state: AgentState) -> dict[str, Any]:
    """Write a READ-only Arabic retrieval plan for the knowledge base."""
    from core.utils.async_utils import run_in_thread
    from subagents.spelling_corrector import correct_spelling

    messages, _ = _prepare_messages(state)
    user_text = get_latest_user_text(messages)
    spell = await run_in_thread(correct_spelling, user_text)
    cleaned = spell.corrected.strip() or user_text
    plan = await run_in_thread(plan_database_query, cleaned, skip_spelling=True)
    arabic_query = plan.search_query or cleaned
    summary = f"READ-only Arabic query: {arabic_query or '(none)'} — {plan.reason}"
    if spell.applied:
        summary = f"Spelling Guardian fixed typos → {summary}"
    if plan.boost_applied:
        summary = f"Query Booster added context notes → {summary}"
    return {
        "query_plan": plan.to_dict(),
        "retrieval_query": arabic_query,
        "retrieval_keywords": plan.keywords,
        "query_boost_applied": plan.boost_applied,
        "query_boost_notes": plan.query_notes,
        "spelling_original": spell.original,
        "spelling_corrected": spell.corrected,
        "spelling_applied": spell.applied,
        "task_plan_summary": summary,
    }


async def query_knowledge_base(state: AgentState) -> dict[str, Any]:
    """Execute READ-only hybrid BM25+vector retrieval and answer from evidence.

    Retrieval uses the Arabic-converted query (embedding + BM25).
    Answering uses the original user question / preferred response language.
    """
    messages, _ = _prepare_messages(state)
    user_text = get_latest_user_text(messages)
    plan = state.get("query_plan") or {}
    if plan and not plan.get("needs_database", True):
        # Planner decided no DB read is needed — fall back to general chat.
        return await call_model(state)

    arabic_query = (
        state.get("retrieval_query")
        or plan.get("search_query")
        or user_text
    )
    response = await knowledge_analysis_response(
        question=user_text,
        search_query=arabic_query,
        keywords=list(state.get("retrieval_keywords") or plan.get("keywords") or []),
        response_language=_preferred_response_language(state),
    )
    return {"messages": [response]}


async def web_search_node(state: AgentState) -> dict[str, Any]:
    """Search the web when the user needs live/online information."""
    from tools.web_search import (
        extract_links,
        screenshot_markdown,
        wants_profile_screenshot,
        web_search_sync,
    )
    from core.utils.async_utils import run_in_thread

    messages, _ = _prepare_messages(state)
    user_text = get_latest_user_text(messages)
    detected = detect_language(user_text)
    preferred = _preferred_response_language(state)
    lang = effective_response_language(preferred, detected)

    search_blob = await run_in_thread(web_search_sync, user_text, 5)
    if wants_profile_screenshot(user_text):
        links = extract_links(search_blob)
        shots = screenshot_markdown(links, max_items=2)
        if shots:
            search_blob = f"{search_blob}\n\n{shots}"
    system = SystemMessage(
        content=(
            f"{SYSTEM_IDENTITY}\n\n{language_rules_for(preferred)}\n\n"
            f"Preferred answer language: {lang}.\n"
            "Answer using the web search results below. "
            "Cite links when useful. If results are weak, say so clearly.\n\n"
            f"WEB SEARCH RESULTS:\n{search_blob}"
        )
    )
    model = await get_model()
    response = await model.ainvoke([system, *messages])
    content = getattr(response, "content", "")
    if not content or not str(content).strip():
        response = AIMessage(content=search_blob)
    return {"messages": [response], "task_plan_summary": "Web search completed"}


async def call_model(state: AgentState) -> dict[str, Any]:
    """General chat with tool access — can search web, read/write files, create PDFs."""
    from core.utils.async_utils import run_in_thread

    messages, _ = _prepare_messages(state)
    user_text = get_latest_user_text(messages)
    detected = detect_language(user_text)
    preferred = _preferred_response_language(state)
    lang = effective_response_language(preferred, detected)

    system = SystemMessage(
        content=(
            f"{SYSTEM_IDENTITY}\n\n{language_rules_for(preferred)}\n\n"
            f"Preferred answer language: {lang}.\n"
            "You have access to tools: web search, file operations, PDF creation, "
            "terminal commands, and git inspection. Use them when the user's request "
            "requires it. For Arabic knowledge-base questions the graph uses the "
            "database path. If the user asks about an uploaded file that is not "
            "attached, tell them to upload or re-attach the document with their question."
        )
    )
    history = [system, *messages]
    model = await get_model()
    try:
        # Importing tools pulls transformers (blocking cwd). Load off the event loop.
        def _load_tools():
            from tools import get_agent_tools

            return get_agent_tools()

        agent_tools = await run_in_thread(_load_tools)
        model_with_tools = model.bind_tools(agent_tools)
        response = await model_with_tools.ainvoke(history)
    except Exception:  # noqa: BLE001 — fallback to plain model if tool binding fails
        response = await model.ainvoke(history)
    content = getattr(response, "content", "")
    if not content or not str(content).strip():
        response = AIMessage(
            content=(
                "أنا جاهز لمساعدتك. اسأل عن قاعدة المعرفة، أو ابحث في الويب، أو ارفع ملف."
                if lang == "ar"
                else "I am ready to help. Ask about the knowledge base, "
                "search the web, or upload a document."
            )
        )
    return {"messages": [response]}


def route_after_decision(state: AgentState) -> AgentRoute | str:
    """Route from decision_agent to the chosen execution node."""
    route = state.get("agent_route", "call_model")
    if route == "query_knowledge_base":
        return "query_planner"
    return route


# ---------------------------------------------------------------------------
# Graph
#   START → prepare_input → decision_agent →
#       query_documents | summarize_document |
#       query_planner → query_knowledge_base | web_search | call_model → END
# ---------------------------------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("prepare_input", prepare_input)
graph_builder.add_node("decision_agent", decision_agent)
graph_builder.add_node("query_documents", query_documents)
graph_builder.add_node("summarize_document", summarize_document_node)
graph_builder.add_node("query_planner", query_planner)
graph_builder.add_node("query_knowledge_base", query_knowledge_base)
graph_builder.add_node("web_search", web_search_node)
graph_builder.add_node("call_model", call_model)

graph_builder.add_edge(START, "prepare_input")
graph_builder.add_edge("prepare_input", "decision_agent")
graph_builder.add_conditional_edges(
    "decision_agent",
    route_after_decision,
    {
        "query_documents": "query_documents",
        "summarize_document": "summarize_document",
        "query_planner": "query_planner",
        "web_search": "web_search",
        "call_model": "call_model",
    },
)
graph_builder.add_edge("query_planner", "query_knowledge_base")
graph_builder.add_edge("query_documents", END)
graph_builder.add_edge("summarize_document", END)
graph_builder.add_edge("query_knowledge_base", END)
graph_builder.add_edge("web_search", END)
graph_builder.add_edge("call_model", END)

graph = graph_builder.compile(name="Arabic Document Intelligence Agent")

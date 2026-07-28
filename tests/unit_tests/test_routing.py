"""Unit tests for Arabic document routing and language detection."""

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import decision_agent
from routing.intent import pick_route, wants_summary
from routing.language import detect_language


def test_detect_language_arabic():
    assert detect_language("ما موضوع هذا المستند؟") == "ar"


def test_detect_language_english():
    assert detect_language("What is the main topic?") == "en"


def test_detect_language_mixed():
    assert detect_language("Explain هذه النقطة in detail please") == "mixed"


def test_wants_summary():
    assert wants_summary("Summarize this PDF")
    assert wants_summary("لخص المستند")


def test_pick_route_with_document_question():
    messages = [HumanMessage(content="What is the conclusion?")]
    assert (
        pick_route("What is the conclusion?", messages, has_document=True)
        == "query_documents"
    )


def test_pick_route_summary():
    messages = [HumanMessage(content="Summarize")]
    assert (
        pick_route("Summarize", messages, has_document=True, summarize_only=True)
        == "summarize_document"
    )


def test_pick_route_chat_without_document():
    messages = [HumanMessage(content="Hello")]
    assert pick_route("Hello", messages, has_document=False) == "call_model"


def test_pick_route_knowledge_question_when_kb_ready():
    messages = [HumanMessage(content="ما موضوع ألف ليلة وليلة؟")]
    assert (
        pick_route(
            "ما موضوع ألف ليلة وليلة؟",
            messages,
            has_document=False,
            knowledge_available=True,
        )
        == "query_knowledge_base"
    )


def test_pick_route_knowledge_disabled_falls_back_to_chat():
    messages = [HumanMessage(content="ما موضوع ألف ليلة وليلة؟")]
    assert (
        pick_route(
            "ما موضوع ألف ليلة وليلة؟",
            messages,
            has_document=False,
            knowledge_available=False,
        )
        == "call_model"
    )


def test_pick_route_web_search_when_needed():
    messages = [HumanMessage(content="Search the web for today's weather in Riyadh")]
    assert (
        pick_route(
            "Search the web for today's weather in Riyadh",
            messages,
            has_document=False,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_web_search_arabic():
    messages = [HumanMessage(content="ابحث في الويب عن أخبار اليوم")]
    assert (
        pick_route(
            "ابحث في الويب عن أخبار اليوم",
            messages,
            has_document=False,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_web_search_with_document_attached():
    messages = [HumanMessage(content="Search the web for Muhammad Qasim GitHub profile")]
    assert (
        pick_route(
            "Search the web for Muhammad Qasim GitHub profile",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_github_question_without_explicit_search():
    messages = [HumanMessage(content="Could you provide his GitHub profile and summarize public repositories?")]
    assert (
        pick_route(
            "Could you provide his GitHub profile and summarize public repositories?",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_github_question_arabic_without_explicit_search():
    messages = [HumanMessage(content="ما هو رابط GitHub الخاص به وما أبرز مستودعاته العامة؟")]
    assert (
        pick_route(
            "ما هو رابط GitHub الخاص به وما أبرز مستودعاته العامة؟",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_kaggle_profile_query_with_document():
    messages = [HumanMessage(content="Can you find his Kaggle profile and summarize his public notebooks?")]
    assert (
        pick_route(
            "Can you find his Kaggle profile and summarize his public notebooks?",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_zindi_profile_query_arabic():
    messages = [HumanMessage(content="هات حسابه على زيندي ولخص أعماله العامة")]
    assert (
        pick_route(
            "هات حسابه على زيندي ولخص أعماله العامة",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_unknown_platform_profile_query():
    messages = [HumanMessage(content="Please find his DataArena profile and public work summary")]
    assert (
        pick_route(
            "Please find his DataArena profile and public work summary",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pick_route_domain_link_query_without_platform_name():
    messages = [HumanMessage(content="Can you check profile at dataarena.ai/user/john and summarize?")]
    assert (
        pick_route(
            "Can you check profile at dataarena.ai/user/john and summarize?",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "web_search"
    )


def test_pdf_path_wins_over_knowledge_when_file_attached():
    """Uploaded PDF stays on the document path even if knowledge DB is ready."""
    messages = [HumanMessage(content="ما موضوع ألف ليلة وليلة؟")]
    assert (
        pick_route(
            "ما موضوع ألف ليلة وليلة؟",
            messages,
            has_document=True,
            knowledge_available=True,
        )
        == "query_documents"
    )


@pytest.mark.anyio
async def test_document_payload_routes_to_query():
    result = await decision_agent(
        {
            "messages": [HumanMessage(content="What is the main conclusion?")],
            "pdf_data_base64": "JVBERi0xLjQK",
            "pdf_filename": "sample.pdf",
        }
    )
    assert result["agent_route"] == "query_documents"


@pytest.mark.anyio
async def test_summarize_flag_routes_to_summarize():
    result = await decision_agent(
        {
            "messages": [HumanMessage(content="hello")],
            "pdf_data_base64": "JVBERi0xLjQK",
            "pdf_summarize_only": True,
        }
    )
    assert result["agent_route"] == "summarize_document"

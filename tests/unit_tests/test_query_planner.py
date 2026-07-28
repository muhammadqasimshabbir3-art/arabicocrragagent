"""Unit tests for READ-only query planner heuristics."""

from agent.query_planner import _heuristic_plan, plan_database_query, to_arabic_retrieval_query
from routing.language import has_arabic


def test_heuristic_small_talk_skips_database():
    plan = _heuristic_plan("Hello")
    assert plan.needs_database is False
    assert plan.mode == "read"


def test_heuristic_content_question_needs_read_query():
    plan = _heuristic_plan("ما موضوع ألف ليلة وليلة؟")
    assert plan.needs_database is True
    assert plan.mode == "read"
    assert plan.search_query
    assert plan.keywords


def test_plan_database_query_always_read_mode():
    plan = plan_database_query("Who wrote this book?")
    assert plan.mode == "read"


def test_english_question_becomes_arabic_retrieval_query():
    arabic = to_arabic_retrieval_query("Who is the main character and why do they tell stories?")
    assert has_arabic(arabic)


def test_arabic_question_stays_arabic():
    q = "ما موضوع ألف ليلة وليلة؟"
    assert to_arabic_retrieval_query(q) == q

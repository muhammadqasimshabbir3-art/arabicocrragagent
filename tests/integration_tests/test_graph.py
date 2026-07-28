"""Integration tests for the Arabic Document Intelligence graph."""

import pytest
from langchain_core.messages import HumanMessage

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_initialization() -> None:
    assert graph is not None
    assert graph.invoke is not None


@pytest.mark.langsmith
async def test_agent_with_simple_query() -> None:
    inputs = {
        "messages": [HumanMessage(content="Hello, what can you help me with?")],
        "user_input": "Hello, what can you help me with?",
    }
    result = await graph.ainvoke(inputs)
    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) > 1

"""Graph and settings smoke tests."""

from langgraph.pregel import Pregel

from agent.graph import graph
from config.settings import get_settings


def test_graph_is_compiled() -> None:
    assert isinstance(graph, Pregel)
    nodes = set(graph.nodes)
    assert "decision_agent" in nodes
    assert "query_documents" in nodes
    assert "run_calculator" not in nodes


def test_settings_defaults() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.embedding_model_id == "BAAI/bge-m3"
    assert settings.ocr_engine in {"qari", "ain", "digital", "none"}
    assert settings.vectorstore_backend in {"chroma", "faiss", "qdrant", "milvus"}
    assert "chroma" in settings.vectorstore_persist_dir or settings.vectorstore_persist_dir in {
        "",
        "memory",
        "none",
        "ephemeral",
    }
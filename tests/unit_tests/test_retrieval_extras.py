"""Unit tests for citations, FAISS store, and lexical reranker."""

from agent.document_qa import format_sources
from chunking.base import TextChunk
from reranker import rerank
from vectorstore.base import RetrievedChunk
from vectorstore.faiss_store import FaissVectorStore


def test_format_sources_marks_used_passages():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="a",
            score=0.9,
            page_start=1,
            page_end=1,
            filename="doc.pdf",
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="b",
            score=0.5,
            page_start=2,
            page_end=2,
            filename="doc.pdf",
        ),
    ]
    footer = format_sources(chunks, relevant_indexes={1})
    assert "Sources (top 2" in footer
    assert "USED" in footer
    assert "reviewed" in footer
    assert "page 1" in footer


def test_lexical_reranker_prefers_overlap():
    chunks = [
        RetrievedChunk(
            chunk_id="1",
            text="سياسة الشحن الدولي",
            score=0.5,
            filename="a.pdf",
        ),
        RetrievedChunk(
            chunk_id="2",
            text="موضوع غير متعلق",
            score=0.5,
            filename="a.pdf",
        ),
    ]
    ranked = rerank("ما هي سياسة الشحن؟", chunks, top_k=2)
    assert ranked[0].chunk_id == "1"
    assert ranked[0].metadata.get("rerank") == "lexical"


def test_faiss_vector_store_roundtrip():
    store = FaissVectorStore(collection_name="unit")
    chunks = [
        TextChunk(chunk_id="a", text="alpha", page_start=1, page_end=1),
        TextChunk(chunk_id="b", text="beta", page_start=2, page_end=2),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    store.add_chunks(chunks, embeddings, filename="t.pdf")
    hits = store.similarity_search([1.0, 0.0, 0.0], top_k=1)
    assert store.count() == 2
    assert hits[0].chunk_id == "a"
    assert hits[0].filename == "t.pdf"

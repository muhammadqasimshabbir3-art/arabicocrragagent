#!/usr/bin/env python3
"""One-shot src/ refactor: 16 folders → 4 packages (agent, subagents, core, tools).

Run from repo root:
    python scripts/refactor_src.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# ---------------------------------------------------------------------------
# 1. File copy map:  src-relative-source → src-relative-destination
# ---------------------------------------------------------------------------
COPIES: list[tuple[str, str]] = [
    # ── core/ ──────────────────────────────────────────────────────────────
    ("config/__init__.py",          "core/config/__init__.py"),
    ("config/settings.py",          "core/config/settings.py"),
    ("state/__init__.py",           "core/state/__init__.py"),
    ("state/schema.py",             "core/state/schema.py"),
    ("utils/__init__.py",           "core/utils/__init__.py"),
    ("utils/async_utils.py",        "core/utils/async_utils.py"),
    ("utils/logging_utils.py",      "core/utils/logging_utils.py"),
    ("utils/sanitize.py",           "core/utils/sanitize.py"),
    ("preprocess/__init__.py",      "core/preprocess/__init__.py"),
    ("preprocess/arabic_normalize.py", "core/preprocess/arabic_normalize.py"),
    ("prompts/__init__.py",         "core/prompts/__init__.py"),
    ("prompts/planner.py",          "core/prompts/planner.py"),
    ("prompts/qa.py",               "core/prompts/qa.py"),
    ("chunking/__init__.py",        "core/chunking/__init__.py"),
    ("chunking/base.py",            "core/chunking/base.py"),
    ("chunking/factory.py",         "core/chunking/factory.py"),
    ("chunking/layout_aware.py",    "core/chunking/layout_aware.py"),
    ("chunking/page_aware.py",      "core/chunking/page_aware.py"),
    ("chunking/recursive.py",       "core/chunking/recursive.py"),
    ("chunking/semantic.py",        "core/chunking/semantic.py"),
    ("llm/__init__.py",             "core/llm/__init__.py"),
    ("llm/factory.py",              "core/llm/factory.py"),
    ("loaders/__init__.py",         "core/loaders/__init__.py"),
    ("loaders/base.py",             "core/loaders/base.py"),
    ("loaders/factory.py",          "core/loaders/factory.py"),
    ("routing/__init__.py",         "core/routing/__init__.py"),
    ("routing/intent.py",           "core/routing/intent.py"),
    ("routing/language.py",         "core/routing/language.py"),
    ("retriever/__init__.py",       "core/retriever/__init__.py"),
    ("retriever/hybrid.py",         "core/retriever/hybrid.py"),
    ("retriever/knowledge.py",      "core/retriever/knowledge.py"),
    ("retriever/semantic.py",       "core/retriever/semantic.py"),
    ("vectorstore/__init__.py",     "core/vectorstore/__init__.py"),
    ("vectorstore/base.py",         "core/vectorstore/base.py"),
    ("vectorstore/chroma_store.py", "core/vectorstore/chroma_store.py"),
    ("vectorstore/factory.py",      "core/vectorstore/factory.py"),
    ("vectorstore/faiss_store.py",  "core/vectorstore/faiss_store.py"),
    ("vectorstore/qdrant_store.py", "core/vectorstore/qdrant_store.py"),

    # ── subagents/ ─────────────────────────────────────────────────────────
    ("ocr/__init__.py",             "subagents/ocr/__init__.py"),
    ("ocr/base.py",                 "subagents/ocr/base.py"),
    ("ocr/digital_text.py",         "subagents/ocr/digital_text.py"),
    ("ocr/factory.py",              "subagents/ocr/factory.py"),
    ("ocr/qari.py",                 "subagents/ocr/qari.py"),
    ("embeddings/__init__.py",      "subagents/embeddings/__init__.py"),
    ("embeddings/base.py",          "subagents/embeddings/base.py"),
    ("embeddings/bge_m3.py",        "subagents/embeddings/bge_m3.py"),
    ("embeddings/factory.py",       "subagents/embeddings/factory.py"),
    ("reranker/__init__.py",        "subagents/reranker/__init__.py"),
    ("agent/document_qa.py",        "subagents/document_qa.py"),
    ("agent/query_planner.py",      "subagents/query_planner.py"),
    ("agent/pdf_analysis.py",       "subagents/pdf_analysis.py"),

    # ── tools/ (custom_tools → tools, keep existing web_search) ───────────
    ("custom_tools/file_tools.py",       "tools/file_tools.py"),
    ("custom_tools/file_search_tools.py","tools/file_search_tools.py"),
    ("custom_tools/git_tools.py",        "tools/git_tools.py"),
    ("custom_tools/terminal_tools.py",   "tools/terminal_tools.py"),
    ("custom_tools/browser_tools.py",    "tools/browser_tools.py"),
    ("custom_tools/report_io.py",        "tools/report_io.py"),
    ("custom_tools/web_search_tools.py", "tools/web_search_tools.py"),
    # existing tools kept in place — web_search.py already at tools/web_search.py
]

# ---------------------------------------------------------------------------
# 2. Import rewrite rules (old_pattern → new_text)
#    Applied to every .py file under src/, tests/, scripts/, streamlit_ui.py
# ---------------------------------------------------------------------------
REWRITES: list[tuple[str, str]] = [
    # subagents — must come before generic "agent." rules
    (r"\bfrom agent\.document_qa\b",    "from subagents.document_qa"),
    (r"\bfrom agent\.query_planner\b",  "from subagents.query_planner"),
    (r"\bfrom agent\.pdf_analysis\b",   "from subagents.pdf_analysis"),
    (r"\bimport agent\.document_qa\b",  "import subagents.document_qa"),
    (r"\bimport agent\.query_planner\b","import subagents.query_planner"),
    (r"\bimport agent\.pdf_analysis\b", "import subagents.pdf_analysis"),

    # ocr → subagents.ocr
    (r"\bfrom ocr\.",    "from subagents.ocr."),
    (r"\bimport ocr\b",  "import subagents.ocr"),
    (r"\bfrom ocr\b",    "from subagents.ocr"),

    # embeddings → subagents.embeddings
    (r"\bfrom embeddings\.", "from subagents.embeddings."),
    (r"\bimport embeddings\b","import subagents.embeddings"),
    (r"\bfrom embeddings\b",  "from subagents.embeddings"),

    # reranker → subagents.reranker
    (r"\bfrom reranker\.", "from subagents.reranker."),
    (r"\bfrom reranker\b", "from subagents.reranker"),
    (r"\bimport reranker\b","import subagents.reranker"),

    # core packages
    (r"\bfrom config\.",    "from core.config."),
    (r"\bfrom config\b",    "from core.config"),
    (r"\bfrom state\.",     "from core.state."),
    (r"\bfrom state\b",     "from core.state"),
    (r"\bfrom utils\.",     "from core.utils."),
    (r"\bfrom utils\b",     "from core.utils"),
    (r"\bfrom preprocess\.", "from core.preprocess."),
    (r"\bfrom preprocess\b", "from core.preprocess"),
    (r"\bfrom prompts\.",   "from core.prompts."),
    (r"\bfrom prompts\b",   "from core.prompts"),
    (r"\bfrom chunking\.",  "from core.chunking."),
    (r"\bfrom chunking\b",  "from core.chunking"),
    (r"\bfrom llm\.",       "from core.llm."),
    (r"\bfrom llm\b",       "from core.llm"),
    (r"\bfrom loaders\.",   "from core.loaders."),
    (r"\bfrom loaders\b",   "from core.loaders"),
    (r"\bfrom routing\.",   "from core.routing."),
    (r"\bfrom routing\b",   "from core.routing"),
    (r"\bfrom retriever\.", "from core.retriever."),
    (r"\bfrom retriever\b", "from core.retriever"),
    (r"\bfrom vectorstore\.","from core.vectorstore."),
    (r"\bfrom vectorstore\b","from core.vectorstore"),

    # broken custom_tools imports
    (r"\bfrom agent\.async_utils\b", "from core.utils.async_utils"),
    (r"\bfrom agent\.workspace\b",   "from tools._workspace"),  # handled separately
]

# ---------------------------------------------------------------------------
# 3. Workspace helper replacements for file/git/terminal tools
#    These files import from agent.workspace which never existed — we inline.
# ---------------------------------------------------------------------------
WORKSPACE_IMPORT_RE = re.compile(
    r"from agent\.workspace import resolve_workspace_path(?:,\s*truncate_text)?|"
    r"from agent\.workspace import truncate_text(?:,\s*resolve_workspace_path)?"
)

WORKSPACE_HELPER = '''\
import os as _os
from pathlib import Path as _Path

def _resolve_workspace_path(filepath: str) -> _Path:
    base = _Path(_os.getenv("PROJECT_DIR", _Path(__file__).resolve().parents[3]))
    p = _Path(filepath)
    return p if p.is_absolute() else base / p

def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\\n... (truncated {len(text) - max_chars} chars)"
'''

WORKSPACE_USAGE_REWRITES = [
    (r"\bresolve_workspace_path\b", "_resolve_workspace_path"),
    (r"\btruncate_text\b", "_truncate_text"),
]

WORKSPACE_FILES = {
    "tools/file_tools.py",
    "tools/git_tools.py",
    "tools/terminal_tools.py",
}

# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------

def rewrite_imports(text: str, src_rel_path: str) -> str:
    """Apply all import rewrite rules to file content."""
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text)

    # Handle workspace imports for specific files
    if src_rel_path in WORKSPACE_FILES:
        if WORKSPACE_IMPORT_RE.search(text):
            text = WORKSPACE_IMPORT_RE.sub("", text)
            # Insert helper after module docstring / future imports
            lines = text.split("\n")
            insert_at = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # skip docstring
                    if i == 0:
                        # find end of docstring
                        for j in range(1, len(lines)):
                            if lines[j].strip().endswith('"""') or lines[j].strip().endswith("'''"):
                                insert_at = j + 1
                                break
                        break
                elif stripped.startswith("from __future__") or stripped.startswith("import ") or stripped.startswith("from "):
                    break
                elif stripped == "" and i > 0:
                    insert_at = i
                    break
            lines.insert(insert_at + 1, WORKSPACE_HELPER)
            text = "\n".join(lines)
        for pat, repl in WORKSPACE_USAGE_REWRITES:
            text = re.sub(pat, repl, text)

    # Fix agent.async_utils in file_search_tools and web_search_tools
    text = re.sub(r"\bfrom agent\.async_utils import\b", "from core.utils.async_utils import", text)

    return text


def copy_and_rewrite(src_rel: str, dst_rel: str) -> None:
    src_path = SRC / src_rel
    dst_path = SRC / dst_rel
    if not src_path.exists():
        print(f"  SKIP (missing): {src_rel}")
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    text = src_path.read_text(encoding="utf-8")
    text = rewrite_imports(text, dst_rel)
    dst_path.write_text(text, encoding="utf-8")
    print(f"  COPY {src_rel} → {dst_rel}")


def rewrite_file_in_place(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except Exception:
        return False
    rewritten = rewrite_imports(original, str(path.relative_to(SRC)))
    if rewritten != original:
        path.write_text(rewritten, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# 5. New files to create
# ---------------------------------------------------------------------------

CORE_INIT = '"""Core infrastructure — config, state, utils, prompts, chunking, llm, loaders, routing, retriever, vectorstore."""\n'
SUBAGENTS_INIT = '"""Subagents — intelligent workers: OCR (Qari), Embeddings (BGE-M3), Reranker, Document QA, Query Planner, PDF Analysis."""\n'

PDF_GENERATOR = '''\
"""PDF generation tool using PyMuPDF — supports Arabic (RTL) and English text."""
from __future__ import annotations

import html
import os
from pathlib import Path

try:
    from langchain.tools import tool
except ImportError:  # graceful fallback if langchain not installed
    def tool(fn):  # type: ignore[misc]
        return fn


def _split_pages(text: str, chars_per_page: int = 2800) -> list[str]:
    """Split long text into page-sized chunks."""
    text = text.strip()
    if not text:
        return [""]
    chunks: list[str] = []
    for start in range(0, len(text), chars_per_page):
        chunk = text[start : start + chars_per_page].strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [""]


@tool
def create_pdf(text: str, output_path: str = "output.pdf", title: str = "") -> str:
    """Create a PDF from the given text.

    Supports Arabic (RTL) and English. Uses PyMuPDF which is already installed.
    Returns the absolute path to the saved PDF.

    Args:
        text: The text content for the PDF.
        output_path: Where to save the PDF (default: output.pdf in current directory).
        title: Optional title shown at the top of the first page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        return f"ERROR: PyMuPDF not installed — {exc}"

    out = Path(output_path)
    if not out.is_absolute():
        out = Path(os.getcwd()) / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Detect primary direction: if >30% Arabic chars → RTL
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    is_rtl = arabic_chars / max(len(text), 1) > 0.3
    direction = "rtl" if is_rtl else "ltr"
    text_align = "right" if is_rtl else "left"

    safe_title = html.escape(title) if title else ""
    pages_text = _split_pages(text)

    doc = fitz.open()
    for page_num, chunk in enumerate(pages_text):
        page = doc.new_page(width=595, height=842)
        safe_body = html.escape(chunk)
        title_html = (
            f'<h2 style="font-size:14pt; color:#2c3e50; margin-bottom:8pt;">{safe_title}</h2>'
            if safe_title and page_num == 0
            else ""
        )
        html_content = (
            f'<div dir="{direction}" style="'
            f'font-family: sans-serif; font-size: 11pt; line-height: 1.6; '
            f'text-align: {text_align};">'
            f"{title_html}"
            f'<p style="white-space: pre-wrap;">{safe_body}</p>'
            f"</div>"
        )
        page.insert_htmlbox(fitz.Rect(40, 40, 555, 802), html_content)

    doc.save(str(out))
    doc.close()
    return str(out)


__all__ = ["create_pdf"]
'''

TOOLS_INIT = '''\
"""Tools package — LangChain @tool functions available to the agent."""
from __future__ import annotations

from tools.web_search import web_search_sync as web_search
from tools.file_tools import read_file, write_file, append_file, delete_file
from tools.file_search_tools import search_files
from tools.git_tools import git_status, git_diff, git_commit
from tools.terminal_tools import execute_terminal_command
from tools.pdf_generator import create_pdf
from tools.report_io import prepare_report_output_path


def get_agent_tools() -> list:
    """Return the safe subset of tools the main agent can bind to the LLM.

    Excludes destructive tools (delete_file, git_commit, browser_tools)
    from auto-binding.
    """
    return [
        web_search,
        read_file,
        write_file,
        create_pdf,
        execute_terminal_command,
        git_status,
        git_diff,
        search_files,
    ]


__all__ = [
    "get_agent_tools",
    "web_search",
    "read_file", "write_file", "append_file", "delete_file",
    "search_files",
    "git_status", "git_diff", "git_commit",
    "execute_terminal_command",
    "create_pdf",
    "prepare_report_output_path",
]
'''

# Updated agent __init__ to re-export from new subagents path
AGENT_INIT = '''\
"""Arabic Document Intelligence Agent — LangGraph entrypoint."""

from .graph import GRAPH_RUN_CONFIG, graph

__all__ = ["graph", "GRAPH_RUN_CONFIG"]
'''

SUBAGENTS_OCR_INIT = '''\
"""OCR subagent package — Qari VLM and digital text extraction."""
from subagents.ocr.factory import get_ocr_engine, extract_document_text
from subagents.ocr.base import OCRDocument, OCREngine, OCRError, OCRPage

__all__ = ["get_ocr_engine", "extract_document_text", "OCRDocument", "OCREngine", "OCRError", "OCRPage"]
'''

SUBAGENTS_EMBEDDINGS_INIT = '''\
"""Embeddings subagent package — BGE-M3 dense vector model."""
from subagents.embeddings.factory import (
    embed_documents,
    embed_query,
    embed_text,
    get_embedder,
    EMBEDDING_DIMENSIONS,
)

__all__ = ["embed_documents", "embed_query", "embed_text", "get_embedder", "EMBEDDING_DIMENSIONS"]
'''

# ---------------------------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Step 1: Copy files to new locations with import rewrites")
    print("=" * 60)
    for src_rel, dst_rel in COPIES:
        copy_and_rewrite(src_rel, dst_rel)

    print("\n" + "=" * 60)
    print("Step 2: Create __init__.py stubs for new packages")
    print("=" * 60)

    stubs: dict[str, str] = {
        "core/__init__.py": CORE_INIT,
        "subagents/__init__.py": SUBAGENTS_INIT,
        "subagents/ocr/__init__.py": SUBAGENTS_OCR_INIT,
        "subagents/embeddings/__init__.py": SUBAGENTS_EMBEDDINGS_INIT,
        "tools/__init__.py": TOOLS_INIT,
        "tools/pdf_generator.py": PDF_GENERATOR,
        "agent/__init__.py": AGENT_INIT,
    }
    for rel, content in stubs.items():
        path = SRC / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  WRITE {rel}")

    print("\n" + "=" * 60)
    print("Step 3: Rewrite imports in all remaining src/ agent/ files")
    print("=" * 60)
    # Rewrite graph.py and embeddings.py (still in agent/, now reference new paths)
    for rel in ["agent/graph.py", "agent/embeddings.py"]:
        p = SRC / rel
        if p.exists():
            changed = rewrite_file_in_place(p)
            print(f"  {'REWRITE' if changed else 'UNCHANGED'} {rel}")

    print("\n" + "=" * 60)
    print("Step 4: Rewrite imports in tests/, scripts/, streamlit_ui.py")
    print("=" * 60)
    targets: list[Path] = []
    for pat in ["tests/**/*.py", "scripts/*.py"]:
        targets.extend(ROOT.glob(pat))
    targets.append(ROOT / "streamlit_ui.py")
    for path in targets:
        if path.exists():
            try:
                original = path.read_text(encoding="utf-8")
            except Exception:
                continue
            rewritten = rewrite_imports(original, path.name)
            rel = path.relative_to(ROOT)
            if rewritten != original:
                path.write_text(rewritten, encoding="utf-8")
                print(f"  REWRITE {rel}")
            else:
                print(f"  UNCHANGED {rel}")

    print("\n" + "=" * 60)
    print("Step 5: Rewrite imports inside newly created src/core/ and src/subagents/")
    print("=" * 60)
    for py_file in (SRC / "core").rglob("*.py"):
        changed = rewrite_file_in_place(py_file)
        if changed:
            print(f"  REWRITE {py_file.relative_to(SRC)}")
    for py_file in (SRC / "subagents").rglob("*.py"):
        changed = rewrite_file_in_place(py_file)
        if changed:
            print(f"  REWRITE {py_file.relative_to(SRC)}")
    for py_file in (SRC / "tools").rglob("*.py"):
        changed = rewrite_file_in_place(py_file)
        if changed:
            print(f"  REWRITE {py_file.relative_to(SRC)}")

    print("\nDone. Old directories still present — verify tests pass then delete them.")
    print("Old dirs to remove: config/ state/ utils/ preprocess/ prompts/ chunking/")
    print("  llm/ loaders/ routing/ retriever/ vectorstore/ ocr/ embeddings/ reranker/ custom_tools/")
    print("  Also remove agent/document_qa.py agent/query_planner.py agent/pdf_analysis.py")


if __name__ == "__main__":
    main()

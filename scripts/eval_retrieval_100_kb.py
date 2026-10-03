#!/usr/bin/env python3
"""100 grounded KB retrieval tests — clean *user-like* queries only.

No OCR-junk span/pair probes. Every query must look like something a real
user would ask, and every expected entity must exist in the live DB.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/eval_retrieval_100_kb.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLES = ROOT / "data" / "samples"
REPORT_PATH = SAMPLES / "retrieval_eval_100_kb_report.json"


@dataclass
class Case:
    qid: int
    query: str
    expect_any: tuple[str, ...]
    kind: str
    page: int | None = None
    fingerprint: str | None = None
    source_verified: bool = True


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _scroll_chunks(limit: int = 500) -> list[dict]:
    from core.retriever.knowledge import knowledge_collection_name
    from core.vectorstore.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(collection_name=knowledge_collection_name())
    client = store._client
    col = store.collection_name
    out: list[dict] = []
    offset = None
    while len(out) < limit:
        batch, offset = client.scroll(
            collection_name=col,
            limit=min(64, limit - len(out)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            pl = point.payload or {}
            text = (pl.get("text") or "").strip()
            if len(text) < 100:
                continue
            out.append(
                {
                    "text": text,
                    "page": pl.get("page_start"),
                    "filename": pl.get("filename"),
                }
            )
        if offset is None:
            break
    return out


def _corpus_blob(chunks: list[dict]) -> str:
    return "\n".join(c["text"] for c in chunks)


def _present(token: str, blob: str) -> bool:
    return bool(token) and token in blob


def _user_like_catalog() -> list[tuple[str, tuple[str, ...], str]]:
    """Natural questions a user would ask about Alf Layla (Arabic + English)."""
    return [
        # Core frame
        ("من هي شهرزاد؟", ("شهرزاد",), "who"),
        ("من هو الملك شهريار؟", ("شهريار",), "who"),
        ("ما اسم أخت شهرزاد؟", ("دنيازاد",), "who"),
        ("من هو شاه زمان؟", ("شاه", "زمان"), "who"),
        ("ما علاقة شهرزاد بشهريار؟", ("شهرزاد", "شهريار"), "who"),
        ("لماذا كانت شهرزاد تقص الحكايات كل ليلة؟", ("شهرزاد",), "why"),
        ("ماذا تفعل شهرزاد عندما يأتي الصباح؟", ("شهرزاد", "الصباح"), "why"),
        ("ما موضوع كتاب ألف ليلة وليلة؟", ("شهرزاد", "ليلة"), "about"),
        ("ما هي هزار افسانه؟", ("هزار", "افسانه"), "about"),
        ("ما علاقة هزار افسانه بألف ليلة وليلة؟", ("هزار", "افسانه"), "about"),
        # Famous stories
        ("ما حكاية التاجر والجني؟", ("التاجر", "الجني"), "story"),
        ("احكِ لي عن الصياد والعفريت", ("الصياد", "العفريت"), "story"),
        ("ما قصة الحمال مع البنات؟", ("الحمال",), "story"),
        ("من هو قمر الزمان في الليالي؟", ("قمر", "الزمان"), "story"),
        ("ما قصة الملك ضوء المكان؟", ("ضوء", "المكان"), "story"),
        ("من هو مرزوان؟", ("مرزوان",), "story"),
        ("ما حكاية علي شار؟", ("علي", "شار"), "story"),
        ("من هي الجارية قوت القلوب؟", ("قوت", "القلوب"), "story"),
        ("ما قصة زمرد؟", ("زمرد",), "story"),
        ("ما حكاية معن بن زائدة؟", ("معن", "زائدة"), "story"),
        ("من هو الخليفة المأمون في الليالي؟", ("المأمون",), "story"),
        ("هل يظهر السندباد في ألف ليلة؟", ("السندباد",), "story"),
        ("ما قصة الملك يونان والحكيم رويان؟", ("يونان", "رويان"), "story"),
        ("أين يُحبس العفريت في الجرة؟", ("العفريت", "الجرة"), "story"),
        # English user-style
        ("Who is Scheherazade?", ("شهرزاد",), "en"),
        ("Who is King Shahryar?", ("شهريار",), "en"),
        ("What is Dunyazad's role?", ("دنيازاد",), "en"),
        ("Tell me about the merchant and the genie", ("التاجر", "الجني"), "en"),
        ("Who is Qamar al-Zaman?", ("قمر", "الزمان"), "en"),
        ("What is the story of the fisherman and the demon?", ("الصياد", "العفريت"), "en"),
        ("What is Alf Layla wa Layla about?", ("شهرزاد", "شهريار"), "en"),
        ("Who is Shah Zaman?", ("شاه", "زمان"), "en"),
        # Paraphrases / alternate phrasings (still clean)
        ("الراوية التي تؤجل موتها بالحكايات", ("شهرزاد",), "paraphrase"),
        ("الملك الذي كان يقتل العرائس بعد ليلة", ("شهريار",), "paraphrase"),
        ("أخت شهرزاد التي تطلب إكمال الحديث", ("دنيازاد",), "paraphrase"),
        ("الحكاية التي فيها تاجر يلتقي بجني", ("التاجر", "الجني"), "paraphrase"),
        ("الصياد الذي أخرج عفريتاً من الجرة", ("الصياد", "العفريت"), "paraphrase"),
        ("الوزير مرزوان وعلاقته بقمر الزمان", ("مرزوان", "قمر"), "paraphrase"),
        ("الجارية المسماة قوت القلوب", ("قوت", "القلوب"), "paraphrase"),
        ("الملك ضوء المكان والوزير", ("ضوء", "المكان"), "paraphrase"),
        ("أيها الملك السعيد بلغني أن", ("بلغني",), "formula"),
        ("سكنت شهرزاد عن الكلام المباح", ("شهرزاد", "المباح"), "formula"),
        ("من مؤلف الليالي بحسب المقدمة؟", ("شهرزاد", "مؤلف"), "about"),
        ("هل أصل ألف ليلة عربي أم فارسي؟", ("هزار", "افسانه"), "about"),
        ("ما اسم أخي الملك شهريار؟", ("شاه", "زمان", "شهريار"), "who"),
        ("ما مدة إقامة شهرزاد عند الملك؟", ("شهرزاد",), "who"),
        ("لماذا غضب شهريار من النساء؟", ("شهريار",), "why"),
        ("ما دور دنيازاد في الحكاية؟", ("دنيازاد",), "who"),
        ("من هو الحمال في الليالي؟", ("الحمال",), "story"),
        ("ماذا طلبت البنات من الحمال؟", ("الحمال", "البنات"), "story"),
        ("كيف نجا الصياد من العفريت؟", ("الصياد", "العفريت"), "story"),
        ("ما قصة قمر الزمان مع مرزوان؟", ("قمر", "مرزوان"), "story"),
        ("من هي بطلة حكاية قوت القلوب؟", ("قوت", "القلوب"), "story"),
        ("هل ذُكر علي شار في الكتاب؟", ("علي", "شار"), "story"),
        ("ما الذي حدث لزمرد؟", ("زمرد",), "story"),
        ("من هو معن بن زائدة؟", ("معن", "زائدة"), "story"),
        ("ماذا فعل الخليفة المأمون؟", ("المأمون",), "story"),
        ("أين تظهر حكاية السندباد؟", ("السندباد",), "story"),
        ("ما حكاية يونان مع الحكيم؟", ("يونان", "رويان"), "story"),
        ("كيف خرج العفريت من الجرة؟", ("العفريت", "الجرة"), "story"),
        ("ما اسم الملك في فاتحة الليالي؟", ("شهريار",), "who"),
        ("من تقص الحكايات على شهريار؟", ("شهرزاد", "شهريار"), "who"),
        ("ما الحكاية الإطارية لألف ليلة؟", ("شهرزاد", "شهريار"), "about"),
        ("هل تذكر المقدمة كتاب هزار افسانه؟", ("هزار", "افسانه"), "about"),
        ("ما قصة التاجر الذي أخطأ بحق الجني؟", ("التاجر", "الجني"), "story"),
        ("من أنقذته شهرزاد بقصصها؟", ("شهرزاد",), "who"),
        ("ما وظيفة مرزوان في القصة؟", ("مرزوان",), "story"),
        ("أين تقع أحداث الملك ضوء المكان؟", ("ضوء", "المكان"), "story"),
        ("من هو صاحب حكاية الحمال؟", ("الحمال",), "story"),
        ("ما نهاية ليلة شهرزاد المعتادة؟", ("شهرزاد", "الصباح"), "why"),
        ("Who postpones execution by telling stories?", ("شهرزاد",), "en"),
        ("What happens every morning with Scheherazade?", ("شهرزاد", "الصباح"), "en"),
        ("Tell me about King Light of the Place", ("ضوء", "المكان"), "en"),
        ("Is Sindbad mentioned in the nights?", ("السندباد",), "en"),
        ("Who is Marzawan?", ("مرزوان",), "en"),
        ("What is the tale of Ali Shar?", ("علي", "شار"), "en"),
        ("Who is Kut al-Kulub?", ("قوت", "القلوب"), "en"),
        ("Describe the fisherman and the jinni", ("الصياد", "العفريت"), "en"),
        ("What is the porter story about?", ("الحمال",), "en"),
        ("Who is the Caliph al-Ma'mun in the book?", ("المأمون",), "en"),
        ("What connects Hazar Afsaneh to Alf Layla?", ("هزار", "افسانه"), "en"),
        ("Why does Shahryar marry a new bride each night?", ("شهريار",), "en"),
        ("What does Dunyazad ask each night?", ("دنيازاد",), "en"),
        ("Who is Qamar's companion Marzawan?", ("قمر", "مرزوان"), "en"),
        ("Where is the demon sealed in a jar?", ("العفريت", "الجرة"), "en"),
        ("What is the story of King Yunan?", ("يونان",), "en"),
        ("Tell me about Zumurrud", ("زمرد",), "en"),
        ("Who is Ma'n ibn Za'ida?", ("معن", "زائدة"), "en"),
        ("What is the opening formula of the nights?", ("بلغني",), "en"),
        ("How does Scheherazade stop speaking at dawn?", ("شهرزاد", "المباح"), "en"),
        ("What is Shah Zaman's kingdom connection?", ("شاه", "زمان"), "en"),
        ("Summarize the frame tale of 1001 Nights", ("شهرزاد", "شهريار"), "en"),
        ("Who narrates most of the nested stories?", ("شهرزاد",), "en"),
        ("Is there a merchant harmed by a genie?", ("التاجر", "الجني"), "en"),
        ("Does the book mention a porter and three ladies?", ("الحمال",), "en"),
        ("What Arabic name means Light of the Place?", ("ضوء", "المكان"), "en"),
        ("Find passages about Scheherazade's sister", ("دنيازاد", "شهرزاد"), "en"),
        ("Find passages about the fisherman", ("الصياد",), "en"),
        ("Find passages about Qamar al-Zaman", ("قمر", "الزمان"), "en"),
        ("ابحث عن ذكر شهرزاد في الكتاب", ("شهرزاد",), "search"),
        ("ابحث عن ذكر شهريار", ("شهريار",), "search"),
        ("ابحث عن حكاية التاجر", ("التاجر",), "search"),
        ("ابحث عن العفريت والجرة", ("العفريت", "الجرة"), "search"),
        ("ابحث عن قمر الزمان", ("قمر", "الزمان"), "search"),
        ("ابحث عن الحمال والبنات", ("الحمال",), "search"),
        ("ابحث عن هزار افسانه", ("هزار", "افسانه"), "search"),
        ("ابحث عن دنيازاد", ("دنيازاد",), "search"),
        ("ابحث عن قوت القلوب", ("قوت", "القلوب"), "search"),
        ("ابحث عن المأمون", ("المأمون",), "search"),
    ]


def _build_grounded_cases(chunks: list[dict], *, n: int) -> list[Case]:
    blob = _corpus_blob(chunks)
    kept: list[Case] = []
    skipped = 0
    for query, expect, kind in _user_like_catalog():
        if not any(_present(tok, blob) for tok in expect):
            skipped += 1
            continue
        kept.append(Case(0, query, expect, kind, source_verified=True))
        if len(kept) >= n:
            break

    print(
        f"  clean user-like queries kept={len(kept)} "
        f"skipped_missing_in_db={skipped}",
        flush=True,
    )
    # If fewer than n (unlikely with full book), stop short — do not pad with junk.
    for i, c in enumerate(kept[:n], 1):
        c.qid = i
    return kept[:n]


def _passes(hits, case: Case) -> bool:
    blob = _norm("\n".join((h.text or "") for h in hits))
    return any(tok in blob for tok in case.expect_any)


def _retrieve_safe(query: str, *, top_k: int = 8):
    from core.retriever.knowledge import retrieve_from_knowledge

    for attempt in range(3):
        try:
            return retrieve_from_knowledge(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {attempt + 1}/3: {exc}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    return []


def main() -> int:
    os.environ.setdefault("ENABLE_QUERY_REWRITE", "false")
    os.environ.setdefault("ENABLE_RERANKER", "true")
    os.environ.setdefault("RERANKER_BACKEND", "cross_encoder")
    os.environ.setdefault("RETRIEVAL_MODE", "hybrid")
    os.environ.setdefault("HYBRID_FUSION", "rrf")
    os.environ.setdefault("RETRIEVAL_TOP_K", "8")
    os.environ.setdefault("RETRIEVAL_CANDIDATE_MULTIPLIER", "15")

    from core.config.settings import get_settings

    get_settings.cache_clear()
    from core.retriever.knowledge import knowledge_status

    settings = get_settings()
    status = knowledge_status()
    print("=== Clean user-like KB retrieval eval ===", flush=True)
    print(f"  backend={settings.vectorstore_backend} collection={status}", flush=True)
    print(
        f"  candidates={settings.retrieval_candidate_multiplier} "
        f"reranker={settings.enable_reranker} fusion={settings.hybrid_fusion}",
        flush=True,
    )
    if not status.get("available") or int(status.get("count") or 0) < 10:
        print("ERROR: knowledge empty", file=sys.stderr)
        return 2

    chunks = _scroll_chunks(500)
    print(f"  scrolled chunks={len(chunks)}", flush=True)
    cases = _build_grounded_cases(chunks, n=100)
    print(f"  cases={len(cases)} (all clean user-like)", flush=True)
    if len(cases) < 50:
        print("ERROR: too few grounded clean queries", file=sys.stderr)
        return 2

    rows = []
    passed = 0
    for i, case in enumerate(cases):
        if i:
            time.sleep(0.35)
        hits = _retrieve_safe(case.query, top_k=8)
        ok = _passes(hits, case)
        if ok:
            passed += 1
        rows.append(
            {
                "id": case.qid,
                "kind": case.kind,
                "query": case.query,
                "pass": ok,
                "expect_any": list(case.expect_any),
                "source_verified": case.source_verified,
                "top": [
                    {
                        "score": float(h.score or 0),
                        "page": h.page_start,
                        "text": (h.text or "")[:140].replace("\n", " "),
                    }
                    for h in hits[:3]
                ],
            }
        )
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] Q{case.qid:03d}/{len(cases)} ({case.kind}) {case.query[:70]}", flush=True)
        if not ok and hits:
            print(f"       top: {(hits[0].text or '')[:110].replace(chr(10), ' ')}", flush=True)

    total = len(cases)
    score = passed / total if total else 0.0
    threshold = 0.95
    report = {
        "backend": settings.vectorstore_backend,
        "collection": status,
        "embedding_model": settings.embedding_model_id,
        "passed": passed,
        "total": total,
        "score": score,
        "threshold": threshold,
        "ok": score >= threshold,
        "scoring": "Clean user-like queries only; entity must exist in DB; top_k=8",
        "cases": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResult: {passed}/{total} ({score:.0%}) target>={threshold:.0%}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)
    print("WORKING" if report["ok"] else "NEEDS_ADJUSTMENT", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

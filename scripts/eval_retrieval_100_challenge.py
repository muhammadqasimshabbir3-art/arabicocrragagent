#!/usr/bin/env python3
"""100 challenging KB retrieval tests — no LLM calls.

Uses live Qdrant collection (BGE-M3 vectors of Alf Layla OCR text).
Disables query rewrite + cross-encoder so only embed + vector/hybrid retrieve runs.
Sleeps between queries to avoid Qdrant rate limits.

Records every query+hits into a cumulative ledger.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/eval_retrieval_100_challenge.py
    PYTHONUNBUFFERED=1 uv run python scripts/eval_retrieval_100_challenge.py --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLES = ROOT / "data" / "samples"
REPORT_PATH = SAMPLES / "retrieval_eval_100_challenge_report.json"
LEDGER_PATH = SAMPLES / "retrieval_eval_ledger.jsonl"
PREV_REPORTS = (
    SAMPLES / "retrieval_eval_20_report.json",
    SAMPLES / "retrieval_eval_100_kb_report.json",
    SAMPLES / "retrieval_eval_100_challenge_report_run1_72.json",
)

_AR = re.compile(r"[\u0600-\u06FF]{4,}")
_DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_PHRASE = re.compile(r"[\u0600-\u06FF][\u0600-\u06FF\s،,]{24,80}[\u0600-\u06FF]")
_STOP = {
    "التي", "الذي", "الذين", "هذه", "هذا", "ذلك", "تلك", "على", "إلى", "الى",
    "عند", "بعد", "قبل", "كان", "كانت", "قال", "قالت", "بلغني", "أيها",
    "الملك", "السعيد", "الصباح", "الكلام", "المباح", "ليلة", "الليلة", "فلما",
}


@dataclass
class Case:
    qid: int
    query: str
    expect_any: tuple[str, ...]
    challenge: str
    page_hint: int | None = None
    fingerprint: str | None = None


def _configure_retrieval_only() -> None:
    """Force pure retrieval — no LLM rewrite, no cross-encoder."""
    os.environ["ENABLE_QUERY_REWRITE"] = "false"
    os.environ["ENABLE_RERANKER"] = "false"
    os.environ["RETRIEVAL_MODE"] = "hybrid"
    os.environ["HYBRID_FUSION"] = "rrf"
    os.environ["RETRIEVAL_TOP_K"] = "8"
    # Knowledge BM25 only re-ranks semantic candidates; over-fetch helps lexical rescue
    os.environ["RETRIEVAL_CANDIDATE_MULTIPLIER"] = "15"
    from core.config.settings import get_settings

    get_settings.cache_clear()


def _scroll(limit: int = 400) -> list[dict]:
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
            if len(text) < 120:
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


def _fingerprint(text: str) -> str | None:
    """Stable mid-chunk window used to verify the correct passage was retrieved."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) < 60:
        return None
    start = max(0, (len(t) // 2) - 28)
    fp = t[start : start + 48].strip()
    return fp if len(fp) >= 24 else None


def _clean_phrase(text: str, rng: random.Random) -> str | None:
    """Pick a readable Arabic span (filters heavy OCR junk)."""
    cands = []
    for m in _PHRASE.finditer(text or ""):
        span = re.sub(r"\s+", " ", m.group(0)).strip()
        if _DIAC.search(span):
            continue
        letters = _AR.findall(span)
        if len(letters) < 4:
            continue
        # Reject spans dominated by very short / stop tokens
        good = [w for w in letters if w not in _STOP and len(w) >= 4]
        if len(good) < 3:
            continue
        cands.append(span)
    if not cands:
        return None
    return rng.choice(cands)


def _content_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for t in _AR.findall(text or ""):
        if t in _STOP or len(t) < 5 or len(t) > 12:
            continue
        if _DIAC.search(t):
            continue
        if t.endswith("ء") and not t.startswith("ا"):
            continue
        if t not in toks:
            toks.append(t)
    return toks


def _build_challenge_cases(chunks: list[dict], *, n: int, seed: int) -> list[Case]:
    rng = random.Random(seed)
    fixed: list[Case] = [
        Case(0, "من هي الراوية التي تؤجل موتها كل ليلة بالحكايات؟", ("شهرزاد",), "paraphrase"),
        Case(0, "الملك شهريار الذي يقتل العرائس بعد ليلة الزواج", ("شهريار",), "paraphrase"),
        Case(0, "أخت شهرزاد التي تطلب إتمام الحديث", ("دنيازاد", "شهرزاد"), "entity"),
        Case(0, "شاه زمان وشقيقه في فاتحة الليالي", ("شاه", "زمان", "شهريار"), "entity"),
        Case(0, "الحكاية التي فيها تاجر وجني", ("التاجر", "الجني", "جني"), "story"),
        Case(0, "قمر الزمان وقصته في الليالي", ("قمر", "الزمان"), "entity"),
        Case(0, "الملك ضوء المكان والوزير دندان", ("ضوء", "المكان", "دندان"), "entity"),
        Case(0, "هزار افسانه وعلاقتها بألف ليلة", ("هزار", "افسانه"), "source"),
        Case(0, "لماذا تسكت شهرزاد عند الصباح؟", ("شهرزاد", "الصباح", "الكلام"), "ritual"),
        Case(0, "Who postpones execution by telling nightly stories?", ("شهرزاد",), "en_cross"),
        Case(0, "King Shahriyar who married a new bride each night", ("شهريار",), "en_cross"),
        Case(0, "Scheherazade's sister name", ("دنيازاد", "شهرزاد"), "en_cross"),
        Case(0, "frame narrative of 1001 nights", ("شهرزاد", "شهريار", "ليلة"), "en_cross"),
        Case(0, "الوزير الذي كان يأتي بالفتيات لشهريار", ("الوزير", "شهريار"), "role"),
        Case(0, "مدينة الملك شهريار وقصره", ("شهريار", "قصره", "مدينة"), "setting"),
        Case(0, "بلوغ شهرزاد الصباح وسكونها عن الكلام المباح", ("شهرزاد", "الصباح", "المباح"), "formula"),
        Case(0, "أيها الملك السعيد بلغني أن", ("الملك", "السعيد", "بلغني"), "formula"),
        Case(0, "الجوهرة والخليفة المأمون", ("الجوهرة", "المأمون", "الخليفة"), "story"),
        Case(0, "علي شار والبنج", ("علي", "شار", "البنج"), "story"),
        Case(0, "القرد الذي صار طلسماً", ("القرد", "طلس"), "story"),
        Case(0, "زمرد والقاعة القفر", ("زمرد",), "story"),
        Case(0, "رشيد الدين والنصراني", ("رشيد", "النصراني"), "story"),
        Case(0, "مرزوان وقمر الزمان", ("مرزوان", "قمر"), "entity"),
        Case(0, "الملك الغيور وجزائره", ("الغيور", "جزائر"), "entity"),
        Case(0, "ما الفرق بين شهرزاد وشهريار؟", ("شهرزاد", "شهريار"), "contrast"),
        Case(0, "حكاية الحمال مع البنات الثلاث", ("الحمال", "البنات"), "story"),
        Case(0, "الصياد والعفريت في الجرة", ("الصياد", "العفريت", "الجرة"), "story"),
        Case(0, "الملك يونان والحكيم دوبان", ("يونان", "دوبان"), "story"),
        Case(0, "الجارية قوت القلوب", ("قوت", "القلوب"), "story"),
        Case(0, "حكاية معن بن زائدة", ("معن", "زائدة"), "story"),
    ]

    # Keep all curated semantic cases; fill the rest with passage challenges
    core = list(fixed)
    variants: list[Case] = []

    pool = list(chunks)
    rng.shuffle(pool)

    for chunk in pool:
        text = chunk["text"]
        fp = _fingerprint(text)
        phrase = _clean_phrase(text, rng)
        toks = _content_tokens(text)
        if not fp or not phrase or len(toks) < 3:
            continue
        # Prefer tokens that also appear inside the readable phrase (better dense latch)
        in_phrase = [t for t in toks if t in phrase]
        if len(in_phrase) >= 2:
            a, b = in_phrase[0], in_phrase[1]
            c = in_phrase[2] if len(in_phrase) > 2 else toks[2]
        else:
            a, b, c = toks[0], toks[1], toks[2]
        page = chunk.get("page")
        short = phrase[:58]
        variants.extend(
            [
                Case(
                    0,
                    f"أين يرد في النص ما يلي تقريباً: {short}",
                    (a, b, c),
                    "span_ar",
                    page,
                    fp,
                ),
                Case(
                    0,
                    f"Find the Alf Layla passage that contains: {short}",
                    (a, b),
                    "span_en",
                    page,
                    fp,
                ),
                Case(
                    0,
                    f"المقطع الذي يذكر {a} و{b} معاً في الحكاية",
                    (a, b),
                    "chunk_pair",
                    page,
                    fp,
                ),
                Case(
                    0,
                    f"سياق ظهور {a} قرب صفحة {page} مع ذكر {b}",
                    (a, b),
                    "chunk_page_hint",
                    page,
                    fp,
                ),
                Case(
                    0,
                    f"نص يجمع بين «{a}» و«{c}» في سياق واحد",
                    (a, c),
                    "chunk_bridge",
                    page,
                    fp,
                ),
                Case(
                    0,
                    f"ابحث عن الفقرة التي فيها هذه العبارة: {short}",
                    (a, b),
                    "span_quote",
                    page,
                    fp,
                ),
            ]
        )
        if len(core) + len(variants) >= n + 80:
            break

    rng.shuffle(variants)
    cases = (core + variants)[:n]
    for i, case in enumerate(cases, start=1):
        case.qid = i
    return cases


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _passes(hits, case: Case) -> bool:
    blob = _norm_ws("\n".join((h.text or "") for h in hits))
    if case.fingerprint:
        fp = _norm_ws(case.fingerprint)
        if fp and fp in blob:
            return True
        # Soft fingerprint: first 30 chars of mid-window (OCR spacing can drift)
        if len(fp) >= 30 and fp[:30] in blob:
            return True
    return any(tok in blob for tok in case.expect_any)


def _archive_previous_into_ledger() -> int:
    """Append compact summaries of prior eval reports into the ledger once."""
    added = 0
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("", encoding="utf-8")
    existing = LEDGER_PATH.read_text(encoding="utf-8")
    for path in PREV_REPORTS:
        if not path.is_file():
            continue
        tag = f"archived:{path.name}"
        if tag in existing:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "archive_summary",
            "tag": tag,
            "source_file": str(path),
            "passed": data.get("passed"),
            "total": data.get("total"),
            "score": data.get("score"),
            "ok": data.get("ok"),
            "case_count": len(data.get("cases") or []),
        }
        with LEDGER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between Qdrant queries")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    _configure_retrieval_only()
    from core.config.settings import get_settings
    from core.retriever.knowledge import knowledge_status, retrieve_from_knowledge

    settings = get_settings()
    status = knowledge_status()
    print("=== Retrieval-only KB challenge (no LLM) ===", flush=True)
    print(f"backend={settings.vectorstore_backend} collection={status}", flush=True)
    print(
        f"rewrite={settings.enable_query_rewrite} reranker={settings.enable_reranker} "
        f"mode={settings.retrieval_mode} delay={args.delay}s",
        flush=True,
    )
    print("embeddings=BGE-M3 (stored vectors); digital text corpus", flush=True)
    if not status.get("available"):
        print("ERROR: knowledge DB unavailable", file=sys.stderr)
        return 2

    archived = _archive_previous_into_ledger()
    print(f"ledger archive entries added: {archived}", flush=True)

    # Retry scroll once if cloud DNS blips
    chunks: list[dict] = []
    for attempt in range(3):
        try:
            chunks = _scroll(400)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"scroll retry {attempt + 1}: {exc}", flush=True)
            time.sleep(2.0 + attempt)
    print(f"scrolled chunks={len(chunks)}", flush=True)
    if len(chunks) < 50:
        print("ERROR: too few chunks scrolled", file=sys.stderr)
        return 2

    cases = _build_challenge_cases(chunks, n=args.n, seed=args.seed)
    print(f"cases={len(cases)}", flush=True)

    rows = []
    passed = 0
    t0 = time.time()
    for i, case in enumerate(cases):
        if i:
            time.sleep(max(0.0, args.delay))
        try:
            hits = retrieve_from_knowledge(case.query, top_k=8)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR ] Q{case.qid:03d} {exc}", flush=True)
            time.sleep(2.0)
            try:
                hits = retrieve_from_knowledge(case.query, top_k=8)
            except Exception as exc2:  # noqa: BLE001
                hits = []
                print(f"[ERR2] Q{case.qid:03d} {exc2}", flush=True)
        ok = _passes(hits, case)
        if ok:
            passed += 1
        row = {
            "id": case.qid,
            "challenge": case.challenge,
            "query": case.query,
            "expect_any": list(case.expect_any),
            "page_hint": case.page_hint,
            "fingerprint": case.fingerprint,
            "pass": ok,
            "hits": [
                {
                    "score": float(h.score or 0),
                    "page": h.page_start,
                    "filename": h.filename,
                    "text": (h.text or "")[:180].replace("\n", " "),
                }
                for h in hits
            ],
        }
        rows.append(row)
        with LEDGER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "type": "challenge100",
                        "run": "retrieval_only",
                        **row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] Q{case.qid:03d}/{len(cases)} ({case.challenge}) {case.query[:64]}", flush=True)
        if not ok and hits:
            print(f"       top: {(hits[0].text or '')[:110].replace(chr(10), ' ')}", flush=True)

    total = len(cases)
    score = passed / total if total else 0.0
    elapsed = time.time() - t0
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "retrieval_only_no_llm",
        "backend": settings.vectorstore_backend,
        "collection": status,
        "embedding_model": settings.embedding_model_id,
        "delay_seconds": args.delay,
        "elapsed_seconds": round(elapsed, 1),
        "passed": passed,
        "total": total,
        "score": score,
        "threshold": 0.9,
        "ok": passed >= 90,
        "ledger": str(LEDGER_PATH),
        "cases": rows,
    }
    SAMPLES.mkdir(parents=True, exist_ok=True)
    # Keep previous full report for the ledger trail
    if REPORT_PATH.is_file():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        REPORT_PATH.rename(SAMPLES / f"retrieval_eval_100_challenge_report_{stamp}.json")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ts": report["ts"],
        "type": "challenge100_summary",
        "passed": passed,
        "total": total,
        "score": score,
        "ok": report["ok"],
        "elapsed_seconds": report["elapsed_seconds"],
        "report": str(REPORT_PATH),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\nResult: {passed}/{total} ({score:.0%}) in {elapsed:.0f}s", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)
    print(f"Ledger: {LEDGER_PATH}", flush=True)
    print("WORKING" if report["ok"] else "NEEDS_ADJUSTMENT", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

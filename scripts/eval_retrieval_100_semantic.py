#!/usr/bin/env python3
"""100 thematic semantic paraphrase retrieval tests.

Same meaning as core Alf Layla topics, but *different wording* —
synonyms, descriptive paraphrases, AR↔EN cross-language, and
colloquial rephrasings. Measures whether hybrid retrieval is robust
to how users actually phrase the same intent.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/eval_retrieval_100_semantic.py
"""

from __future__ import annotations

import argparse
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
REPORT_PATH = SAMPLES / "retrieval_eval_100_semantic_report.json"


@dataclass
class Case:
    qid: int
    query: str
    expect_any: tuple[str, ...]
    theme: str
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


def _semantic_catalog() -> list[tuple[str, tuple[str, ...], str]]:
    """Same thematic meaning, deliberately different surface forms.

    Each `theme` groups paraphrases that should retrieve the same entities.
    Queries avoid copying the entity name when possible (semantic challenge).
    """
    return [
        # ── theme: scheherazade (storyteller who delays death) ─────────────
        ("من الفتاة التي تؤجل موتها بالحكايات كل ليلة؟", ("شهرزاد",), "scheherazade"),
        ("الراوية التي تنقذ حياتها بسرد القصص", ("شهرزاد",), "scheherazade"),
        ("من تقص الحكايات على الملك حتى يطلع الفجر؟", ("شهرزاد",), "scheherazade"),
        ("الفتاة التي تقطع كلامها عند الصباح", ("شهرزاد",), "scheherazade"),
        ("The woman who survives by telling nightly tales", ("شهرزاد",), "scheherazade"),
        ("Who buys one more day of life with a story?", ("شهرزاد",), "scheherazade"),
        ("narrator of the nested Arabian nights tales", ("شهرزاد",), "scheherazade"),
        ("البطلة التي تؤجل حكم الإعدام بالكلام", ("شهرزاد",), "scheherazade"),
        # ── theme: shahryar (king who kills brides) ────────────────────────
        ("الملك الذي كان يقتل زوجاته بعد ليلة واحدة", ("شهريار",), "shahryar"),
        ("من الملك المنتقم من النساء في فاتحة الليالي؟", ("شهريار",), "shahryar"),
        ("الحاكم الذي يتزوج عذراء كل ليلة ثم يأمر بقتلها", ("شهريار",), "shahryar"),
        ("The vengeful king in the frame story of 1001 Nights", ("شهريار",), "shahryar"),
        ("Which ruler loses trust in women after betrayal?", ("شهريار",), "shahryar"),
        ("ملك ساسان صاحب الجند الذي غدرت به زوجته", ("شهريار",), "shahryar"),
        ("who listens to Scheherazade's stories each night", ("شهريار",), "shahryar"),
        # ── theme: dunyazad (sister who asks for stories) ──────────────────
        ("أخت الراوية التي تطلب إكمال الحكاية", ("دنيازاد",), "dunyazad"),
        ("من تطلب من أختها أن تحكي للملك؟", ("دنيازاد",), "dunyazad"),
        ("الصغيرة التي توقظ الفضول لسماع بقية القصة", ("دنيازاد",), "dunyazad"),
        ("Scheherazade's sister who prompts the tales", ("دنيازاد",), "dunyazad"),
        ("the sibling who asks to hear the rest of the story", ("دنيازاد",), "dunyazad"),
        # ── theme: shah zaman (brother king) ───────────────────────────────
        ("أخ الملك الذي حكم سمرقند", ("شاه", "زمان"), "shah_zaman"),
        ("الملك الشقيق في فاتحة ألف ليلة", ("شاه", "زمان"), "shah_zaman"),
        ("Shahryar's brother who ruled Samarkand", ("شاه", "زمان"), "shah_zaman"),
        ("من هو أخو شهريار في المقدمة؟", ("شاه", "زمان"), "shah_zaman"),
        # ── theme: merchant & genie ────────────────────────────────────────
        ("قصة تاجر يُخطئ بحق جني تحت الشجرة", ("التاجر", "الجني"), "merchant_genie"),
        ("الحكاية التي يبدأ فيها تاجر بلقاء كائن جنّي", ("التاجر", "الجني"), "merchant_genie"),
        ("a merchant who harms a genie by throwing a date pit", ("التاجر", "الجني"), "merchant_genie"),
        ("التاجر الذي استحق الموت من الجني", ("التاجر", "الجني"), "merchant_genie"),
        ("Tell me the tale where a trader meets a jinni", ("التاجر", "الجني"), "merchant_genie"),
        # ── theme: fisherman & demon in jar ────────────────────────────────
        ("من أخرج عفريتاً محبوساً من وعاء؟", ("الصياد", "العفريت"), "fisherman"),
        ("القصة عن صياد يفتح جرة مسحورة", ("الصياد", "العفريت", "الجرة"), "fisherman"),
        ("fisherman who frees a sealed demon from a jar", ("الصياد", "العفريت"), "fisherman"),
        ("كيف ينجو رجل من عفريت أراد قتله بعد تحريره؟", ("الصياد", "العفريت"), "fisherman"),
        ("العفريت المحبوس في النحاس والخاتم", ("العفريت", "الجرة"), "fisherman"),
        ("the demon trapped in a bottle by Solomon", ("العفريت", "الجرة"), "fisherman"),
        # ── theme: porter / hammāl ─────────────────────────────────────────
        ("قصة الحمال مع ثلاث سيدات في بغداد", ("الحمال",), "porter"),
        ("the porter invited by three ladies", ("الحمال",), "porter"),
        ("من حمل البضائع ثم دخل دار البنات؟", ("الحمال",), "porter"),
        ("حكاية الرجل الذي يعمل حمّالاً مع الفتيات", ("الحمال",), "porter"),
        # ── theme: qamar al-zaman ──────────────────────────────────────────
        ("الأمير الذي رفض الزواج ثم وقع في حب أميرة", ("قمر", "الزمان"), "qamar"),
        ("قصة قمر مع وزيره الوفي", ("قمر", "مرزوان"), "qamar"),
        ("prince Qamar and his loyal companion", ("قمر", "مرزوان"), "qamar"),
        ("من هو الأمير المرتبط باسم الزمان في الليالي؟", ("قمر", "الزمان"), "qamar"),
        ("حكاية الأمير والأميرة والجنّية ميمونة", ("قمر",), "qamar"),
        # ── theme: light of the place (daw' al-makan) ──────────────────────
        ("الملك الملقب بضوء المكان", ("ضوء", "المكان"), "daw_al_makan"),
        ("King whose title means Light of the Place", ("ضوء", "المكان"), "daw_al_makan"),
        ("من يُدعى ضوء المكان في الحكايات؟", ("ضوء", "المكان"), "daw_al_makan"),
        ("قصة الملك ضوء المكان ووزيره", ("ضوء", "المكان"), "daw_al_makan"),
        # ── theme: marzawan ────────────────────────────────────────────────
        ("الوزير الصديق لقمر الزمان", ("مرزوان",), "marzawan"),
        ("loyal vizier companion of the prince", ("مرزوان",), "marzawan"),
        ("من يساعد قمر في رحلته من الوزراء؟", ("مرزوان", "قمر"), "marzawan"),
        # ── theme: kut al-kulub ────────────────────────────────────────────
        ("الجارية ذات الاسم المرتبط بالقلوب", ("قوت", "القلوب"), "kut_al_kulub"),
        ("the slave-girl named Strength of Hearts", ("قوت", "القلوب"), "kut_al_kulub"),
        ("من هي قوت القلوب في الليالي؟", ("قوت", "القلوب"), "kut_al_kulub"),
        ("حكاية الجارية قوت", ("قوت", "القلوب"), "kut_al_kulub"),
        # ── theme: ali shar ────────────────────────────────────────────────
        ("حكاية علي شار في الكتاب", ("علي", "شار"), "ali_shar"),
        ("Is Ali Shar one of the nights' tales?", ("علي", "شار"), "ali_shar"),
        ("من بطل حكاية علي شار؟", ("علي", "شار"), "ali_shar"),
        # ── theme: zumurrud ────────────────────────────────────────────────
        ("من هي زمرد في الحكايات؟", ("زمرد",), "zumurrud"),
        ("the heroine called Zumurrud / emerald", ("زمرد",), "zumurrud"),
        ("قصة الجارية زمرد", ("زمرد",), "zumurrud"),
        # ── theme: ma'n ibn za'ida ─────────────────────────────────────────
        ("حكاية معن بن زائدة", ("معن", "زائدة"), "man_ibn_zaida"),
        ("Who is Ma'n ibn Za'ida in the nights?", ("معن", "زائدة"), "man_ibn_zaida"),
        ("شخصية معن في الليالي", ("معن",), "man_ibn_zaida"),
        # ── theme: caliph al-ma'mun ────────────────────────────────────────
        ("الخليفة العباسي المأمون في الليالي", ("المأمون",), "mamun"),
        ("Which Abbasid caliph appears in these tales?", ("المأمون",), "mamun"),
        ("ماذا يُروى عن المأمون في الكتاب؟", ("المأمون",), "mamun"),
        # ── theme: yunan & sage ────────────────────────────────────────────
        ("الملك يونان والحكيم الذي عالجه", ("يونان", "رويان"), "yunan"),
        ("King Yunan and the wise physician", ("يونان",), "yunan"),
        ("قصة الحكيم رويان مع الملك", ("يونان", "رويان"), "yunan"),
        # ── theme: hazar afsaneh / origins ─────────────────────────────────
        ("ما أصل الليالي الفارسي المذكور في المقدمة؟", ("هزار", "افسانه"), "origins"),
        ("the Persian book of a thousand legends", ("هزار", "افسانه"), "origins"),
        ("كيف ترتبط هزار افسانه بألف ليلة؟", ("هزار", "افسانه"), "origins"),
        ("هل الليالي مترجمة عن أصل فارسي؟", ("هزار", "افسانه"), "origins"),
        ("What is Hazar Afsaneh?", ("هزار", "افسانه"), "origins"),
        # ── theme: frame tale / 1001 nights topic ──────────────────────────
        ("ما الحكاية الإطارية التي تجمع الليالي؟", ("شهرزاد", "شهريار"), "frame"),
        ("summarize the outer story of One Thousand and One Nights", ("شهرزاد", "شهريار"), "frame"),
        ("لماذا تُروى كل هذه الحكايات داخل قصة أكبر؟", ("شهرزاد",), "frame"),
        ("the framing narrative of Alf Layla", ("شهرزاد", "شهريار"), "frame"),
        ("عن ماذا يدور كتاب ألف ليلة وليلة في جملته؟", ("شهرزاد", "شهريار"), "frame"),
        # ── theme: dawn formula / opening formula ──────────────────────────
        ("العبارة التي تختم بها شهرزاد كلامها عند الفجر", ("شهرزاد", "المباح"), "formula"),
        ("How does the storyteller fall silent at sunrise?", ("شهرزاد", "المباح"), "formula"),
        ("افتتاحية «أيها الملك السعيد بلغني أن»", ("بلغني",), "formula"),
        ("the classic opening 'O happy king, I have heard that'", ("بلغني",), "formula"),
        ("سكنت عن الكلام المباح — ماذا تعني؟", ("المباح", "شهرزاد"), "formula"),
        # ── theme: sindbad (may be sparse in corpus) ───────────────────────
        ("هل ذُكر بحّار شهير باسم السندباد؟", ("السندباد",), "sindbad"),
        ("Is the sailor Sindbad present in this edition?", ("السندباد",), "sindbad"),
        ("ابحث عن حكاية الرحّال السندباد", ("السندباد",), "sindbad"),
        # ── theme: cross-lingual same intent ───────────────────────────────
        ("tell me who the nightly storyteller is", ("شهرزاد",), "crosslang"),
        ("أريد معرفة هوية الملك القاسي على العرائس", ("شهريار",), "crosslang"),
        ("who is the sister that requests more stories?", ("دنيازاد",), "crosslang"),
        ("find the fisherman-and-jar episode", ("الصياد", "العفريت"), "crosslang"),
        ("دلّني على قصة التاجر مع الكائن الخارق", ("التاجر", "الجني"), "crosslang"),
        ("where do we read about Light-of-the-Place king?", ("ضوء", "المكان"), "crosslang"),
        ("من الراوية الأشهر في التراث العربي للحكايات؟", ("شهرزاد",), "crosslang"),
        ("describe the jar-bound supernatural being", ("العفريت", "الجرة"), "crosslang"),
        ("companion vizier of Qamar — who is he?", ("مرزوان",), "crosslang"),
        ("ابحث عن أصل الكتاب الفارسي المذكور", ("هزار", "افسانه"), "crosslang"),
        ("the girl whose name means food/strength of hearts", ("قوت", "القلوب"), "crosslang"),
        ("من يؤجل القتل بسرد القصص المتداخلة؟", ("شهرزاد",), "crosslang"),
    ]


def _build_cases(chunks: list[dict], *, n: int) -> list[Case]:
    blob = _corpus_blob(chunks)
    kept: list[Case] = []
    skipped = 0
    for query, expect, theme in _semantic_catalog():
        if not any(_present(tok, blob) for tok in expect):
            skipped += 1
            continue
        kept.append(Case(0, query, expect, theme, source_verified=True))
        if len(kept) >= n:
            break

    print(
        f"  semantic paraphrases kept={len(kept)} "
        f"skipped_missing_in_db={skipped}",
        flush=True,
    )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-boost",
        action="store_true",
        help="Use LLM query booster (2-line notes) for ambiguous queries.",
    )
    parser.add_argument(
        "--no-llm-boost",
        action="store_true",
        help="Disable LLM booster and run raw semantic queries only.",
    )
    args = parser.parse_args()

    os.environ.setdefault("ENABLE_QUERY_REWRITE", "false")
    os.environ.setdefault("ENABLE_RERANKER", "true")
    os.environ.setdefault("RERANKER_BACKEND", "cross_encoder")
    os.environ.setdefault("RETRIEVAL_MODE", "hybrid")
    os.environ.setdefault("HYBRID_FUSION", "rrf")
    os.environ.setdefault("RETRIEVAL_TOP_K", "8")
    os.environ.setdefault("RETRIEVAL_CANDIDATE_MULTIPLIER", "15")
    os.environ.setdefault("ENABLE_QUERY_BOOSTER", "true")

    from core.config.settings import get_settings
    from subagents.query_planner import expand_ambiguous_query

    get_settings.cache_clear()
    from core.retriever.knowledge import knowledge_status

    settings = get_settings()
    use_llm_boost = bool(getattr(settings, "query_booster_use_llm", True))
    if args.llm_boost:
        use_llm_boost = True
    if args.no_llm_boost:
        use_llm_boost = False
    status = knowledge_status()
    print("=== Thematic semantic paraphrase KB retrieval eval ===", flush=True)
    print(f"  backend={settings.vectorstore_backend} collection={status}", flush=True)
    print(
        f"  candidates={settings.retrieval_candidate_multiplier} "
        f"reranker={settings.enable_reranker} fusion={settings.hybrid_fusion}",
        flush=True,
    )
    print(f"  query_booster_llm={use_llm_boost}", flush=True)
    if not status.get("available") or int(status.get("count") or 0) < 10:
        print("ERROR: knowledge empty", file=sys.stderr)
        return 2

    chunks = _scroll_chunks(500)
    print(f"  scrolled chunks={len(chunks)}", flush=True)
    cases = _build_cases(chunks, n=100)
    print(f"  cases={len(cases)} (thematic same-meaning paraphrases)", flush=True)
    if len(cases) < 50:
        print("ERROR: too few grounded semantic queries", file=sys.stderr)
        return 2

    rows = []
    passed = 0
    boosted_count = 0
    by_theme: dict[str, list[bool]] = {}
    for i, case in enumerate(cases):
        if i:
            time.sleep(0.35)
        retrieval_query, boost_notes, boost_applied = expand_ambiguous_query(
            case.query,
            use_llm=use_llm_boost,
        )
        if boost_applied:
            boosted_count += 1
        hits = _retrieve_safe(retrieval_query, top_k=8)
        ok = _passes(hits, case)
        if ok:
            passed += 1
        by_theme.setdefault(case.theme, []).append(ok)
        rows.append(
            {
                "id": case.qid,
                "theme": case.theme,
                "query": case.query,
                "retrieval_query": retrieval_query,
                "query_boost_applied": boost_applied,
                "query_boost_notes": boost_notes,
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
        print(
            f"[{mark}] Q{case.qid:03d}/{len(cases)} ({case.theme}) {case.query[:70]}",
            flush=True,
        )
        if boost_applied:
            print(f"       boost: {boost_notes}", flush=True)
        if not ok and hits:
            print(
                f"       top: {(hits[0].text or '')[:110].replace(chr(10), ' ')}",
                flush=True,
            )

    total = len(cases)
    score = passed / total if total else 0.0
    threshold = 0.90  # paraphrases are harder than literal entity questions
    theme_scores = {
        theme: {
            "passed": sum(1 for x in results if x),
            "total": len(results),
            "score": (sum(1 for x in results if x) / len(results)) if results else 0.0,
        }
        for theme, results in sorted(by_theme.items())
    }
    report = {
        "backend": settings.vectorstore_backend,
        "collection": status,
        "embedding_model": settings.embedding_model_id,
        "passed": passed,
        "total": total,
        "score": score,
        "threshold": threshold,
        "ok": score >= threshold,
        "scoring": (
            "Thematic semantic paraphrases (same meaning, different wording); "
            "entity must exist in DB; top_k=8; optional 2-line query booster"
        ),
        "query_booster_llm": use_llm_boost,
        "boosted_case_count": boosted_count,
        "theme_scores": theme_scores,
        "cases": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResult: {passed}/{total} ({score:.0%}) target>={threshold:.0%}", flush=True)
    print("Theme breakdown:", flush=True)
    for theme, stats in theme_scores.items():
        print(
            f"  {theme}: {stats['passed']}/{stats['total']} ({stats['score']:.0%})",
            flush=True,
        )
    print(f"Report: {REPORT_PATH}", flush=True)
    print("WORKING" if report["ok"] else "NEEDS_ADJUSTMENT", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

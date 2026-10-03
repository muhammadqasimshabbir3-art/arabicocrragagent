#!/usr/bin/env python3
"""Run a fixed 20-query retrieval evaluation (document + knowledge).

Pass rule: at least one of top-3 hits contains any expected keyword (casefold).
Target: >= 18/20.

Usage:
    uv run python scripts/eval_retrieval_20.py
"""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_PDF = ROOT / "data" / "samples" / "arabic_admin_circular.pdf"
REPORT_PATH = ROOT / "data" / "samples" / "retrieval_eval_20_report.json"


@dataclass
class Case:
    qid: int
    source: str  # doc | kb
    query: str
    expect_any: tuple[str, ...]
    notes: str = ""


# 12 document + 8 knowledge cases. Keywords match noisy OCR / KB text.
CASES: list[Case] = [
    Case(1, "doc", "ما مدة الإجازة السنوية؟", ("واحد وعشرون", "جازة سنوية", "اجازة سنوية", "يومًا", "يوما")),
    Case(2, "doc", "كم يوم إجازة سنوية يستحق الموظف؟", ("واحد وعشرون", "جازة سنوية", "اجازة")),
    Case(3, "doc", "ما شروط الإجازة المرضية؟", ("مرضية", "تقرير طبي", "جنة طبية", "سبعة ايام")),
    Case(4, "doc", "ما الحد الأقصى للاستئذان القصير شهرياً؟", ("ثماني ساعات", "ساعتين", "ستئذان")),
    Case(5, "doc", "ما موضوع التعميم الإداري؟", ("ساعات عمل مرنة", "تعميم", "٧٤٤١", "7441")),
    Case(6, "doc", "ما هي ساعات العمل الأساسية؟", ("ثامنة", "سبع ساعات", "ساعات عمل")),
    Case(7, "doc", "على من يسري هذا التعميم؟", ("موظفين مدنيين", "نطاق", "دارات")),
    Case(8, "doc", "ما مدة حفظ المستندات الرسمية؟", ("خمس سنوات", "مستندات", "رشيف")),
    Case(9, "doc", "كم يوم عمل عن بعد مسموح أسبوعياً؟", ("عن بُعد", "عن بعد", "يومين", "سبوع")),
    Case(10, "doc", "ما واجب الموظف بشأن السرية؟", ("سرية", "افشاء", "اذن كتابي")),
    Case(11, "doc", "What is the annual leave duration?", ("واحد وعشرون", "جازة سنوية", "اجازة")),
    Case(12, "doc", "How many remote work days per week are allowed?", ("عن بُعد", "عن بعد", "يومين")),
    Case(13, "kb", "من هي شهرزاد؟", ("شهرزاد", "شهريار", "الليالي")),
    Case(14, "kb", "ما موضوع ألف ليلة وليلة؟", ("ليلة", "الليالي", "شهرزاد", "حكاي")),
    Case(15, "kb", "من هو الملك شهريار؟", ("شهريار", "شهرزاد", "الملك")),
    Case(16, "kb", "ما علاقة هزار افسانه بألف ليلة؟", ("هزار", "افسانه", "الليالي", "حكاي")),
    Case(17, "kb", "Who is Scheherazade in One Thousand and One Nights?", ("شهرزاد", "شهريار", "ليلة")),
    Case(18, "kb", "what is narrative structure of alf laila", ("ليلة", "الليالي", "حكاي", "شهرزاد")),
    Case(19, "kb", "ما هي حكايات الليالي؟", ("حكاي", "ليلة", "الليالي", "شهرزاد")),
    Case(20, "kb", "من مؤلف كتاب الليالي؟", ("مؤلف", "الليالي", "شهرزاد", "ليلة")),
]


def _hit_text(hit) -> str:
    return (hit.text or "").casefold()


def _passes(hits, expect_any: tuple[str, ...]) -> bool:
    blob = "\n".join(_hit_text(h) for h in hits)
    return any(tok.casefold() in blob for tok in expect_any)


def main() -> int:
    from core.retriever.knowledge import retrieve_from_knowledge
    from core.retriever.semantic import get_or_build_index, retrieve

    if not SAMPLE_PDF.is_file():
        print(f"Missing sample PDF: {SAMPLE_PDF}", file=sys.stderr)
        return 2

    payload = base64.b64encode(SAMPLE_PDF.read_bytes()).decode("ascii")
    index = get_or_build_index(
        payload,
        filename=SAMPLE_PDF.name,
        mime_type="application/pdf",
    )

    rows: list[dict] = []
    passed = 0
    for case in CASES:
        if case.source == "doc":
            hits = retrieve(index, case.query, top_k=3)
        else:
            hits = retrieve_from_knowledge(case.query, top_k=3)
        ok = _passes(hits, case.expect_any)
        if ok:
            passed += 1
        preview = [
            {
                "score": float(getattr(h, "score", 0) or 0),
                "page": getattr(h, "page_start", None),
                "text": (h.text or "")[:160].replace("\n", " "),
            }
            for h in hits
        ]
        rows.append(
            {
                "id": case.qid,
                "source": case.source,
                "query": case.query,
                "pass": ok,
                "expect_any": list(case.expect_any),
                "hits": preview,
            }
        )
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] Q{case.qid:02d} ({case.source}) {case.query}")
        if not ok:
            top = preview[0]["text"] if preview else "(no hits)"
            print(f"       top: {top}")

    total = len(CASES)
    score = passed / total if total else 0.0
    report = {
        "passed": passed,
        "total": total,
        "score": score,
        "threshold": 0.9,
        "ok": passed >= 18,
        "cases": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResult: {passed}/{total} ({score:.0%}) — report: {REPORT_PATH}")
    print("WORKING" if report["ok"] else "NEEDS_ADJUSTMENT")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

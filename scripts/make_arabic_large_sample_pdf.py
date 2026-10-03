#!/usr/bin/env python3
"""Build a larger Arabic sample PDF (~500–900 extractable lines) for document RAG.

Uses the same PyMuPDF HTML text-box approach as the small admin circular.
Content is unique (no near-duplicate cycles) so digital_text dedupe does not
collapse the document below the target size. A sibling .txt keeps clean Unicode.
"""

from __future__ import annotations

from pathlib import Path

TOPICS = [
    ("الحضور والانصراف", "تسجيل الحضور عبر النظام المعتمد وعدم توكيل الغير"),
    ("ساعات العمل المرنة", "نافذة مرنة بموافقة المدير المباشر ضمن حدود اللائحة"),
    ("العمل عن بُعد", "يومان أسبوعيًا كحد أقصى مع أدوات اتصال آمنة"),
    ("الإجازة السنوية", "واحد وعشرون يومًا بعد سنة خدمة كاملة مع جواز التجزئة"),
    ("الإجازة المرضية", "تقرير طبي معتمد واشتراط اللجنة الطبية بعد سبعة أيام"),
    ("الاستئذان القصير", "ساعتان يوميًا وبحد أقصى ثماني ساعات شهريًا"),
    ("الانتداب", "بدل انتداب للمهام خارج المقر مع تقرير خلال خمسة أيام"),
    ("التدريب", "الدورات المعتمدة جزء من ساعات العمل دون خصم من الإجازة"),
    ("السرية", "عدم إفشاء المعلومات إلا بإذن كتابي من الجهة المختصة"),
    ("حفظ المستندات", "حفظ مالي عشر سنوات وبشري خمس سنوات بعد انتهاء الخدمة"),
    ("الأرشفة الإلكترونية", "رفع المستند خلال سبعة أيام ومنع الإتلاف المبكر"),
    ("أمن المعلومات", "كلمات مرور قوية ومصادقة ثنائية للحسابات الحساسة"),
    ("السلامة المهنية", "إخلاء طارئ مرتان سنويًا والإبلاغ عن الحوادث خلال يوم"),
    ("تضارب المصالح", "حظر الهدايا والمنافع المرتبطة بالوظيفة"),
    ("التظلم", "حق التظلم خلال خمسة عشر يومًا أمام لجنة التظلمات"),
    ("الترقيات", "ترقية وفق تقييم الأداء والشواغر المعتمدة"),
    ("العلاوات", "علاوة سنوية مرتبطة بتقييم الأداء وعدم وجود جزاءات قائمة"),
    ("إنهاء الخدمة", "إجراءات الاستقالة والفصل وفق النظام مع تسليم العهد"),
    ("الزي الرسمي", "الالتزام بالزي المعتمد في الوحدات التي تتطلبه"),
    ("استخدام المركبات", "مركبات الوزارة للمهام الرسمية فقط مع سجل الرحلات"),
]


def _build_lines(*, target_lines: int = 700) -> list[str]:
    lines: list[str] = [
        "دليل السياسات والإجراءات الإدارية — وزارة الشؤون الإدارية",
        "تعميم إداري رقم ١٤٤٧/٠٧ — النسخة الموسعة للاختبار",
        "صدر في الرياض بتاريخ ١٥ محرم ١٤٤٧ هـ",
        "",
    ]
    article = 1
    while len([ln for ln in lines if ln.strip()]) < target_lines:
        for topic, gist in TOPICS:
            n = f"{article:03d}"
            lines.extend(
                [
                    f"المادة {n} — {topic}",
                    f"تهدف هذه المادة إلى تنظيم {topic} بما يحقق جودة الخدمة والعدالة بين الموظفين.",
                    f"القاعدة الأساسية: {gist}.",
                    f"يطبق الحكم على جميع الموظفين المدنيين ما لم يُستثنَ بقرار مكتوب من المدير العام بشأن المادة {n}.",
                    f"يتولى المدير المباشر متابعة الالتزام بالمادة {n} ورفع الملاحظات إلى الموارد البشرية شهريًا.",
                    f"عند المخالفة للمادة {n} تُتبع إجراءات التحقيق وفق الفصل الخاص بالسلوك الوظيفي.",
                    f"تُراجع صياغة المادة {n} كل سنتين أو عند صدور نظام أعلى يتعارض معها.",
                    f"للاستفسار عن المادة {n} يُراسل قسم السياسات على البريد الداخلي policies@example.gov.sa.",
                    "",
                ]
            )
            article += 1
            if len([ln for ln in lines if ln.strip()]) >= target_lines:
                break
        if article > 500:
            break
    lines.extend(
        [
            "أحكام ختامية",
            "يعمل بهذا الدليل من تاريخ صدوره ويُلغى كل ما يتعارض معه من تعليمات سابقة.",
            "تختص الإدارة القانونية بتفسير الأحكام عند الاختلاف.",
        ]
    )
    return lines


def write_arabic_large_sample_pdf(
    path: Path,
    *,
    target_lines: int = 700,
) -> Path:
    """Create a multi-page Arabic PDF with extractable text (HTML text box)."""
    import fitz

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _build_lines(target_lines=target_lines)

    doc = fitz.open()
    per_page = 26
    for start in range(0, len(lines), per_page):
        chunk = lines[start : start + per_page]
        page = doc.new_page(width=595, height=842)
        body = "\n".join(chunk)
        safe = (
            body.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        title = "دليل السياسات الإدارية" if start == 0 else "دليل السياسات — تابع"
        html = f"""
        <div dir="rtl" style="font-family: sans-serif; font-size: 11pt; line-height: 1.55;">
          <h2 style="font-size: 14pt; color: #3d2a1a;">{title}</h2>
          <p style="white-space: pre-wrap;">{safe}</p>
        </div>
        """
        page.insert_htmlbox(fitz.Rect(40, 40, 555, 800), html)

    # Sibling .txt keeps clean Unicode for inspection / text-ingest tests
    path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lines",
        type=int,
        default=700,
        help="Target non-empty lines in the authored text (default 700)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "samples"
        / "arabic_policy_handbook_large.pdf",
    )
    args = parser.parse_args()
    out = write_arabic_large_sample_pdf(args.out, target_lines=args.lines)
    import fitz

    d = fitz.open(out)
    text = "\n".join(p.get_text() for p in d)
    page_count = d.page_count
    d.close()
    nlines = len([ln for ln in text.splitlines() if ln.strip()])
    print(f"Wrote {out}")
    print(f"also wrote {out.with_suffix('.txt')}")
    print(f"pages={page_count} non_empty_lines≈{nlines} chars={len(text)}")

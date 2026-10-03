"""Streamlit UI for وثيقة — Arabic Document Intelligence."""

from __future__ import annotations

import asyncio
import base64

import streamlit as st
from langchain_core.messages import HumanMessage

from agent.graph import GRAPH_RUN_CONFIG, graph

st.set_page_config(
    page_title="وثيقة | Wathiqa",
    page_icon="📜",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@700&family=Cairo:wght@400;600&display=swap');
      html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
      h1 { font-family: 'Amiri', serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("وثيقة")
st.caption("وكيل المستندات العربية — استخراج نص رقمي · فهرسة · إجابة موثّقة بالمصادر")

col1, col2 = st.columns([1.4, 1])
with col1:
    uploaded = st.file_uploader(
        "المستند / Document",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    )
    question = st.text_area(
        "السؤال / Question",
        value="ما موضوع هذا المستند؟",
        height=120,
    )
    summarize = st.checkbox("تلخيص فقط / Summarize only", value=False)
    run = st.button("اسأل وثيقة / Ask", type="primary")

with col2:
    st.info(
        "١ ارفع المستند\n\n"
        "٢ اكتب سؤالك بالعربية أو الإنجليزية\n\n"
        "٣ اقرأ الإجابة مع المصادر ورقم الصفحة"
    )

if run:
    payload: dict = {
        "messages": [HumanMessage(content=question)],
        "user_input": question,
    }
    if uploaded is not None:
        raw = uploaded.getvalue()
        b64 = base64.b64encode(raw).decode("ascii")
        payload.update(
            {
                "pdf_data_base64": b64,
                "pdf_filename": uploaded.name,
                "pdf_summarize_only": summarize,
                "document_data_base64": b64,
                "document_filename": uploaded.name,
                "document_mime_type": uploaded.type or "",
                "summarize_only": summarize,
            }
        )
    with st.spinner("جاري المعالجة…"):
        # Graph nodes are async — must use ainvoke (not sync invoke).
        result = asyncio.run(graph.ainvoke(payload, config=GRAPH_RUN_CONFIG))
    st.success(result.get("task_plan_summary") or "Done")
    for message in result.get("messages") or []:
        role = getattr(message, "type", "ai")
        content = getattr(message, "content", str(message))
        with st.chat_message("user" if role == "human" else "assistant"):
            # Prefer plain text — avoid raw HTML/markdown injection via st.markdown.
            st.text(str(content) if content is not None else "")

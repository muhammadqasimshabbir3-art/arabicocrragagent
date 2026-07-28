interface StackInfoPanelProps {
  uiLang: "ar" | "en";
}

function envLabel(key: string, fallback: string): string {
  const raw = import.meta.env[key];
  return raw != null && String(raw).trim() !== "" ? String(raw).trim() : fallback;
}

export function StackInfoPanel({ uiLang }: StackInfoPanelProps) {
  const isAr = uiLang === "ar";
  const vectorDb = envLabel("VITE_STACK_VECTOR_DB", "Chroma · data/chroma");
  const embeddings = envLabel("VITE_STACK_EMBEDDINGS", "BAAI/bge-m3");
  const ocr = envLabel("VITE_STACK_OCR", "Qari-OCR v0.3");

  return (
    <aside className="panel stack-panel">
      <div className="panel-title">{isAr ? "البنية التقنية" : "Stack"}</div>
      <ul className="stack-list">
        <li>
          <span>{isAr ? "قاعدة المتجهات" : "Vector DB"}</span>
          <strong>{vectorDb}</strong>
        </li>
        <li>
          <span>{isAr ? "التضمين" : "Embeddings"}</span>
          <strong>{embeddings}</strong>
        </li>
        <li>
          <span>OCR</span>
          <strong>{ocr}</strong>
        </li>
        <li>
          <span>{isAr ? "أدوات" : "Tools"}</span>
          <strong>{isAr ? "قاعدة معرفة · بحث ويب" : "Knowledge DB · Web search"}</strong>
        </li>
      </ul>
      <p className="stack-note">
        {isAr
          ? "للإنتاج: عيّن VECTORSTORE_BACKEND=qdrant و QDRANT_URL في .env"
          : "Production: set VECTORSTORE_BACKEND=qdrant and QDRANT_URL in .env"}
      </p>
    </aside>
  );
}

import { FileUp, Play, RotateCcw, Square, X } from "lucide-react";
import type { DragEvent, KeyboardEvent } from "react";
import { useState } from "react";
import { IS_PRODUCTION, LANGGRAPH_API_URL, USES_DEV_PROXY } from "../config";
import type { AgentRunSettings, StepState } from "../types";

interface AgentConfigFormProps {
  settings: AgentRunSettings;
  onChange: <K extends keyof AgentRunSettings>(key: K, value: AgentRunSettings[K]) => void;
  running: boolean;
  steps?: StepState[];
  serverOnline: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onClearAnswer: () => void;
  /** Auto-run summarize after upload — no user summary prompt needed. */
  onAutoSummarize: (overrides: Partial<AgentRunSettings>) => void;
  canReset?: boolean;
  answer: string;
  error: string | null;
  uiLang: "ar" | "en";
}

const ACCEPTED = [
  "application/pdf",
  ".pdf",
  "image/jpeg",
  ".jpg",
  ".jpeg",
  "image/png",
  ".png",
  "image/tiff",
  ".tif",
  ".tiff",
  "image/bmp",
  ".bmp",
  "image/webp",
  ".webp",
  "image/gif",
  ".gif",
  "text/plain",
  ".txt",
  "text/markdown",
  ".md",
  ".markdown",
  "text/csv",
  ".csv",
  ".tsv",
  ".log",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".docx",
].join(",");

const MAX_FILE_BYTES = 50 * 1024 * 1024;

export function AgentConfigForm({
  settings,
  onChange,
  running,
  steps = [],
  serverOnline,
  onStart,
  onStop,
  onReset,
  onClearAnswer,
  onAutoSummarize,
  canReset = true,
  answer,
  error,
  uiLang,
}: AgentConfigFormProps) {
  const isAr = uiLang === "ar";
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const filename = settings.pdf_filename || settings.document_filename || "";
  const responseLang = settings.response_language === "en" ? "en" : "ar";
  const searchSource = settings.search_source ?? "auto";
  const activeStep = steps.find((s) => s.status === "running");
  const pdfMode = Boolean(filename);
  // Once the answer is on screen, leave "busy" UI even if the run flag lags.
  const busy = running && !answer;
  const isDocBusy =
    busy &&
    pdfMode &&
    (Boolean(settings.summarize_only || settings.pdf_summarize_only) ||
      activeStep?.id === "summarize_document" ||
      activeStep?.id === "prepare_input" ||
      activeStep?.id === "decision_agent");
  const isGenerating =
    busy &&
    (activeStep?.id === "query_documents" ||
      activeStep?.id === "query_knowledge_base" ||
      activeStep?.id === "query_planner" ||
      activeStep?.id === "web_search" ||
      activeStep?.id === "summarize_document" ||
      activeStep?.id === "call_model");

  const canType = serverOnline;
  const canSend = Boolean(serverOnline && settings.user_input.trim()) && !busy;

  const copy = {
    uploadBig: isAr ? "① ارفع الملف هنا" : "① Drop your file here",
    uploadClick: isAr ? "أو انقر للاختيار" : "or click to choose",
    uploadHint: isAr
      ? "PDF · صور · TXT · DOCX — يُلخَّص تلقائيًا بعد الرفع (حد أقصى 50MB)"
      : "Text-layer PDF · TXT · DOCX — digital text extraction (max 50MB)",
    fileTooLarge: isAr
      ? "الملف أكبر من 50 ميجابايت. اختر ملفًا أصغر."
      : "File exceeds 50MB. Choose a smaller file.",
    askBig: isAr ? "② اكتب سؤالك" : "② Type your question",
    askHintPdf: isAr
      ? "اسأل عن الملف المرفوع بعد اكتمال الفهرسة"
      : "Ask about the uploaded file after indexing finishes",
    askHintKb: isAr
      ? "بدون ملف: الوضع التلقائي يختار قاعدة المعرفة أو بحث الإنترنت أو المحادثة"
      : "No file: Auto picks database, internet search, or chat",
    askHintWeb: isAr
      ? "بحث الإنترنت مفعّل: يبحث على الويب (مثل بحث جوجل) لهذا السؤال"
      : "Internet search on: live web lookup (Google-style) for this question",
    askHintDb: isAr
      ? "قاعدة البيانات مفعّلة: بحث في المعرفة المفهرسة فقط"
      : "Database on: search the seeded knowledge base only",
    answerLang: isAr ? "لغة الإجابة" : "Answer language",
    searchSource: isAr ? "مصدر البحث" : "Search source",
    sourceAuto: isAr ? "تلقائي" : "Auto",
    sourceDb: isAr ? "قاعدة البيانات" : "Database",
    sourceWeb: isAr ? "بحث الإنترنت" : "Search Internet",
    placeholder: isAr
      ? pdfMode
        ? "اكتب سؤالك عن المستند…"
        : "اكتب سؤالك هنا…"
      : pdfMode
        ? "Ask a question about the document…"
        : "Type your question here…",
    modePdf: isAr ? "وضع المستند" : "Document mode",
    modeKb: isAr ? "وضع قاعدة المعرفة" : "Knowledge mode",
    modeWeb: isAr ? "بحث الإنترنت" : "Internet search",
    modeDb: isAr ? "وضع قاعدة البيانات" : "Database mode",
    send: isAr ? "اسأل الآن" : "Ask now",
    sendWait: isAr ? "جاري العمل…" : "Working…",
    stop: isAr ? "إيقاف" : "Stop",
    stopHint: isAr
      ? "قد يستمر الخادم في المعالجة بعد الإيقاف المحلي."
      : "Server may continue processing after local cancel.",
    reset: isAr ? "مسح" : "Clear",
    clear: isAr ? "إزالة" : "Remove",
    ready: isAr ? "جاري المعالجة" : "Processing",
    readyIdle: isAr ? "تم الرفع — اسأل هنا" : "Uploaded — ask here",
    answer: isAr ? "③ الإجابة" : "③ Answer",
    answerEmpty: isAr
      ? "الإجابة تظهر هنا بجانب السؤال"
      : "Answer appears here beside the question",
    error: isAr ? "حدث خطأ" : "Something went wrong",
    offline: isAr
      ? "الخادم غير متصل. شغّل ./start.sh both"
      : "Backend offline. Run ./start.sh both",
    indexing: isAr ? "جاري الفهرسة…" : "Indexing…",
    generating: isAr ? "جاري إنشاء الإجابة…" : "Generating answer…",
    working: isAr ? "جاري العمل…" : "Working…",
  };

  const readDocumentFile = (file: File) => {
    if (busy || !serverOnline) return;
    if (file.size > MAX_FILE_BYTES) {
      setUploadError(copy.fileTooLarge);
      return;
    }
    setUploadError(null);
    const reader = new FileReader();
    reader.onload = () => {
      const data = typeof reader.result === "string" ? reader.result : "";
      const base64 = data.includes(",") ? data.split(",")[1] : data;
      onAutoSummarize({
        pdf_data_base64: base64,
        pdf_filename: file.name,
        document_data_base64: base64,
        document_filename: file.name,
        document_mime_type: file.type || "",
        pdf_analysis_enabled: true,
        pdf_summarize_only: true,
        summarize_only: true,
      });
    };
    reader.readAsDataURL(file);
  };

  const clearDocument = () => {
    setUploadError(null);
    onChange("pdf_data_base64", "");
    onChange("pdf_filename", "");
    onChange("document_data_base64", "");
    onChange("document_filename", "");
    onChange("document_mime_type", "");
    onChange("pdf_analysis_enabled", false);
    onChange("pdf_summarize_only", false);
    onChange("summarize_only", false);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onStart();
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!busy && serverOnline) setDragOver(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (busy || !serverOnline) return;
    const file = e.dataTransfer.files?.[0];
    if (file) readDocumentFile(file);
  };

  return (
    <section className="action-desk">
      <div className="action-grid with-answer">
        <div className={`action-card upload-card${filename ? " has-file" : ""}${isDocBusy ? " is-busy" : ""}`}>
          <div
            className={`upload-zone giant${filename ? " has-file" : ""}${dragOver ? " drag-over" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <FileUp className="upload-icon" size={42} strokeWidth={1.75} />
            <strong className="upload-big-label">{copy.uploadBig}</strong>
            <span className="upload-click">{copy.uploadClick}</span>
            <small>{copy.uploadHint}</small>
            <input
              type="file"
              accept={ACCEPTED}
              disabled={busy || !serverOnline}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) readDocumentFile(file);
                e.target.value = "";
              }}
              aria-label={copy.uploadBig}
            />
          </div>
          {uploadError && <p className="hint warn">{uploadError}</p>}
          {filename && (
            <div className="file-chip">
              <div>
                <strong>{filename}</strong>
                <small>
                  {busy
                    ? copy.ready
                    : answer
                      ? isAr
                        ? "مكتمل — اسأل هنا"
                        : "Done — ask here"
                      : copy.readyIdle}
                </small>
              </div>
              <button
                type="button"
                className="btn small ghost"
                disabled={busy}
                onClick={clearDocument}
              >
                <X size={14} />
                {copy.clear}
              </button>
            </div>
          )}
        </div>

        <div className="action-card ask-card">
          <div className="composer-toggles">
            <div className="answer-lang-row">
              <span className="answer-lang-label">{copy.answerLang}</span>
              <div className="answer-lang-toggle" role="group" aria-label={copy.answerLang}>
                <button
                  type="button"
                  className={responseLang === "ar" ? "active" : ""}
                  disabled={busy}
                  onClick={() => onChange("response_language", "ar")}
                >
                  العربية
                </button>
                <button
                  type="button"
                  className={responseLang === "en" ? "active" : ""}
                  disabled={busy}
                  onClick={() => onChange("response_language", "en")}
                >
                  English
                </button>
              </div>
            </div>
            <div className="answer-lang-row">
              <span className="answer-lang-label">{copy.searchSource}</span>
              <div className="answer-lang-toggle" role="group" aria-label={copy.searchSource}>
                <button
                  type="button"
                  className={searchSource === "auto" ? "active" : ""}
                  disabled={busy}
                  onClick={() => onChange("search_source", "auto")}
                >
                  {copy.sourceAuto}
                </button>
                <button
                  type="button"
                  className={searchSource === "database" ? "active" : ""}
                  disabled={busy}
                  onClick={() => onChange("search_source", "database")}
                  title={
                    isAr
                      ? "ابحث في قاعدة المعرفة المفهرسة فقط"
                      : "Search the seeded knowledge base only"
                  }
                >
                  {copy.sourceDb}
                </button>
                <button
                  type="button"
                  className={searchSource === "web" ? "active" : ""}
                  disabled={busy}
                  onClick={() => onChange("search_source", "web")}
                  title={
                    isAr
                      ? "بحث الإنترنت / بحث جوجل لهذا السؤال"
                      : "Search the internet (Google-style) for this question"
                  }
                >
                  {copy.sourceWeb}
                </button>
              </div>
            </div>
          </div>
          <label className="ask-label" htmlFor="wathiqa-question">
            {copy.askBig}
          </label>
          <div
            className={`mode-pill ${
              searchSource === "web"
                ? "web"
                : searchSource === "database"
                  ? "db"
                  : pdfMode
                    ? "pdf"
                    : "kb"
            }`}
          >
            {searchSource === "web"
              ? copy.modeWeb
              : searchSource === "database"
                ? copy.modeDb
                : pdfMode
                  ? copy.modePdf
                  : copy.modeKb}
          </div>
          <p className="ask-mode-hint">
            {searchSource === "web"
              ? copy.askHintWeb
              : searchSource === "database"
                ? copy.askHintDb
                : pdfMode
                  ? copy.askHintPdf
                  : copy.askHintKb}
          </p>
          <textarea
            id="wathiqa-question"
            className="composer-input giant"
            value={settings.user_input}
            onChange={(e) => {
              onChange("user_input", e.target.value);
              if (!busy) {
                if (answer) onClearAnswer();
                onChange("pdf_summarize_only", false);
                onChange("summarize_only", false);
              }
            }}
            onKeyDown={onKeyDown}
            placeholder={copy.placeholder}
            disabled={!canType}
            rows={5}
            dir="auto"
          />
          <div className="composer-actions sticky-actions">
            {!busy ? (
              <button
                type="button"
                className="btn primary giant-btn"
                disabled={!canSend}
                onClick={onStart}
              >
                <Play size={20} />
                {copy.send}
              </button>
            ) : (
              <>
                <button type="button" className="btn primary giant-btn" disabled>
                  <Play size={20} />
                  {copy.sendWait}
                </button>
                <button type="button" className="btn danger giant-btn" onClick={onStop}>
                  <Square size={18} />
                  {copy.stop}
                </button>
              </>
            )}
            <button
              type="button"
              className="btn ghost"
              onClick={onReset}
              disabled={busy || !canReset}
            >
              <RotateCcw size={16} />
              {copy.reset}
            </button>
          </div>
          {busy && <p className="hint muted cancel-note">{copy.stopHint}</p>}
        </div>

        <div
          className={`action-card answer-card${error ? " has-error" : ""}${answer ? " has-answer" : ""}`}
        >
          <strong className="answer-label">{error ? copy.error : copy.answer}</strong>
          {busy && !answer && !error && (
            <div
              className={`inline-progress compact${isDocBusy ? " scanning" : ""}${isGenerating ? " generating" : ""}`}
              role="status"
              aria-live="polite"
            >
              <div className="inline-progress-label">
                <strong>
                  {isDocBusy ? copy.indexing : isGenerating ? copy.generating : copy.working}
                </strong>
              </div>
              <div className="inline-progress-track">
                <div className="inline-progress-fill" />
              </div>
            </div>
          )}
          {error ? (
            <p className="answer-body error-text">{error}</p>
          ) : answer ? (
            <div className="answer-body" dir="auto">
              {answer}
            </div>
          ) : (
            !busy && <p className="answer-placeholder">{copy.answerEmpty}</p>
          )}
        </div>
      </div>

      {!serverOnline && (
        <p className="hint warn">
          {IS_PRODUCTION || !USES_DEV_PROXY ? (
            <>
              {isAr ? "الخادم غير متاح على" : "Backend unreachable at"}{" "}
              <code>{LANGGRAPH_API_URL}</code>
            </>
          ) : (
            copy.offline
          )}
        </p>
      )}
    </section>
  );
}

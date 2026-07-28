import { FileUp, Play, RotateCcw, Square, X } from "lucide-react";
import type { KeyboardEvent } from "react";
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
  const filename = settings.pdf_filename || settings.document_filename || "";
  const responseLang = settings.response_language === "en" ? "en" : "ar";
  const activeStep = steps.find((s) => s.status === "running");
  const pdfMode = Boolean(filename);
  // Once the answer is on screen, leave "busy" UI even if the run flag lags.
  const busy = running && !answer;
  const isDocBusy =
    busy &&
    pdfMode &&
    (Boolean(settings.summarize_only || settings.pdf_summarize_only) ||
      activeStep?.id === "ingest_document" ||
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
      ? "PDF · صور · TXT · DOCX — يُلخَّص تلقائيًا بعد الرفع"
      : "PDF · images · TXT · DOCX — auto-summarizes on upload",
    askBig: isAr ? "② اكتب سؤالك" : "② Type your question",
    askHintPdf: isAr
      ? "اسأل عن الملف المرفوع بعد اكتمال الفهرسة"
      : "Ask about the uploaded file after indexing finishes",
    askHintKb: isAr
      ? "بدون ملف: يبحث في قاعدة المعرفة (BM25 + متجهات)"
      : "No file: searches the knowledge DB (BM25 + vectors)",
    answerLang: isAr ? "لغة الإجابة" : "Answer language",
    placeholder: isAr
      ? pdfMode
        ? "اكتب سؤالك عن المستند…"
        : "اكتب سؤالك هنا…"
      : pdfMode
        ? "Ask a question about the document…"
        : "Type your question here…",
    modePdf: isAr ? "وضع المستند" : "Document mode",
    modeKb: isAr ? "وضع قاعدة المعرفة" : "Knowledge mode",
    send: isAr ? "اسأل الآن" : "Ask now",
    sendWait: isAr ? "جاري العمل…" : "Working…",
    stop: isAr ? "إيقاف" : "Stop",
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

  return (
    <section className="action-desk">
      <div className="action-grid with-answer">
        <div className={`action-card upload-card${filename ? " has-file" : ""}${isDocBusy ? " is-busy" : ""}`}>
          <div className={`upload-zone giant${filename ? " has-file" : ""}`}>
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
          <label className="ask-label" htmlFor="wathiqa-question">
            {copy.askBig}
          </label>
          <div className={`mode-pill ${pdfMode ? "pdf" : "kb"}`}>
            {pdfMode ? copy.modePdf : copy.modeKb}
          </div>
          <p className="ask-mode-hint">{pdfMode ? copy.askHintPdf : copy.askHintKb}</p>
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

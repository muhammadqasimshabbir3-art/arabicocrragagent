import type { CSSProperties } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Circle,
  CircleDashed,
  Loader2,
  MinusCircle,
} from "lucide-react";
import { progressPercent } from "../lib/streamProgress";
import {
  WORKFLOW_STEPS,
  localizeStepDetail,
  localizeTaskPlan,
} from "../lib/workflowSteps";
import type { StepState, StepStatus } from "../types";

interface WorkflowPipelineProps {
  steps: StepState[];
  running: boolean;
  reconnected?: boolean;
  taskPlanSummary?: string;
  agentRoute?: string;
  uiLang: "ar" | "en";
}

const LABELS: Record<string, { ar: string; en: string; arDesc: string; enDesc: string }> = {
  prepare_input: {
    ar: "تهيئة الإدخال",
    en: "Prepare input",
    arDesc: "تنسيق السؤال وسجل المحادثة",
    enDesc: "Format the question and chat history",
  },
  decision_agent: {
    ar: "توجيه القرار",
    en: "Route decision",
    arDesc: "اختيار مسار: فهرسة أو سؤال أو تلخيص",
    enDesc: "Choose ingest, query, summarize, or chat",
  },
  ingest_document: {
    ar: "فهرسة المستند",
    en: "Ingest document",
    arDesc: "تحميل → OCR → تقطيع → تضمين → فهرسة",
    enDesc: "Load → OCR → chunk → embed → index",
  },
  query_documents: {
    ar: "إجابة موثّقة",
    en: "Grounded QA",
    arDesc: "استرجاع → طبقة ثقة → إجابة مع مصادر",
    enDesc: "Retrieve → trust layer → answer with sources",
  },
  query_planner: {
    ar: "مخطط الاستعلام",
    en: "Query planner",
    arDesc: "تحويل السؤال إلى استعلام عربي ثم قراءة فقط من قاعدة المعرفة",
    enDesc: "Convert question to Arabic retrieval query, then READ-only KB search",
  },
  query_knowledge_base: {
    ar: "قاعدة المعرفة",
    en: "Knowledge DB",
    arDesc: "تضمين عربي + BM25 + متجهات ثم إجابة موثّقة",
    enDesc: "Arabic embedding + BM25 + vectors then grounded answer",
  },
  summarize_document: {
    ar: "تلخيص",
    en: "Summarize",
    arDesc: "فهرسة ثم ملخص موجز للمستند",
    enDesc: "Index then summarize the document",
  },
  call_model: {
    ar: "محادثة عامة",
    en: "General chat",
    arDesc: "رد عام دون اختلاق حقائق من المستند",
    enDesc: "General help without inventing document facts",
  },
};

function StatusIcon({ status }: { status: StepStatus }) {
  switch (status) {
    case "running":
      return <Loader2 size={18} className="spin step-icon running" />;
    case "completed":
      return <CheckCircle2 size={18} className="step-icon completed" />;
    case "skipped":
      return <MinusCircle size={18} className="step-icon skipped" />;
    case "error":
      return <AlertCircle size={18} className="step-icon error" />;
    default:
      return <Circle size={16} className="step-icon pending" />;
  }
}

export function WorkflowPipeline({
  steps,
  running,
  reconnected,
  taskPlanSummary,
  agentRoute,
  uiLang,
}: WorkflowPipelineProps) {
  const isAr = uiLang === "ar";
  const progress = progressPercent(steps);
  const activeStep = steps.find((s) => s.status === "running");
  const planText = localizeTaskPlan(taskPlanSummary, agentRoute, uiLang);

  return (
    <section className="pipeline-panel">
      <div className="pipeline-header">
        <div>
          <div className="panel-title">
            <span>{isAr ? "مسار المعالجة" : "Processing pipeline"}</span>
            {running && <CircleDashed size={16} className="spin" />}
          </div>
          <p className="panel-desc">
            {isAr
              ? "خطوات حية من LangGraph — تضيء أثناء التنفيذ."
              : "Live LangGraph steps — light up as they run."}
          </p>
          {activeStep && running && (
            <p className="active-step-hint">
              {isAr ? "الآن:" : "Now:"}{" "}
              <strong>
                {(LABELS[activeStep.id] ?? { ar: activeStep.id, en: activeStep.id })[
                  isAr ? "ar" : "en"
                ]}
              </strong>
            </p>
          )}
        </div>
        <div className="progress-ring" style={{ "--progress": `${progress}%` } as CSSProperties}>
          <span>{progress}%</span>
        </div>
      </div>

      {reconnected && running && (
        <div className="task-plan reconnect-banner">
          {isAr
            ? "أُعيد الاتصال — الوكيل ما زال يعمل على الخادم"
            : "Reconnected — agent kept running on the server"}
        </div>
      )}

      {planText && <div className="task-plan">{planText}</div>}

      <ol className="pipeline-steps">
        {WORKFLOW_STEPS.map((def) => {
          const state = steps.find((s) => s.id === def.id);
          const status = state?.status ?? "pending";
          const labels = LABELS[def.id];
          const detail = localizeStepDetail(
            def.id,
            state?.detail,
            uiLang,
            agentRoute,
          );
          return (
            <li key={def.id} className={`pipeline-step ${status}`}>
              <div className="step-marker">
                <StatusIcon status={status} />
                <div className="step-line" />
              </div>
              <div className="step-body">
                <div className="step-title-row">
                  {def.emoji && (
                    <span className="step-emoji" aria-hidden>
                      {def.emoji}
                    </span>
                  )}
                  <strong>{labels ? labels[isAr ? "ar" : "en"] : def.label}</strong>
                  {def.optional && (
                    <span className="optional-tag">{isAr ? "اختياري" : "optional"}</span>
                  )}
                  <span className={`step-badge ${status}`}>
                    {status === "running"
                      ? isAr
                        ? "يعمل"
                        : "running"
                      : status === "completed"
                        ? isAr
                          ? "تم"
                          : "done"
                        : status === "skipped"
                          ? isAr
                            ? "تخطى"
                            : "skipped"
                          : status === "error"
                            ? isAr
                              ? "خطأ"
                              : "error"
                            : isAr
                              ? "انتظار"
                              : "pending"}
                  </span>
                </div>
                <p>{labels ? labels[isAr ? "arDesc" : "enDesc"] : def.description}</p>
                {detail && <p className="step-detail">{detail}</p>}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

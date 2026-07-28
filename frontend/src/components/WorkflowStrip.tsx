import type { CSSProperties } from "react";
import { progressPercent } from "../lib/streamProgress";
import { WORKFLOW_STEPS } from "../lib/workflowSteps";
import type { StepState, StepStatus } from "../types";

const SHORT: Record<string, { ar: string; en: string }> = {
  prepare_input: { ar: "إدخال", en: "Input" },
  decision_agent: { ar: "توجيه", en: "Route" },
  ingest_document: { ar: "فهرسة", en: "Index" },
  query_documents: { ar: "إجابة", en: "Answer" },
  query_planner: { ar: "خطة", en: "Plan" },
  query_knowledge_base: { ar: "قاعدة", en: "KB" },
  web_search: { ar: "ويب", en: "Web" },
  summarize_document: { ar: "تلخيص", en: "Summary" },
  call_model: { ar: "محادثة", en: "Chat" },
};

interface WorkflowStripProps {
  steps: StepState[];
  running: boolean;
  completed?: boolean;
  uiLang: "ar" | "en";
}

function phaseLabel(
  activeId: string | undefined,
  isAr: boolean,
  running: boolean,
  completed: boolean,
): string {
  if (completed && !running) return isAr ? "اكتمل" : "Completed";
  if (!running) return isAr ? "مسار المعالجة" : "Workflow";
  if (activeId === "ingest_document") return isAr ? "مسح وفهرسة…" : "Scanning…";
  if (activeId === "query_planner") return isAr ? "كتابة الاستعلام…" : "Writing query…";
  if (activeId === "web_search") return isAr ? "بحث الويب…" : "Web search…";
  if (
    activeId === "query_documents" ||
    activeId === "query_knowledge_base" ||
    activeId === "summarize_document" ||
    activeId === "call_model"
  ) {
    return isAr ? "إنشاء الإجابة…" : "Generating…";
  }
  return isAr ? "جاري العمل…" : "Working…";
}

export function WorkflowStrip({
  steps,
  running,
  completed = false,
  uiLang,
}: WorkflowStripProps) {
  const isAr = uiLang === "ar";
  const progress = completed && !running ? 100 : progressPercent(steps);
  const active = steps.find((s) => s.status === "running");
  const title = phaseLabel(active?.id, isAr, running, completed);

  return (
    <section
      className={`workflow-strip${running ? " is-running" : ""}${completed && !running ? " is-complete" : ""}`}
      aria-label={isAr ? "ملخص مسار المعالجة" : "Workflow progress summary"}
    >
      <div className="strip-head">
        <span className="strip-title">{title}</span>
        <div
          className={`strip-bar${running ? " striped" : ""}`}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="strip-bar-fill"
            style={{ width: `${Math.max(progress, running ? 8 : 0)}%` } as CSSProperties}
          />
        </div>
        <span className="strip-pct">{progress}%</span>
      </div>
      <StripPills steps={steps} activeId={running ? active?.id : undefined} isAr={isAr} />
    </section>
  );
}

function StripPills({
  steps,
  activeId,
  isAr,
}: {
  steps: StepState[];
  activeId?: string;
  isAr: boolean;
}) {
  return (
    <div className="strip-pills">
      {WORKFLOW_STEPS.map((def, index) => {
        const state = steps.find((s) => s.id === def.id);
        const status = (state?.status ?? "pending") as StepStatus;
        const label = SHORT[def.id] ?? { ar: def.label, en: def.label };
        return (
          <div
            key={def.id}
            className={`strip-pill ${status}${activeId === def.id ? " active" : ""}`}
            title={label[isAr ? "ar" : "en"]}
          >
            <span className="strip-pill-num">{index + 1}</span>
            {def.emoji && (
              <span className="strip-pill-emoji" aria-hidden>
                {def.emoji}
              </span>
            )}
            <span className="strip-pill-label">{label[isAr ? "ar" : "en"]}</span>
          </div>
        );
      })}
    </div>
  );
}

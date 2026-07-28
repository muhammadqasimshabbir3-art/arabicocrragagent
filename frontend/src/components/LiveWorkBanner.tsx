import { useEffect, useState } from "react";
import { Loader2, ScanText, Sparkles } from "lucide-react";
import type { StepState } from "../types";

type Phase = "idle" | "prepare" | "scan" | "answer";

function detectPhase(steps: StepState[], running: boolean): Phase {
  if (!running) return "idle";
  const active = steps.find((s) => s.status === "running");
  if (!active) return "prepare";
  if (active.id === "ingest_document") return "scan";
  if (
    active.id === "query_documents" ||
    active.id === "query_knowledge_base" ||
    active.id === "query_planner" ||
    active.id === "web_search" ||
    active.id === "summarize_document" ||
    active.id === "call_model"
  ) {
    return "answer";
  }
  return "prepare";
}

const COPY: Record<
  Exclude<Phase, "idle">,
  { arTitle: string; enTitle: string; arHint: string; enHint: string }
> = {
  prepare: {
    arTitle: "جاري التحضير…",
    enTitle: "Preparing…",
    arHint: "الوكيل يعمل — لحظة من فضلك",
    enHint: "Agent is working — please wait",
  },
  scan: {
    arTitle: "جاري الفهرسة…",
    enTitle: "Indexing…",
    arHint: "OCR وتقطيع وتضمين — قد يستغرق وقتًا على المستندات الكبيرة",
    enHint: "OCR, chunking, embeddings — large docs can take a minute",
  },
  answer: {
    arTitle: "جاري إنشاء الإجابة…",
    enTitle: "Generating answer…",
    arHint: "استرجاع المصادر وكتابة الرد الموثّق",
    enHint: "Retrieving sources and writing a grounded reply",
  },
};

interface LiveWorkBannerProps {
  steps: StepState[];
  running: boolean;
  uiLang: "ar" | "en";
}

export function LiveWorkBanner({ steps, running, uiLang }: LiveWorkBannerProps) {
  const isAr = uiLang === "ar";
  const phase = detectPhase(steps, running);
  const [elapsedSec, setElapsedSec] = useState(0);

  useEffect(() => {
    if (!running) {
      setElapsedSec(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - started) / 1000));
    }, 400);
    return () => window.clearInterval(timer);
  }, [running]);

  if (!running || phase === "idle") return null;

  const copy = COPY[phase];
  const active = steps.find((s) => s.status === "running");
  const Icon = phase === "scan" ? ScanText : phase === "answer" ? Sparkles : Loader2;

  return (
    <section
      className={`live-work-banner phase-${phase}`}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="live-work-top">
        <div className="live-work-label">
          <Icon size={20} className={phase === "prepare" ? "spin" : "live-icon-pulse"} />
          <div>
            <strong>{isAr ? copy.arTitle : copy.enTitle}</strong>
            <p>{isAr ? copy.arHint : copy.enHint}</p>
          </div>
        </div>
        <div className="live-work-meta">
          <span className="live-work-time">
            {elapsedSec < 60
              ? isAr
                ? `${elapsedSec} ث`
                : `${elapsedSec}s`
              : isAr
                ? `${Math.floor(elapsedSec / 60)} د ${elapsedSec % 60} ث`
                : `${Math.floor(elapsedSec / 60)}m ${elapsedSec % 60}s`}
          </span>
        </div>
      </div>

      <div className="live-work-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}>
        <div className="live-work-bar-fill indeterminate" />
        <div className="live-work-bar-shine" aria-hidden />
      </div>

      {active?.detail && <p className="live-work-detail">{active.detail}</p>}
    </section>
  );
}

import type { StepState, WorkflowStep } from "../types";

export const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    id: "prepare_input",
    nodes: ["prepare_input"],
    label: "Prepare input",
    description: "Format the user input and message history",
    emoji: "☀️",
  },
  {
    id: "decision_agent",
    nodes: ["decision_agent"],
    label: "Decision routing",
    description: "Choose ingest, PDF query, knowledge DB, web search, summarize, or chat",
    emoji: "🧭",
  },
  {
    id: "ingest_document",
    nodes: ["ingest_document"],
    label: "Document ingest",
    description: "Load → OCR → normalize → chunk → embed → index",
    optional: true,
    emoji: "📜",
  },
  {
    id: "query_documents",
    nodes: ["query_documents"],
    label: "Grounded QA",
    description: "Retrieve → trust-layer read → answer with citations",
    optional: true,
    emoji: "🐪",
  },
  {
    id: "query_planner",
    nodes: ["query_planner"],
    label: "Query planner",
    description: "Write a READ-only database retrieval plan",
    optional: true,
    emoji: "📝",
  },
  {
    id: "query_knowledge_base",
    nodes: ["query_knowledge_base"],
    label: "Knowledge DB",
    description: "Hybrid BM25 + vector search → grounded answer",
    optional: true,
    emoji: "🗄️",
  },
  {
    id: "web_search",
    nodes: ["web_search"],
    label: "Web search",
    description: "Live web/browser lookup when the question needs online info",
    optional: true,
    emoji: "🌐",
  },
  {
    id: "summarize_document",
    nodes: ["summarize_document"],
    label: "Summarize",
    description: "Index document and produce a grounded summary",
    optional: true,
    emoji: "🌵",
  },
  {
    id: "call_model",
    nodes: ["call_model"],
    label: "General LLM",
    description: "Conversational help without inventing document facts",
    optional: true,
    emoji: "💬",
  },
];

/** Live step detail lines — bilingual for UI language toggle. */
export const STEP_DETAILS: Record<string, { ar: string; en: string }> = {
  prepare_input: { ar: "تم تجهيز الإدخال.", en: "Input structured." },
  decision_agent: { ar: "تم التوجيه.", en: "Routed." },
  ingest_document: { ar: "تمت فهرسة المستند.", en: "Document indexed." },
  query_documents: {
    ar: "إجابة موثّقة مع المصادر.",
    en: "Grounded answer with citations.",
  },
  query_planner: {
    ar: "خطة استعلام عربية جاهزة.",
    en: "Arabic READ-only query plan ready.",
  },
  query_knowledge_base: {
    ar: "استرجاع بتضمين عربي + BM25 ثم إجابة.",
    en: "Arabic embedding + BM25 retrieve then answer.",
  },
  web_search: {
    ar: "بحث ويب مباشر ثم إجابة.",
    en: "Live web search then answer.",
  },
  summarize_document: { ar: "الملخص جاهز.", en: "Summary ready." },
  call_model: { ar: "اكتملت المحادثة.", en: "Chat complete." },
  _starting: { ar: "الوكيل يبدأ…", en: "Agent starting…" },
  _done: { ar: "تم", en: "Done" },
};

/** Backend route → bilingual task-plan banner. */
export const ROUTE_SUMMARIES: Record<string, { ar: string; en: string }> = {
  ingest_document: {
    ar: "فهرسة المستند → OCR → تقطيع → تضمين → فهرسة",
    en: "Ingest document → OCR → chunk → embed → index",
  },
  query_documents: {
    ar: "استرجاع → طبقة ثقة → إجابة موثّقة مع المصادر",
    en: "Retrieve → trust-layer read → grounded answer + citations",
  },
  query_knowledge_base: {
    ar: "توجيه → تحويل السؤال إلى عربي → تضمين/بحث → إجابة",
    en: "Decision → Arabic query rewrite → embed/search → answer",
  },
  web_search: {
    ar: "بحث ويب / متصفح → إجابة من النتائج الحية",
    en: "Web/browser search → answer from live results",
  },
  summarize_document: {
    ar: "فهرسة (إن لزم) → تلخيص المستند",
    en: "Ingest (if needed) → summarize document",
  },
  call_model: {
    ar: "محادثة عامة",
    en: "General conversation",
  },
};

const NODE_TO_STEP = new Map<string, string>();
for (const step of WORKFLOW_STEPS) {
  for (const node of step.nodes) {
    NODE_TO_STEP.set(node, step.id);
  }
}

export function stepIdForNode(nodeName: string): string | undefined {
  return NODE_TO_STEP.get(nodeName);
}

export function initialStepStates(): StepState[] {
  return WORKFLOW_STEPS.map((step) => ({
    id: step.id,
    status: "pending" as const,
  }));
}

/** English detail stored in step state (keys stay stable; UI localizes on render). */
export function detailForNode(nodeName: string, payload: Record<string, unknown>): string {
  const stepId = stepIdForNode(nodeName) ?? nodeName;
  if (stepId === "decision_agent") {
    const route = payload.agent_route ? String(payload.agent_route) : "";
    if (route && ROUTE_SUMMARIES[route]) return ROUTE_SUMMARIES[route].en;
    if (payload.task_plan_summary) return String(payload.task_plan_summary);
    return STEP_DETAILS.decision_agent.en;
  }
  return STEP_DETAILS[stepId]?.en ?? STEP_DETAILS._done.en;
}

export function localizeStepDetail(
  stepId: string,
  storedDetail: string | undefined,
  uiLang: "ar" | "en",
  agentRoute?: string,
): string | undefined {
  const lang = uiLang === "ar" ? "ar" : "en";
  if (!storedDetail) {
    return STEP_DETAILS[stepId]?.[lang];
  }
  if (
    storedDetail === STEP_DETAILS._starting.en ||
    storedDetail === STEP_DETAILS._starting.ar
  ) {
    return STEP_DETAILS._starting[lang];
  }
  if (stepId === "decision_agent") {
    if (agentRoute && ROUTE_SUMMARIES[agentRoute]) {
      return ROUTE_SUMMARIES[agentRoute][lang];
    }
    for (const summary of Object.values(ROUTE_SUMMARIES)) {
      if (storedDetail === summary.en || storedDetail === summary.ar) {
        return summary[lang];
      }
    }
  }
  const known = STEP_DETAILS[stepId];
  if (known) {
    if (storedDetail === known.en || storedDetail === known.ar) {
      return known[lang];
    }
    return storedDetail;
  }
  return storedDetail;
}

export function localizeTaskPlan(
  summary: string | undefined,
  agentRoute: string | undefined,
  uiLang: "ar" | "en",
): string | undefined {
  const lang = uiLang === "ar" ? "ar" : "en";
  if (agentRoute && ROUTE_SUMMARIES[agentRoute]) {
    return ROUTE_SUMMARIES[agentRoute][lang];
  }
  if (!summary) return undefined;
  for (const entry of Object.values(ROUTE_SUMMARIES)) {
    if (summary === entry.en || summary === entry.ar) return entry[lang];
  }
  return summary;
}

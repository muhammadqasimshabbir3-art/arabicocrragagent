import type { AgentState, StepState, StepStatus } from "../types";
import {
  detailForNode,
  initialStepStates,
  stepIdForNode,
  WORKFLOW_STEPS,
} from "./workflowSteps";

export interface ThreadSnapshot {
  values?: AgentState | Record<string, unknown>;
  next?: string[];
  tasks?: Array<{ name: string; result?: unknown; error?: string | null }>;
}

export function normalizeStreamEvent(raw: string): string {
  return raw.split("|")[0]?.trim() ?? raw;
}

export function normalizeNodeName(raw: string): string {
  const base = raw.split("|")[0]?.trim() ?? raw;
  return base.replace(/^graph:/, "");
}

export function parseEventNodeName(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const evt = data as Record<string, unknown>;
  const metadata = evt.metadata as Record<string, unknown> | undefined;
  const fromMeta = metadata?.langgraph_node;
  if (typeof fromMeta === "string" && fromMeta) return fromMeta;
  if (typeof evt.name === "string" && evt.name) return evt.name;
  return null;
}

export function parseEventNodeOutput(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== "object") return {};
  const evt = data as Record<string, unknown>;
  const payload = evt.data;
  if (!payload || typeof payload !== "object") return {};
  const record = payload as Record<string, unknown>;
  const output = record.output;
  if (output && typeof output === "object" && !Array.isArray(output)) {
    return output as Record<string, unknown>;
  }
  return record;
}

export function isChainStartEvent(data: unknown): boolean {
  if (!data || typeof data !== "object") return false;
  const evt = data as Record<string, unknown>;
  return evt.event === "on_chain_start" || evt.event === "on_tool_start";
}

export function isChainEndEvent(data: unknown): boolean {
  if (!data || typeof data !== "object") return false;
  const evt = data as Record<string, unknown>;
  return evt.event === "on_chain_end" || evt.event === "on_tool_end";
}

function activeGraphNodes(thread: ThreadSnapshot): string[] {
  const fromTasks = (thread.tasks ?? [])
    .filter((task) => !task.error && task.result === undefined)
    .map((task) => normalizeNodeName(task.name));
  if (fromTasks.length) return fromTasks;
  return (thread.next ?? []).map((raw) => normalizeNodeName(raw));
}

export function markStepRunning(steps: StepState[], stepId: string, detail?: string): StepState[] {
  return steps.map((step) => {
    if (step.id === stepId) {
      return {
        ...step,
        status: "running" as StepStatus,
        detail: detail ?? step.detail,
        startedAt: step.startedAt ?? new Date().toISOString(),
      };
    }
    if (step.status === "running" && step.id !== stepId) {
      return { ...step, status: "pending" as StepStatus, detail: undefined };
    }
    return step;
  });
}

export function markStepCompleted(steps: StepState[], stepId: string, detail?: string): StepState[] {
  return steps.map((step) => {
    if (step.id === stepId) {
      return {
        ...step,
        status: "completed" as StepStatus,
        detail: detail ?? step.detail,
        completedAt: new Date().toISOString(),
      };
    }
    return step;
  });
}

export function startPipeline(steps: StepState[]): StepState[] {
  if (!WORKFLOW_STEPS.length) return steps;
  return markStepRunning(steps, WORKFLOW_STEPS[0].id, "Agent starting…");
}

export function applyCompletedNodes(
  steps: StepState[],
  completedNodes: Set<string>,
  state: AgentState,
): StepState[] {
  let next = steps;
  for (const node of completedNodes) {
    const stepId = stepIdForNode(node);
    if (!stepId) continue;
    next = markStepCompleted(next, stepId, detailForNode(node, state as Record<string, unknown>));
  }
  return next;
}

export function buildPipelineFromState(
  state: AgentState,
  nextNodes: string[] = [],
  thread: ThreadSnapshot = {},
  knownCompleted: Set<string> = new Set(),
): { steps: StepState[]; completedNodes: Set<string> } {
  const completedNodes = new Set(knownCompleted);
  let steps = initialStepStates();

  if (state.task_plan_summary || state.agent_route) {
    completedNodes.add("prepare_input");
    completedNodes.add("decision_agent");
  }

  steps = applyCompletedNodes(steps, completedNodes, state);

  const activeNodes = activeGraphNodes({ ...thread, next: nextNodes });
  if (activeNodes.length > 0) {
    const activeNode = activeNodes[0];
    const stepId = stepIdForNode(activeNode);
    if (stepId) {
      steps = markStepRunning(steps, stepId, detailForNode(activeNode, state as Record<string, unknown>));
    }
  }

  const finished =
    nextNodes.length === 0 &&
    ((state.messages && state.messages.length > 0) || Boolean(state.task_plan_summary));

  if (finished) {
    steps = steps.map((s) => {
      if (s.status === "running") {
        return { ...s, status: "completed" as StepStatus, completedAt: new Date().toISOString() };
      }
      if (s.status === "pending") {
        const def = WORKFLOW_STEPS.find((d) => d.id === s.id);
        if (def?.optional) {
          return { ...s, status: "skipped" as StepStatus, detail: "Not used this run" };
        }
      }
      return s;
    });
  }

  return { steps, completedNodes };
}

export function finalizePipeline(
  _steps: StepState[],
  state: AgentState,
  completedNodes: Set<string>,
  nextNodes?: string[],
  thread?: ThreadSnapshot,
): StepState[] {
  const rebuilt = buildPipelineFromState(state, nextNodes ?? [], thread ?? {}, completedNodes);
  completedNodes.clear();
  for (const node of rebuilt.completedNodes) {
    completedNodes.add(node);
  }
  return rebuilt.steps;
}

export function payloadForNode(
  _nodeName: string,
  nodeUpdate: Record<string, unknown>,
  state: AgentState,
): Record<string, unknown> {
  return { ...state, ...nodeUpdate };
}

export function progressPercent(steps: StepState[]): number {
  const relevant = WORKFLOW_STEPS.filter((def) => {
    const state = steps.find((s) => s.id === def.id);
    if (!def.optional) return true;
    return state?.status === "running" || state?.status === "completed" || state?.status === "error";
  });
  const total = Math.max(relevant.length, 1);
  const completed = relevant.filter((def) => {
    const state = steps.find((s) => s.id === def.id);
    return state?.status === "completed" || state?.status === "skipped";
  }).length;
  const stillRunning = steps.some((s) => s.status === "running");
  const running = stillRunning ? 0.5 : 0;
  const raw = Math.round(((completed + running) / total) * 100);
  // While a node is active, never report 100% (optional-step flicker used to spike to 100).
  if (stillRunning) return Math.min(99, raw);
  return Math.min(100, raw);
}

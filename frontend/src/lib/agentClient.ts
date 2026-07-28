import { Client } from "@langchain/langgraph-sdk";
import { ASSISTANT_ID, LANGSMITH_API_KEY, LANGGRAPH_API_URL, GRAPH_RUN_CONFIG } from "../config";
import type { RunRequest } from "../types";

function resolveApiUrlForSdk(apiUrl: string): string {
  if (apiUrl.startsWith("/")) {
    return new URL(apiUrl, window.location.origin).toString().replace(/\/$/, "");
  }
  return apiUrl.replace(/\/$/, "");
}

export function healthCheckUrl(): string {
  return `${resolveApiUrlForSdk(LANGGRAPH_API_URL)}/ok`;
}

/** Shared SDK client — re-used across calls. */
export function getAgentClient(): Client {
  return new Client({
    apiUrl: resolveApiUrlForSdk(LANGGRAPH_API_URL),
    ...(LANGSMITH_API_KEY ? { apiKey: LANGSMITH_API_KEY } : {}),
  });
}

/** Check if the LangGraph server is reachable. */
export async function checkAgentHealth(): Promise<{ ok: boolean; latencyMs: number }> {
  const start = performance.now();
  const elapsed = () => Math.round(performance.now() - start);

  try {
    const res = await fetch(healthCheckUrl(), {
      method: "GET",
      credentials: "include",
      headers: {
        Accept: "application/json",
      },
    });
    if (res.ok) {
      try {
        const ct = res.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const body = await res.json();
          if (body && body.ok === true) return { ok: true, latencyMs: elapsed() };
        }
      } catch {
        // JSON parse failed — likely HTML index page.
      }
    }
  } catch {
    // fall through to SDK check
  }

  try {
    const client = getAgentClient();
    await client.assistants.search({ limit: 1 });
    return { ok: true, latencyMs: elapsed() };
  } catch {
    return { ok: false, latencyMs: 0 };
  }
}

export function buildAgentInput(request: RunRequest) {
  const payload =
    request.document_data_base64 ||
    (request.pdf_analysis_enabled ? request.pdf_data_base64 : "") ||
    request.pdf_data_base64 ||
    "";
  const filename =
    request.document_filename || request.pdf_filename || "uploaded.pdf";
  const summarize = Boolean(request.summarize_only || request.pdf_summarize_only);

  return {
    user_input: request.user_input.trim(),
    response_language: request.response_language === "en" ? "en" : "ar",
    ...(payload
      ? {
          pdf_data_base64: payload,
          pdf_filename: filename,
          pdf_summarize_only: summarize,
          document_data_base64: payload,
          document_filename: filename,
          document_mime_type: request.document_mime_type || "",
          summarize_only: summarize,
        }
      : {}),
    messages: request.conversation_messages ?? [],
  };
}

export async function fetchThreadState(threadId: string) {
  return getAgentClient().threads.getState(threadId);
}

export async function fetchRun(threadId: string, runId: string) {
  return getAgentClient().runs.get(threadId, runId);
}

const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);

export async function fetchLatestActiveRun(threadId: string): Promise<string | null> {
  try {
    const runs = await getAgentClient().runs.list(threadId, { limit: 10 });
    const active = runs.find((run) => ACTIVE_RUN_STATUSES.has(run.status));
    return active?.run_id ?? null;
  } catch {
    return null;
  }
}

export async function isRunStillActive(threadId: string, runId: string | null): Promise<boolean> {
  try {
    const thread = await fetchThreadState(threadId);
    if ((thread.next?.length ?? 0) > 0) return true;
    let resolved = runId;
    if (!resolved) {
      resolved = await fetchLatestActiveRun(threadId);
      if (!resolved) return false;
    }
    const run = await fetchRun(threadId, resolved);
    return ACTIVE_RUN_STATUSES.has(run.status);
  } catch {
    return false;
  }
}

export function joinRunStream(threadId: string, runId: string, signal?: AbortSignal) {
  return getAgentClient().runs.joinStream(threadId, runId, {
    streamMode: ["updates", "values", "events"],
    cancelOnDisconnect: false,
    signal,
  });
}

/** Create a run first (captures run_id for reconnect), then join its stream. */
export async function streamAgentRun(
  request: RunRequest,
  signal?: AbortSignal,
) {
  const client = getAgentClient();
  const thread = await client.threads.create();
  const input = buildAgentInput(request);
  const run = await client.runs.create(thread.thread_id, ASSISTANT_ID, {
    input,
    config: GRAPH_RUN_CONFIG,
    streamMode: ["updates", "values", "events"],
    onDisconnect: "continue",
  });
  const stream = client.runs.joinStream(thread.thread_id, run.run_id, {
    streamMode: ["updates", "values", "events"],
    cancelOnDisconnect: false,
    signal,
  });
  return { threadId: thread.thread_id, runId: run.run_id, stream };
}

export async function cancelAgentRun(threadId: string, runId: string | null): Promise<void> {
  if (!runId) return;
  try {
    await getAgentClient().runs.cancel(threadId, runId);
  } catch {
    // Server may not support cancel; UI abort still stops local tracking.
  }
}

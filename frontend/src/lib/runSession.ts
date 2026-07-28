/** Persist active LangGraph run so a browser refresh can reconnect. */
import type { AgentState, StepState } from "../types";

const STORAGE_KEY = "wathiqa-run-session";
const MEMORY_KEY = "wathiqa-session-memory";

export interface RunSession {
  threadId: string;
  runId: string | null;
  startedAt: string;
  running?: boolean;
  steps?: StepState[];
  partialState?: AgentState;
  reconnected?: boolean;
}

export type MemoryMessage = { type: "human" | "ai"; content: string };

function readStorage(): RunSession | null {
  for (const store of [sessionStorage, localStorage]) {
    try {
      const raw = store.getItem(STORAGE_KEY);
      if (!raw) continue;
      const parsed = JSON.parse(raw) as RunSession;
      if (parsed.threadId) return parsed;
    } catch {
      // ignore
    }
  }
  return null;
}

function writeStorage(session: RunSession): void {
  const raw = JSON.stringify(session);
  for (const store of [sessionStorage, localStorage]) {
    try {
      store.setItem(STORAGE_KEY, raw);
    } catch {
      // ignore quota / private mode
    }
  }
}

export function saveRunSession(session: RunSession): void {
  writeStorage(session);
}

export function patchRunSession(patch: Partial<RunSession>): void {
  const current = loadRunSession();
  if (!current?.threadId) return;
  saveRunSession({ ...current, ...patch });
}

export function loadRunSession(): RunSession | null {
  return readStorage();
}

export function clearRunSession(): void {
  for (const store of [sessionStorage, localStorage]) {
    try {
      store.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }
}

/** Short hidden chat memory for the agent (not shown in Answer UI). */
export function loadSessionMemory(): MemoryMessage[] {
  try {
    const raw = sessionStorage.getItem(MEMORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as MemoryMessage[];
    return Array.isArray(parsed) ? parsed.filter((m) => m?.content && (m.type === "human" || m.type === "ai")) : [];
  } catch {
    return [];
  }
}

export function saveSessionMemory(messages: MemoryMessage[]): void {
  try {
    sessionStorage.setItem(MEMORY_KEY, JSON.stringify(messages));
  } catch {
    // ignore
  }
}

export function clearSessionMemory(): void {
  try {
    sessionStorage.removeItem(MEMORY_KEY);
  } catch {
    // ignore
  }
}

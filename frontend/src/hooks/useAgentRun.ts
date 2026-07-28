import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  cancelAgentRun,
  fetchLatestActiveRun,
  fetchRun,
  fetchThreadState,
  isRunStillActive,
  joinRunStream,
  streamAgentRun,
} from "../lib/agentClient";
import {
  clearRunSession,
  clearSessionMemory,
  loadRunSession,
  loadSessionMemory,
  saveRunSession,
  saveSessionMemory,
} from "../lib/runSession";
import {
  buildPipelineFromState,
  finalizePipeline,
  isChainEndEvent,
  isChainStartEvent,
  markStepCompleted,
  markStepRunning,
  normalizeNodeName,
  normalizeStreamEvent,
  parseEventNodeName,
  parseEventNodeOutput,
  payloadForNode,
  startPipeline,
  type ThreadSnapshot,
} from "../lib/streamProgress";
import {
  detailForNode,
  initialStepStates,
  stepIdForNode,
} from "../lib/workflowSteps";
import type { AgentState, LogEntry, RunRequest, StepState } from "../types";

const POLL_MS = 1000;
/** Short hidden session memory (messages) for the agent — not shown in Answer UI. */
const SESSION_MEMORY_MESSAGES = 8;
/** Keep memory compact so long RAG answers don't blow the context window. */
const MEMORY_AI_MAX_CHARS = 1200;
const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);

function trimForMemory(
  messages: Array<{ type: "human" | "ai"; content: string }>,
): Array<{ type: "human" | "ai"; content: string }> {
  return messages
    .filter((m) => m.content.trim())
    .map((m) =>
      m.type === "ai" && m.content.length > MEMORY_AI_MAX_CHARS
        ? { ...m, content: `${m.content.slice(0, MEMORY_AI_MAX_CHARS)}…` }
        : m,
    )
    .slice(-SESSION_MEMORY_MESSAGES);
}

function nowIso() {
  return new Date().toISOString();
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function makeLog(level: LogEntry["level"], message: string): LogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    time: new Date().toLocaleTimeString(),
    level,
    message,
  };
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          return String((part as { text?: string }).text ?? "");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  return content != null ? String(content).trim() : "";
}

/** Latest AI reply after the latest human turn — never falls back to old chat. */
function extractLatestAnswer(state: AgentState | null): string {
  const messages = state?.messages ?? [];
  let lastHuman = -1;
  for (let i = 0; i < messages.length; i += 1) {
    const type = (messages[i].type ?? "").toLowerCase();
    if (type === "human" || type === "humanmessage" || type.includes("human")) {
      lastHuman = i;
    }
  }
  for (let i = messages.length - 1; i > lastHuman; i -= 1) {
    const msg = messages[i];
    const type = (msg.type ?? "").toLowerCase();
    if (type === "ai" || type === "aimessage" || type.includes("ai")) {
      const text = messageText(msg.content);
      if (text) return text;
    }
  }
  return "";
}

function hydrateFromSession() {
  const session = loadRunSession();
  if (!session?.threadId) {
    return {
      steps: initialStepStates(),
      result: null as AgentState | null,
      running: false,
      threadId: null as string | null,
      runId: null as string | null,
      reconnected: false,
    };
  }
  return {
    steps: session.steps?.length ? session.steps : initialStepStates(),
    result: session.partialState ?? null,
    running: Boolean(session.running),
    threadId: session.threadId,
    runId: session.runId,
    reconnected: Boolean(session.reconnected),
  };
}

function publishSteps(setSteps: (s: StepState[]) => void, next: StepState[]) {
  flushSync(() => setSteps(next));
}

async function resolveRunId(threadId: string, runId: string | null): Promise<string | null> {
  if (runId) return runId;
  return fetchLatestActiveRun(threadId);
}

export function useAgentRun() {
  const hydrated = hydrateFromSession();
  const abortRef = useRef<AbortController | null>(null);
  const pollRef = useRef<number | null>(null);
  const completedNodesRef = useRef<Set<string>>(new Set());
  const trackingRef = useRef(hydrated.running);
  const resumeLockRef = useRef(false);
  const threadIdRef = useRef<string | null>(hydrated.threadId);
  const runIdRef = useRef<string | null>(hydrated.runId);
  const stepsLiveRef = useRef<StepState[]>(hydrated.steps);

  const [running, setRunning] = useState(hydrated.running);
  const [reconnected, setReconnected] = useState(hydrated.reconnected);
  const [steps, setSteps] = useState<StepState[]>(hydrated.steps);
  const [result, setResult] = useState<AgentState | null>(hydrated.result);
  const [answer, setAnswer] = useState(() => extractLatestAnswer(hydrated.result));
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [threadId, setThreadId] = useState<string | null>(hydrated.threadId);
  const [runId, setRunId] = useState<string | null>(hydrated.runId);
  const [conversationMessages, setConversationMessages] = useState<
    Array<{ type: "human" | "ai"; content: string }>
  >(() => loadSessionMemory());

  stepsLiveRef.current = steps;

  const persistMemory = useCallback(
    (messages: Array<{ type: "human" | "ai"; content: string }>) => {
      const trimmed = trimForMemory(messages);
      setConversationMessages(trimmed);
      saveSessionMemory(trimmed);
    },
    [],
  );

  const clearAnswer = useCallback(() => {
    // Clear only what the user sees. Keep short session memory for the agent.
    setAnswer("");
    setError(null);
  }, []);

  const persistSession = useCallback(
    (patch: {
      threadId?: string | null;
      runId?: string | null;
      running?: boolean;
      steps?: StepState[];
      partialState?: AgentState | null;
      reconnected?: boolean;
    }) => {
      const tid = patch.threadId ?? threadIdRef.current;
      if (!tid) return;
      const rid = patch.runId !== undefined ? patch.runId : runIdRef.current;
      saveRunSession({
        threadId: tid,
        runId: rid,
        startedAt: nowIso(),
        running: patch.running ?? trackingRef.current,
        steps: patch.steps ?? undefined,
        partialState: patch.partialState ?? undefined,
        reconnected: patch.reconnected ?? undefined,
      });
    },
    [],
  );

  const pushLog = useCallback((level: LogEntry["level"], message: string) => {
    flushSync(() => setLogs((prev) => [...prev, makeLog(level, message)]));
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const publishFromState = useCallback(
    (state: AgentState, nextNodes: string[], thread: ThreadSnapshot = {}) => {
      const rebuilt = buildPipelineFromState(
        state,
        nextNodes,
        thread,
        completedNodesRef.current,
      );
      completedNodesRef.current = rebuilt.completedNodes;
      flushSync(() => setResult(state));
      publishSteps(setSteps, rebuilt.steps);
      persistSession({
        steps: rebuilt.steps,
        partialState: state,
        running: trackingRef.current,
      });
      return rebuilt;
    },
    [persistSession],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    stopPolling();
    trackingRef.current = false;
    completedNodesRef.current = new Set();
    threadIdRef.current = null;
    runIdRef.current = null;
    clearRunSession();
    clearSessionMemory();
    setReconnected(false);
    setRunning(false);
    setSteps(initialStepStates());
    setResult(null);
    setAnswer("");
    setError(null);
    setLogs([]);
    setThreadId(null);
    setRunId(null);
    setConversationMessages([]);
  }, [stopPolling]);

  const refreshThread = useCallback(
    async (activeThreadId: string, latestState: AgentState = {}) => {
      const thread = await fetchThreadState(activeThreadId);
      const merged = { ...latestState, ...(thread.values as AgentState) } as AgentState;
      const nextNodes = thread.next ?? [];
      publishFromState(merged, nextNodes, thread as ThreadSnapshot);
      return { state: merged, next: nextNodes, thread: thread as ThreadSnapshot };
    },
    [publishFromState],
  );

  const startPolling = useCallback(
    (activeThreadId: string) => {
      stopPolling();
      pollRef.current = window.setInterval(() => {
        if (!trackingRef.current) return;
        void refreshThread(activeThreadId).catch(() => {
          // thread may be gone
        });
      }, POLL_MS);
    },
    [refreshThread, stopPolling],
  );

  const finishRun = useCallback(
    async (
      activeThreadId: string | null,
      activeRunId: string | null,
      finalState?: AgentState,
      options: { force?: boolean } = {},
    ) => {
      const aiText = finalState ? extractLatestAnswer(finalState) : "";
      const stillActive =
        Boolean(activeThreadId) && (await isRunStillActive(activeThreadId!, activeRunId));

      // Answer already arrived but run status can lag — don't leave the UI stuck on Wait/Stop.
      if (stillActive && !options.force && !aiText) {
        persistSession({ threadId: activeThreadId, runId: activeRunId, running: true });
        return false;
      }

      trackingRef.current = false;
      stopPolling();
      setRunning(false);
      abortRef.current = null;

      if (finalState) {
        const finalized = finalizePipeline(
          buildPipelineFromState(finalState, [], {}, completedNodesRef.current).steps,
          finalState,
          completedNodesRef.current,
          [],
          {},
        );
        publishSteps(setSteps, finalized);
        setResult(finalState);
        if (aiText) {
          setAnswer(aiText);
          setConversationMessages((prev) => {
            const base =
              prev.length && prev[prev.length - 1]?.type === "ai"
                ? prev.slice(0, -1)
                : prev;
            const next = trimForMemory([
              ...base,
              { type: "ai" as const, content: aiText },
            ]);
            saveSessionMemory(next);
            return next;
          });
        } else {
          setAnswer("");
        }
      }

      clearRunSession();
      pushLog("success", "Workflow finished");
      return true;
    },
    [persistSession, pushLog, stopPolling],
  );

  const waitForRunViaPolling = useCallback(
    async (activeThreadId: string, activeRunId: string | null, abort: AbortController) => {
      while (!abort.signal.aborted && trackingRef.current) {
        const { state, next, thread } = await refreshThread(activeThreadId);
        const resolvedRunId = await resolveRunId(activeThreadId, activeRunId);
        if (resolvedRunId) {
          runIdRef.current = resolvedRunId;
          setRunId(resolvedRunId);
        }
        let done = next.length === 0;
        if (resolvedRunId) {
          try {
            const runMeta = await fetchRun(activeThreadId, resolvedRunId);
            done = !ACTIVE_RUN_STATUSES.has(runMeta.status) && next.length === 0;
            if (runMeta.status === "error") {
              throw new Error("Agent run failed on the server");
            }
          } catch (err) {
            if (err instanceof Error && err.message.includes("failed")) throw err;
            done = next.length === 0;
          }
        }
        if (done) {
          const finalized = finalizePipeline(
            buildPipelineFromState(state, next, thread, completedNodesRef.current).steps,
            state,
            completedNodesRef.current,
            next,
            thread,
          );
          publishSteps(setSteps, finalized);
          setResult(state);
          return state;
        }
        await sleep(POLL_MS);
      }
      return null;
    },
    [refreshThread],
  );

  const processStream = useCallback(
    async (
      stream: AsyncIterable<{ event?: string; data?: unknown }>,
      ctx: {
        threadId: string;
        abort: AbortController;
        initialState?: AgentState;
      },
    ) => {
      let latestState: AgentState = ctx.initialState ?? {};
      let latestNext: string[] = [];
      let latestThread: ThreadSnapshot = {};
      const loggedNodes = new Set<string>();
      let streamDisconnected = false;

      const syncFromThread = async (partial?: AgentState) => {
        const refreshed = await refreshThread(ctx.threadId, partial ?? latestState);
        latestState = refreshed.state;
        latestNext = refreshed.next;
        latestThread = refreshed.thread;
        return refreshed;
      };

      const mergeAndPublish = (patch: Record<string, unknown>, nodeName?: string) => {
        latestState = { ...latestState, ...patch } as AgentState;
        if (nodeName) {
          completedNodesRef.current.add(nodeName);
          const stepId = stepIdForNode(nodeName);
          if (stepId) {
            const detail = detailForNode(nodeName, latestState as Record<string, unknown>);
            let nextSteps = markStepCompleted(
              buildPipelineFromState(
                latestState,
                latestNext,
                latestThread,
                completedNodesRef.current,
              ).steps,
              stepId,
              detail,
            );
            // Start next pending required step hint from thread next
            const active = latestNext[0];
            if (active) {
              const activeId = stepIdForNode(normalizeNodeName(active));
              if (activeId) {
                nextSteps = markStepRunning(
                  nextSteps,
                  activeId,
                  detailForNode(normalizeNodeName(active), latestState as Record<string, unknown>),
                );
              }
            }
            publishSteps(setSteps, nextSteps);
            flushSync(() => setResult(latestState));
            persistSession({
              threadId: ctx.threadId,
              steps: nextSteps,
              partialState: latestState,
              running: true,
            });
            return;
          }
        }
        publishFromState(latestState, latestNext, latestThread);
      };

      try {
        for await (const chunk of stream) {
          if (ctx.abort.signal.aborted) {
            streamDisconnected = true;
            break;
          }

          const event = normalizeStreamEvent(String(chunk.event ?? ""));

          if (event === "metadata") {
            const meta = chunk.data as { run_id?: string } | undefined;
            if (meta?.run_id) {
              const id = String(meta.run_id);
              runIdRef.current = id;
              setRunId(id);
              persistSession({ threadId: ctx.threadId, runId: id, running: true });
            }
          }

          if (event === "error") {
            const errData = chunk.data as { message?: string; error?: string };
            throw new Error(errData?.message || errData?.error || "Agent stream error");
          }

          if (event === "events") {
            if (isChainStartEvent(chunk.data)) {
              const nodeName = normalizeNodeName(parseEventNodeName(chunk.data) ?? "");
              if (nodeName && !loggedNodes.has(nodeName)) {
                loggedNodes.add(nodeName);
                pushLog("info", `Node started: ${nodeName}`);
                const stepId = stepIdForNode(nodeName);
                if (stepId) {
                  publishSteps(
                    setSteps,
                    markStepRunning(
                      stepsLiveRef.current,
                      stepId,
                      detailForNode(nodeName, latestState as Record<string, unknown>),
                    ),
                  );
                }
                await syncFromThread();
              }
            }
            if (isChainEndEvent(chunk.data)) {
              const nodeName = normalizeNodeName(parseEventNodeName(chunk.data) ?? "");
              if (nodeName && stepIdForNode(nodeName)) {
                const nodeOutput = parseEventNodeOutput(chunk.data);
                const payload =
                  Object.keys(nodeOutput).length > 0
                    ? nodeOutput
                    : (latestState as Record<string, unknown>);
                const merged = payloadForNode(nodeName, payload, latestState);
                const detail = detailForNode(nodeName, merged);
                pushLog("info", `${nodeName}: ${detail}`);
                await syncFromThread(merged as AgentState);
              }
            }
          }

          if (event === "updates" && chunk.data && typeof chunk.data === "object") {
            for (const [rawNode, nodeUpdate] of Object.entries(chunk.data)) {
              const nodeName = normalizeNodeName(rawNode);
              if (stepIdForNode(nodeName)) {
                const merged = payloadForNode(
                  nodeName,
                  nodeUpdate as Record<string, unknown>,
                  latestState,
                );
                const detail = detailForNode(nodeName, merged);
                if (!loggedNodes.has(`${nodeName}:done`)) {
                  loggedNodes.add(`${nodeName}:done`);
                  pushLog("info", `${nodeName}: ${detail}`);
                }
                mergeAndPublish(merged, nodeName);
              }
            }
            await syncFromThread();
          }

          if (event === "values" && chunk.data) {
            latestState = { ...(latestState as AgentState), ...(chunk.data as AgentState) };
            await syncFromThread(latestState);
          }
        }
      } catch (err) {
        if (!ctx.abort.signal.aborted) throw err;
        streamDisconnected = true;
      }

      if (!streamDisconnected) {
        await syncFromThread(latestState);
        const finalized = finalizePipeline(
          buildPipelineFromState(
            latestState,
            latestNext,
            latestThread,
            completedNodesRef.current,
          ).steps,
          latestState,
          completedNodesRef.current,
          latestNext,
          latestThread,
        );
        flushSync(() => setResult(latestState));
        publishSteps(setSteps, finalized);
        persistSession({
          threadId: ctx.threadId,
          steps: finalized,
          partialState: latestState,
          running: trackingRef.current,
        });
      }

      return { state: latestState, streamDisconnected };
    },
    [persistSession, publishFromState, pushLog, refreshThread],
  );

  const trackRun = useCallback(
    async (
      activeThreadId: string,
      activeRunId: string | null,
      options: { resume?: boolean; initialState?: AgentState } = {},
    ) => {
      if (resumeLockRef.current) return;
      resumeLockRef.current = true;

      const abort = new AbortController();
      abortRef.current = abort;
      trackingRef.current = true;
      setRunning(true);
      setThreadId(activeThreadId);
      threadIdRef.current = activeThreadId;

      if (options.resume) {
        setReconnected(true);
        pushLog("info", "Reconnected to in-progress run after page refresh");
      }

      const resolvedRunId = await resolveRunId(activeThreadId, activeRunId);
      if (resolvedRunId) {
        runIdRef.current = resolvedRunId;
        setRunId(resolvedRunId);
      }

      persistSession({
        threadId: activeThreadId,
        runId: resolvedRunId,
        running: true,
        reconnected: options.resume,
      });

      startPolling(activeThreadId);

      try {
        const { state, next, thread } = await refreshThread(
          activeThreadId,
          options.initialState ?? {},
        );
        publishFromState(state, next, thread);

        const stillActive = await isRunStillActive(activeThreadId, resolvedRunId);

        if (stillActive) {
          if (resolvedRunId) {
            try {
              const stream = joinRunStream(activeThreadId, resolvedRunId, abort.signal);
              const { state: streamed, streamDisconnected } = await processStream(stream, {
                threadId: activeThreadId,
                abort,
                initialState: state,
              });
              if (!streamDisconnected) {
                const finished = await finishRun(activeThreadId, runIdRef.current, streamed);
                if (finished) return;
                pushLog("warn", "Run still active after stream — continuing via polling");
              } else {
                pushLog("warn", "Stream disconnected — continuing via polling");
              }
            } catch {
              pushLog("warn", "Stream rejoin unavailable — tracking via polling");
            }
          }
          const polled = await waitForRunViaPolling(activeThreadId, runIdRef.current, abort);
          await finishRun(activeThreadId, runIdRef.current, polled ?? undefined, { force: true });
          return;
        }

        const finalized = finalizePipeline(
          buildPipelineFromState(state, next, thread, completedNodesRef.current).steps,
          state,
          completedNodesRef.current,
          next,
          thread,
        );
        publishSteps(setSteps, finalized);
        setResult(state);
        await finishRun(activeThreadId, resolvedRunId, state, { force: true });
      } catch (err) {
        if (abort.signal.aborted) {
          pushLog("warn", "Run stopped");
        } else {
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          pushLog("error", message);
        }
        trackingRef.current = false;
        stopPolling();
        setRunning(false);
        persistSession({ threadId: activeThreadId, runId: runIdRef.current, running: false });
      } finally {
        resumeLockRef.current = false;
      }
    },
    [
      finishRun,
      persistSession,
      processStream,
      publishFromState,
      pushLog,
      refreshThread,
      startPolling,
      stopPolling,
      waitForRunViaPolling,
    ],
  );

  const run = useCallback(
    async (request: RunRequest) => {
      if (!request.user_input?.trim()) {
        setError("User input is required.");
        return;
      }

      abortRef.current?.abort();
      stopPolling();
      completedNodesRef.current = new Set();
      clearRunSession();

      setRunning(true);
      setReconnected(false);
      setError(null);
      setResult(null);
      setAnswer("");
      setLogs([]);
      publishSteps(setSteps, startPipeline(initialStepStates()));

      const currentUserMessage = { type: "human" as const, content: request.user_input.trim() };
      // Keep short hidden memory for the agent; Answer UI still shows only the latest reply.
      const memory = conversationMessages
        .filter((m) => m.content.trim())
        .slice(-(SESSION_MEMORY_MESSAGES - 1));
      const conversation = trimForMemory([...memory, currentUserMessage]);
      const finalRequest: RunRequest = {
        ...request,
        conversation_messages: conversation,
      };

      persistMemory(conversation);
      pushLog("info", "Sending request to the document agent…");

      const abort = new AbortController();
      abortRef.current = abort;
      trackingRef.current = true;

      try {
        const { threadId: createdThreadId, runId: createdRunId, stream } = await streamAgentRun(
          finalRequest,
          abort.signal,
        );
        threadIdRef.current = createdThreadId;
        runIdRef.current = createdRunId;
        setThreadId(createdThreadId);
        setRunId(createdRunId);
        pushLog("info", `Thread ${createdThreadId.slice(0, 8)}…`);

        persistSession({
          threadId: createdThreadId,
          runId: createdRunId,
          running: true,
          steps: startPipeline(initialStepStates()),
        });
        startPolling(createdThreadId);

        const { state, streamDisconnected } = await processStream(stream, {
          threadId: createdThreadId,
          abort,
        });

        if (streamDisconnected || (await isRunStillActive(createdThreadId, runIdRef.current))) {
          const polled = await waitForRunViaPolling(createdThreadId, runIdRef.current, abort);
          await finishRun(createdThreadId, runIdRef.current, polled ?? state, { force: true });
        } else {
          const finished = await finishRun(createdThreadId, runIdRef.current, state);
          if (!finished) {
            const polled = await waitForRunViaPolling(createdThreadId, runIdRef.current, abort);
            await finishRun(createdThreadId, runIdRef.current, polled ?? state, { force: true });
          }
        }
      } catch (err) {
        if (abort.signal.aborted) {
          pushLog("warn", "Run stopped");
          persistSession({
            threadId: threadIdRef.current,
            runId: runIdRef.current,
            running: false,
          });
        } else {
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          pushLog("error", message);
          setSteps((prev) =>
            prev.map((s) =>
              s.status === "running" ? { ...s, status: "error", detail: message } : s,
            ),
          );
        }
        trackingRef.current = false;
        stopPolling();
        setRunning(false);
      }
    },
    [
      conversationMessages,
      finishRun,
      persistMemory,
      persistSession,
      processStream,
      pushLog,
      startPolling,
      stopPolling,
      waitForRunViaPolling,
    ],
  );

  const cancel = useCallback(() => {
    const tid = threadIdRef.current;
    const rid = runIdRef.current;
    abortRef.current?.abort();
    trackingRef.current = false;
    stopPolling();
    clearRunSession();
    setRunning(false);
    pushLog("warn", "Stopping agent…");
    if (tid) {
      void cancelAgentRun(tid, rid);
    }
  }, [pushLog, stopPolling]);

  useEffect(() => {
    const session = loadRunSession();
    if (!session?.threadId || !session.running) return;

    void trackRun(session.threadId, session.runId, {
      resume: true,
      initialState: session.partialState ?? {},
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  return {
    running,
    reconnected,
    steps,
    result,
    answer,
    error,
    logs,
    threadId,
    runId,
    conversationMessages,
    run,
    cancel,
    reset,
    clearAnswer,
  };
}

/// <reference types="vite/client" />

interface ImportMetaEnv {
  // ── Backend connection ────────────────────────────────────────────────────
  /** Full URL of the Railway LangGraph server, e.g. https://…up.railway.app */
  readonly VITE_LANGGRAPH_API_URL?: string;
  /** Alias accepted by config.ts */
  readonly VITE_API_URL?: string;
  /** Graph / assistant ID — defaults to "agent" */
  readonly VITE_LANGGRAPH_ASSISTANT_ID?: string;
  /** LangSmith API key sent as x-api-key to authenticated deployments */
  readonly VITE_LANGSMITH_API_KEY?: string;

  // ── Optional form defaults ────────────────────────────────────────────────
  /** Pre-fills the user-input textarea */
  readonly VITE_DEFAULT_USER_INPUT?: string;
  /** "true" / "false" — pre-checks the web-search toggle */
  readonly VITE_DEFAULT_WEB_SEARCH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

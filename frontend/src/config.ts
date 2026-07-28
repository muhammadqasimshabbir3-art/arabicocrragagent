/**
 * Local / deployed connection settings.
 *
 * Local dev (root .env via Vite envDir):
 *   VITE_LANGGRAPH_API_URL=http://127.0.0.1:2024
 *
 * Vercel production — vercel.json proxies /api → Railway (no CORS needed):
 *   VITE_LANGGRAPH_API_URL=/api   (or leave unset; defaults to /api)
 *   VITE_LANGSMITH_API_KEY=<your langsmith key>   (only if server requires auth)
 *
 * Remote dev with smith.langchain.com:
 *   VITE_LANGGRAPH_API_URL=http://127.0.0.1:2024
 *   The LangGraph server CORS config must allow https://smith.langchain.com
 *
 * Local dev — leave VITE_LANGGRAPH_API_URL empty to use Vite proxy /api → LANGGRAPH_PORT
 */
function readApiUrl(): string {
  const candidates = [
    import.meta.env.VITE_LANGGRAPH_API_URL,
    import.meta.env.VITE_API_URL,
  ];
  for (const value of candidates) {
    const trimmed = value?.trim();
    if (trimmed) return trimmed;
  }
  return "";
}

export const LANGGRAPH_API_URL = readApiUrl() || "/api";

export const ASSISTANT_ID =
  import.meta.env.VITE_LANGGRAPH_ASSISTANT_ID?.trim() || "agent";

/** Optional LangSmith / LangGraph Cloud API key for authenticated deployments. */
export const LANGSMITH_API_KEY =
  import.meta.env.VITE_LANGSMITH_API_KEY?.trim() || "";

export const GRAPH_RUN_CONFIG = { recursion_limit: 100 };

/** True when running the Vite production bundle (e.g. on Vercel). */
export const IS_PRODUCTION = import.meta.env.PROD;

/** True when the UI falls back to the local dev proxy instead of a direct API URL. */
export const USES_DEV_PROXY = LANGGRAPH_API_URL === "/api";

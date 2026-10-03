/**
 * Local / deployed connection settings.
 *
 * Local dev (root .env via Vite envDir):
 *   VITE_LANGGRAPH_API_URL=http://127.0.0.1:2024
 *
 * Vercel production — set absolute Railway URL (build-time env on Vercel):
 *   VITE_LANGGRAPH_API_URL=https://your-service.up.railway.app
 *
 * Never put API secrets in VITE_* variables. Lock Railway CORS_ALLOW_ORIGINS
 * to the Vercel origin (see docs/DEPLOYMENT.md).
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

/**
 * Intentionally empty — do not ship LangSmith/LangGraph API keys via VITE_*.
 * Authenticated deployments should use a server-side proxy or private networking.
 */
export const LANGSMITH_API_KEY = "";

/** Keep in sync with backend GRAPH_RUN_CONFIG.recursion_limit */
export const GRAPH_RUN_CONFIG = { recursion_limit: 50 };

/** True when running the Vite production bundle (e.g. on Vercel). */
export const IS_PRODUCTION = import.meta.env.PROD;

/** True when the UI falls back to the local/production proxy instead of a direct API URL. */
export const USES_DEV_PROXY = LANGGRAPH_API_URL === "/api";

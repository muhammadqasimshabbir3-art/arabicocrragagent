import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, XCircle } from "lucide-react";
import { ASSISTANT_ID, LANGSMITH_API_KEY, LANGGRAPH_API_URL } from "../config";
import { healthCheckUrl } from "../lib/agentClient";

interface DiagStep {
  label: string;
  status: "pending" | "running" | "ok" | "fail";
  detail?: string;
}

/** Runs 3 sequential checks and shows exactly where the connection breaks. */
export function ConnectionDiagnostic({
  run,
  uiLang = "en",
}: {
  run: boolean;
  uiLang?: "ar" | "en";
}) {
  const isAr = uiLang === "ar";
  const [steps, setSteps] = useState<DiagStep[]>([
    { label: isAr ? "متغيرات البيئة" : "Env vars set", status: "pending" },
    { label: `GET ${healthCheckUrl()}`, status: "pending" },
    {
      label: isAr ? "POST /threads (تحقق المصادقة)" : "POST /threads (auth check)",
      status: "pending",
    },
  ]);

  useEffect(() => {
    if (!run) return;

    async function diagnose() {
      const update = (i: number, patch: Partial<DiagStep>) =>
        setSteps((prev) => prev.map((s, idx) => (idx === i ? { ...s, ...patch } : s)));

      // Step 0 — env var check
      update(0, { status: "running" });
      const hasUrl = Boolean(LANGGRAPH_API_URL && LANGGRAPH_API_URL !== "/api");
      const hasKey = Boolean(LANGSMITH_API_KEY);
      if (hasUrl) {
        update(0, {
          status: "ok",
          detail: `URL=${LANGGRAPH_API_URL} · Key=${hasKey ? "set ✓" : "not set (optional)"}`,
        });
      } else {
        update(0, { status: "fail", detail: "VITE_LANGGRAPH_API_URL is missing or resolves to /api" });
        return;
      }

      // Step 1 — plain /ok ping
      update(1, { status: "running" });
      try {
        const res = await fetch(healthCheckUrl(), { method: "GET" });
        if (res.ok) {
          // Prefer JSON { ok: true } — static index.html responses are HTML and
          // indicate the frontend is serving the route instead of the API.
          const ct = res.headers.get("content-type") || "";
          if (ct.includes("application/json")) {
            try {
              const body = await res.json();
              if (body && body.ok === true) {
                update(1, { status: "ok", detail: `HTTP ${res.status}` });
              } else {
                update(1, { status: "fail", detail: `Unexpected /ok body — expected {ok:true}` });
                return;
              }
            } catch (e) {
              update(1, { status: "fail", detail: `Invalid JSON from /ok: ${String(e)}` });
              return;
            }
          } else {
            // Likely index.html — helpful guidance to the user
            update(1, {
              status: "fail",
              detail:
                "GET /ok returned HTML (index page). This means the frontend is serving /api — either add a Vercel rewrite to proxy /api → Railway or set VITE_LANGGRAPH_API_URL to the Railway URL.",
            });
            return;
          }
        } else {
          update(1, { status: "fail", detail: `HTTP ${res.status} — server reachable but returned error` });
          return;
        }
      } catch (e) {
        update(1, {
          status: "fail",
          detail: `Network/CORS error: ${String(e)}. Set CORS_ALLOW_ORIGINS on Railway.`,
        });
        return;
      }

      // Step 2 — authenticated POST /threads
      update(2, { status: "running" });
      try {
        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (LANGSMITH_API_KEY) headers["x-api-key"] = LANGSMITH_API_KEY;
        const res = await fetch(`${LANGGRAPH_API_URL}/threads`, {
          method: "POST",
          headers,
          body: JSON.stringify({}),
        });
        const body = await res.text();
        if (res.ok) {
          update(2, { status: "ok", detail: `Thread created ✓ (assistant: ${ASSISTANT_ID})` });
        } else {
          update(2, {
            status: "fail",
            detail: `HTTP ${res.status}: ${body.slice(0, 200)}`,
          });
        }
      } catch (e) {
        update(2, {
          status: "fail",
          detail: `CORS blocked POST /threads: ${String(e)}`,
        });
      }
    }

    void diagnose();
  }, [run]);

  if (!run) return null;

  return (
    <section className="panel" style={{ marginTop: "1rem" }}>
      <div className="panel-title">
        {isAr ? "تشخيص الاتصال" : "Connection diagnostic"}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px", padding: "12px 0" }}>
        {steps.map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
            <span style={{ flexShrink: 0, marginTop: "2px" }}>
              {s.status === "pending" && <span style={{ color: "var(--muted)" }}>○</span>}
              {s.status === "running" && <Loader2 size={16} className="spin" style={{ color: "var(--accent)" }} />}
              {s.status === "ok" && <CheckCircle2 size={16} style={{ color: "var(--success)" }} />}
              {s.status === "fail" && <XCircle size={16} style={{ color: "var(--danger)" }} />}
            </span>
            <div>
              <strong style={{ fontSize: "0.85rem" }}>{s.label}</strong>
              {s.detail && (
                <p style={{ margin: "2px 0 0", fontSize: "0.78rem", color: "var(--muted)", wordBreak: "break-all" }}>
                  {s.detail}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="deploy-note" style={{ marginTop: "8px" }}>
        <AlertTriangle size={12} style={{ display: "inline", marginRight: "4px" }} />
        {isAr ? (
          <>
            إذا فشل الخطوة ٢ بسبب CORS: عيّن <code>CORS_ALLOW_ORIGINS</code> ليشمل أصل الواجهة
            (مثل <code>http://localhost:5173</code>).
          </>
        ) : (
          <>
            If step 2 fails with CORS: set <code>CORS_ALLOW_ORIGINS</code> to include your UI origin
            (e.g. <code>http://localhost:8501</code>, <code>http://localhost:5173</code>, or your
            Vercel URL).
          </>
        )}
      </div>
    </section>
  );
}

import { Loader2, RefreshCw } from "lucide-react";
import { useState } from "react";
import { ConnectionDiagnostic } from "./ConnectionDiagnostic";
import type { ServerStatus } from "../hooks/useServerHealth";

interface ConnectionPanelProps {
  status: ServerStatus;
  onRefresh: () => void;
  uiLang: "ar" | "en";
}

export function ConnectionPanel({ status, onRefresh, uiLang }: ConnectionPanelProps) {
  const [diagRun, setDiagRun] = useState(false);
  const online = status === "online";
  const checking = status === "checking";
  const isAr = uiLang === "ar";

  return (
    <aside className="panel connection-panel connection-panel-compact">
      <div className="backend-status-row">
        <div className="backend-status-label">
          <span
            className={`status-light ${online ? "online" : checking ? "checking" : "offline"}`}
            aria-hidden
          />
          <span className="backend-status-text">
            {checking
              ? isAr
                ? "جاري الفحص…"
                : "Checking…"
              : online
                ? isAr
                  ? "الخادم متصل"
                  : "Server online"
                : isAr
                  ? "الخادم متوقف"
                  : "Server offline"}
          </span>
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={onRefresh}
          title={isAr ? "تحديث الاتصال" : "Refresh connection"}
        >
          {checking ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
        </button>
      </div>

      {!online && !checking && (
        <div className="deploy-note">
          <button
            type="button"
            className="btn ghost small"
            onClick={() => setDiagRun(true)}
            disabled={diagRun}
          >
            {isAr ? "تشخيص الاتصال" : "Run diagnostic"}
          </button>
          <ConnectionDiagnostic run={diagRun} uiLang={uiLang} />
        </div>
      )}
    </aside>
  );
}

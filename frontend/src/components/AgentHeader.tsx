import { Radio, ScanText } from "lucide-react";
import type { ServerStatus } from "../hooks/useServerHealth";

interface AgentHeaderProps {
  serverStatus: ServerStatus;
  latencyMs: number | null;
  apiUrl: string;
  running: boolean;
  isScanning: boolean;
  uiLang: "ar" | "en";
  onUiLangChange: (lang: "ar" | "en") => void;
}

export function AgentHeader({
  serverStatus,
  latencyMs,
  running,
  isScanning,
  uiLang,
  onUiLangChange,
}: AgentHeaderProps) {
  const isAr = uiLang === "ar";

  return (
    <header className="site-header">
      <div className="topbar">
        <div className="topbar-brand">
          <span className="brand-mark" aria-hidden>
            🐪
          </span>
          <div className="topbar-titles">
            <strong className="brand-name">{isAr ? "وثيقة بصيرة" : "Wathiqa Basira"}</strong>
            <span className="brand-product">
              {isAr ? "ذكاء المستندات العربية" : "Arabic Document Intelligence"}
            </span>
          </div>
        </div>

        <div className="topbar-actions">
          <div className={`status-pill ${serverStatus}`}>
            <Radio size={14} />
            <span>
              {serverStatus === "online"
                ? isAr
                  ? "جاهز"
                  : "Ready"
                : serverStatus === "offline"
                  ? isAr
                    ? "غير متصل"
                    : "Offline"
                  : isAr
                    ? "…"
                    : "…"}
              {serverStatus === "online" && latencyMs != null ? ` · ${latencyMs}ms` : ""}
            </span>
          </div>
          {running && (
            <div className="status-pill running">
              <ScanText size={14} />
              <span>{isScanning ? (isAr ? "جاري المسح…" : "Scanning…") : isAr ? "يعمل…" : "Working…"}</span>
            </div>
          )}
          <div className="lang-toggle" role="group" aria-label={isAr ? "لغة الواجهة" : "UI language"}>
            <button
              type="button"
              className={uiLang === "ar" ? "active" : ""}
              onClick={() => onUiLangChange("ar")}
            >
              ع
            </button>
            <button
              type="button"
              className={uiLang === "en" ? "active" : ""}
              onClick={() => onUiLangChange("en")}
            >
              EN
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

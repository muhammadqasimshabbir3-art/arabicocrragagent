import { Github, Linkedin, Radio, ScanText, Trophy } from "lucide-react";
import type { ServerStatus } from "../hooks/useServerHealth";
import { AUTHOR } from "../lib/authorLinks";

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
          <nav className="header-social" aria-label={isAr ? "ملفات المؤلف" : "Author profiles"}>
            <a
              href={AUTHOR.github}
              target="_blank"
              rel="noopener noreferrer"
              title="GitHub"
              aria-label="GitHub"
            >
              <Github size={16} />
            </a>
            <a
              href={AUTHOR.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              title="LinkedIn"
              aria-label="LinkedIn"
            >
              <Linkedin size={16} />
            </a>
            <a
              href={AUTHOR.zindi.profile}
              target="_blank"
              rel="noopener noreferrer"
              title={`Zindi · #${AUTHOR.zindi.rank}`}
              aria-label={`Zindi rank ${AUTHOR.zindi.rank}`}
            >
              <Trophy size={16} />
            </a>
          </nav>
          <div
            className={`status-pill ${serverStatus}`}
            role="status"
            aria-live="polite"
          >
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
            <div className="status-pill running" role="status" aria-live="polite">
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

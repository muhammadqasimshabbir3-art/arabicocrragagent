import { useCallback, useEffect, useState } from "react";
import { AgentHeader } from "./components/AgentHeader";
import { AgentConfigForm } from "./components/AgentConfigForm";
import { ConnectionPanel } from "./components/ConnectionPanel";
import { LiveWorkBanner } from "./components/LiveWorkBanner";
import { StackInfoPanel } from "./components/StackInfoPanel";
import { WorkflowPipeline } from "./components/WorkflowPipeline";
import { WorkflowStrip } from "./components/WorkflowStrip";
import { useAgentRun } from "./hooks/useAgentRun";
import { useServerHealth } from "./hooks/useServerHealth";
import { loadRunSettings, saveRunSettings } from "./lib/settingsStorage";
import type { AgentRunSettings } from "./types";

type UiLang = "ar" | "en";

const LANG_KEY = "wathiqa-ui-lang";
const PIPELINE_OPEN_KEY = "wathiqa-pipeline-open";

function loadUiLang(): UiLang {
  try {
    const raw = localStorage.getItem(LANG_KEY);
    if (raw === "en" || raw === "ar") return raw;
  } catch {
    // ignore
  }
  return "ar";
}

export default function App() {
  const health = useServerHealth();
  const agent = useAgentRun();
  const [settings, setSettings] = useState<AgentRunSettings>(loadRunSettings);
  const [uiLang, setUiLang] = useState<UiLang>(loadUiLang);
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const isAr = uiLang === "ar";
  const isScanning = agent.steps.some(
    (step) => step.id === "ingest_document" && step.status === "running",
  );
  const hasAnswer = Boolean(agent.answer.trim());
  const workInProgress = agent.running && !hasAnswer;

  useEffect(() => {
    document.documentElement.lang = uiLang;
    document.documentElement.dir = uiLang === "ar" ? "rtl" : "ltr";
    document.title =
      uiLang === "ar"
        ? "وثيقة بصيرة — ارفع واسأل"
        : "Wathiqa Basira — Upload and ask";
    try {
      localStorage.setItem(LANG_KEY, uiLang);
    } catch {
      // ignore
    }
  }, [uiLang]);

  useEffect(() => {
    try {
      setPipelineOpen(localStorage.getItem(PIPELINE_OPEN_KEY) === "1");
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(PIPELINE_OPEN_KEY, pipelineOpen ? "1" : "0");
    } catch {
      // ignore
    }
  }, [pipelineOpen]);

  const updateSetting = useCallback(
    <K extends keyof AgentRunSettings>(key: K, value: AgentRunSettings[K]) => {
      setSettings((prev) => {
        const next = { ...prev, [key]: value };
        saveRunSettings(next);
        return next;
      });
    },
    [],
  );

  const startAgent = () => void agent.run(settings);

  const autoSummarizeUpload = useCallback(
    (overrides: Partial<AgentRunSettings>) => {
      const next = { ...settings, ...overrides };
      setSettings(next);
      saveRunSettings(next);
      void agent.run(next);
    },
    [agent, settings],
  );

  return (
    <div className="app-shell action-first">
      <AgentHeader
        serverStatus={health.status}
        latencyMs={health.latencyMs}
        apiUrl={health.apiUrl}
        running={workInProgress}
        isScanning={isScanning && workInProgress}
        uiLang={uiLang}
        onUiLangChange={setUiLang}
      />

      <p className="one-liner">
        {isAr
          ? "ارفع PDF أو صورة أو TXT أو DOCX فيُلخَّص تلقائيًا — أو اسأل من قاعدة المعرفة"
          : "Upload PDF, image, TXT, or DOCX for auto-summary — or ask the knowledge base"}
      </p>

      <WorkflowStrip
        steps={agent.steps}
        running={workInProgress}
        completed={hasAnswer}
        uiLang={uiLang}
      />
      <LiveWorkBanner steps={agent.steps} running={workInProgress} uiLang={uiLang} />

      <div className="desk-layout stacked">
        <AgentConfigForm
          settings={settings}
          onChange={updateSetting}
          running={agent.running}
          steps={agent.steps}
          serverOnline={health.status === "online"}
          onStart={startAgent}
          onStop={agent.cancel}
          onReset={agent.reset}
          onClearAnswer={agent.clearAnswer}
          onAutoSummarize={autoSummarizeUpload}
          canReset={Boolean(agent.answer) || agent.conversationMessages.length > 0}
          answer={agent.answer}
          error={agent.error}
          uiLang={uiLang}
        />

        <aside className="desk-rail desk-rail-below">
          <section className="panel pipeline-collapsible">
            <div className="pipeline-collapsible-head">
              <strong>{isAr ? "مسار المعالجة" : "Processing pipeline"}</strong>
              <button
                type="button"
                className="btn small ghost"
                onClick={() => setPipelineOpen((v) => !v)}
              >
                {pipelineOpen
                  ? isAr
                    ? "إخفاء"
                    : "Hide"
                  : isAr
                    ? "إظهار"
                    : "Show"}
              </button>
            </div>
            {pipelineOpen && (
              <WorkflowPipeline
                steps={agent.steps}
                running={agent.running}
                reconnected={agent.reconnected}
                taskPlanSummary={agent.result?.task_plan_summary}
                agentRoute={agent.result?.agent_route}
                uiLang={uiLang}
              />
            )}
            {!pipelineOpen && (
              <p className="pipeline-collapsible-note">
                {isAr
                  ? "مطوي افتراضيًا. افتحه عند الحاجة."
                  : "Collapsed by default. Open when needed."}
              </p>
            )}
          </section>
          <ConnectionPanel status={health.status} onRefresh={health.check} uiLang={uiLang} />
          <StackInfoPanel uiLang={uiLang} />
        </aside>
      </div>
    </div>
  );
}

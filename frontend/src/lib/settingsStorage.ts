import type { AgentRunSettings } from "../types";
import { defaultRunSettings } from "./defaultSettings";

const STORAGE_KEY = "wathiqa-ui-run-settings";
const LEGACY_STORAGE_KEY = "arabic-doc-agent-run-settings";

export function loadRunSettings(): AgentRunSettings {
  const defaults = defaultRunSettings();
  try {
    const raw =
      localStorage.getItem(STORAGE_KEY) ??
      sessionStorage.getItem(STORAGE_KEY) ??
      localStorage.getItem(LEGACY_STORAGE_KEY) ??
      sessionStorage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Partial<AgentRunSettings>;
    return { ...defaults, ...parsed };
  } catch {
    return defaults;
  }
}

export function saveRunSettings(settings: AgentRunSettings): void {
  // Document payloads can be large; keep them in React state for follow-up
  // questions during the current session instead of persisting to storage.
  const {
    pdf_data_base64: _pdfData,
    document_data_base64: _docData,
    ...storedSettings
  } = settings;
  const raw = JSON.stringify(storedSettings);
  try {
    localStorage.setItem(STORAGE_KEY, raw);
    sessionStorage.setItem(STORAGE_KEY, raw);
  } catch {
    // ignore quota / private mode
  }
}

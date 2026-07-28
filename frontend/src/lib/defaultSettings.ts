import type { AgentRunSettings } from "../types";

function envStr(key: string, fallback: string): string {
  const raw = import.meta.env[key];
  return raw != null && String(raw).trim() !== "" ? String(raw).trim() : fallback;
}

export function defaultRunSettings(): AgentRunSettings {
  return {
    user_input: envStr("VITE_DEFAULT_USER_INPUT", "ما موضوع هذا المستند؟"),
    pdf_analysis_enabled: false,
    pdf_data_base64: "",
    pdf_filename: "",
    pdf_summarize_only: false,
    document_data_base64: "",
    document_filename: "",
    document_mime_type: "",
    summarize_only: false,
    response_language: "ar",
  };
}

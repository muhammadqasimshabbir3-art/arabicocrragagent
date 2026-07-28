export type StepStatus = "pending" | "running" | "completed" | "skipped" | "error";

export interface WorkflowStep {
  id: string;
  nodes: string[];
  label: string;
  description: string;
  optional?: boolean;
  emoji?: string;
}

export interface StepState {
  id: string;
  status: StepStatus;
  startedAt?: string;
  completedAt?: string;
  detail?: string;
}

export interface AgentState {
  task_plan_summary?: string;
  agent_route?: string;
  detected_language?: string;
  document_fingerprint?: string;
  indexed_chunk_count?: number;
  messages?: Array<{
    content?: unknown;
    type?: string;
    tool_calls?: Array<{ name?: string; args?: unknown }>;
  }>;
  user_input?: string;
  pdf_filename?: string;
  pdf_summarize_only?: boolean;
  document_filename?: string;
  document_mime_type?: string;
  summarize_only?: boolean;
}

export interface LogEntry {
  id: string;
  time: string;
  level: "info" | "success" | "warn" | "error";
  message: string;
}

export interface RunRequest {
  user_input: string;
  pdf_analysis_enabled?: boolean;
  pdf_data_base64?: string;
  pdf_filename?: string;
  pdf_summarize_only?: boolean;
  document_data_base64?: string;
  document_filename?: string;
  document_mime_type?: string;
  summarize_only?: boolean;
  /** Preferred answer language — independent of document / UI language. */
  response_language?: "ar" | "en";
  conversation_messages?: Array<{ type: "human" | "ai"; content: string }>;
}

export type AgentRunSettings = RunRequest;

export const DEFAULT_AGENT_TEMPERATURE = 0.7;
export const DEFAULT_AGENT_MAX_ITERATIONS = 20;
export const DEFAULT_MAX_CONTEXT_TOKENS = 200_000;

export type Role = "system" | "user" | "assistant" | "tool";

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
}

export interface FunctionCall {
  name: string;
  arguments: string;
}

export interface LLMToolCall {
  id: string;
  type: "function" | string;
  function: FunctionCall;
  provider_metadata?: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  type: "function" | string;
  function: FunctionCall;
  provider_metadata?: Record<string, unknown>;
}

export interface MessageContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: {
    url: string;
    detail?: "auto" | "low" | "high";
  };
}

export interface Message {
  role: Role;
  content: string;
  multi_content?: MessageContentPart[];
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
  images?: string[];
  reasoning_content?: string;
}

export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: unknown;
}

export interface ChatTool {
  type: "function";
  function: FunctionDefinition;
}

export interface ChatOptions {
  temperature?: number;
  top_p?: number;
  seed?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  thinking?: boolean;
  tools?: ChatTool[];
  tool_choice?: "auto" | "required" | "none" | string;
  parallel_tool_calls?: boolean;
  format?: unknown;
}

export interface StreamResponse {
  id?: string;
  response_type:
    | "answer"
    | "references"
    | "thinking"
    | "tool_call"
    | "tool_result"
    | "error"
    | "reflection"
    | "session_title"
    | "agent_query"
    | "complete";
  content: string;
  done: boolean;
  tool_calls?: LLMToolCall[];
  data?: Record<string, unknown>;
  usage?: TokenUsage;
  finish_reason?: string;
}

export interface ChatResponse {
  content: string;
  reasoning_content?: string;
  tool_calls?: LLMToolCall[];
  finish_reason?: string;
  usage?: TokenUsage;
  answer_streamed?: boolean;
  answer_event_id?: string;
}

export interface ChatClient {
  chatStream(messages: Message[], opts: ChatOptions): AsyncIterable<StreamResponse>;
  getModelName(): string;
  getModelID(): string;
}

export interface ToolResult {
  success: boolean;
  output: string;
  data?: Record<string, unknown>;
  error?: string;
  images?: string[];
}

export interface RuntimeTool {
  name(): string;
  description(): string;
  parameters(): unknown;
  execute(args: unknown, signal?: AbortSignal): Promise<ToolResult>;
  cleanup?(): Promise<void> | void;
}

export interface ExecutedToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result: ToolResult;
  reflection?: string;
  duration: number;
  provider_metadata?: Record<string, unknown>;
}

export interface AgentStep {
  iteration: number;
  thought: string;
  reasoning_content?: string;
  tool_calls: ExecutedToolCall[];
  timestamp: string;
}

export interface SearchResult {
  id?: string;
  content?: string;
  knowledge_id?: string;
  knowledge_base_id?: string;
  knowledge_title?: string;
  score?: number;
  [key: string]: unknown;
}

export interface AgentState {
  current_round: number;
  round_steps: AgentStep[];
  is_complete: boolean;
  final_answer: string;
  knowledge_refs: SearchResult[];
}

export interface RecentDocInfo {
  chunk_id?: string;
  knowledge_base_id?: string;
  knowledge_id: string;
  title?: string;
  description?: string;
  file_name?: string;
  file_size?: number;
  type?: string;
  created_at?: string;
  faq_standard_question?: string;
  faq_similar_questions?: string[];
  faq_answers?: string[];
}

export interface SelectedDocumentInfo {
  knowledge_id: string;
  knowledge_base_id?: string;
  title?: string;
  file_name?: string;
  file_type?: string;
}

export interface KnowledgeBaseInfo {
  id: string;
  name: string;
  type?: string;
  description?: string;
  doc_count?: number;
  capabilities?: string[];
  recent_docs?: RecentDocInfo[];
}

export interface AgentConfig {
  max_iterations: number;
  allowed_tools: string[];
  temperature: number;
  knowledge_bases: string[];
  knowledge_ids: string[];
  system_prompt?: string;
  use_custom_system_prompt: boolean;
  multi_turn_enabled: boolean;
  history_turns: number;
  thinking?: boolean;
  retrieve_kb_only_when_mentioned?: boolean;
  retain_retrieval_history?: boolean;
  skills_enabled?: boolean;
  skill_dirs?: string[];
  allowed_skills?: string[];
  llm_call_timeout?: number;
  max_tool_output_chars?: number;
  max_context_tokens?: number;
  parallel_tool_calls?: boolean;
}

export interface AgentContext {
  session_id: string;
  message_id?: string;
  language?: string;
  knowledge_bases: KnowledgeBaseInfo[];
  selected_documents: SelectedDocumentInfo[];
  history: Message[];
}

export interface AgentEventMap {
  thought: { content: string; iteration: number; done: boolean };
  answer: { content: string; done: boolean };
  tool_call: { tool_call_id: string; tool_name: string; arguments?: Record<string, unknown>; iteration: number };
  tool_result: {
    tool_call_id: string;
    tool_name: string;
    output: string;
    error?: string;
    success: boolean;
    duration: number;
    iteration: number;
    data?: Record<string, unknown>;
  };
  complete: { state: AgentState; duration_ms: number };
  error: { error: string; stage: string };
  status: { message: string };
}

export type AgentEventName = keyof AgentEventMap;

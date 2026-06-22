import type {
  ChatClient,
  ChatOptions,
  LLMToolCall,
  Message,
  MessageContentPart,
  StreamResponse,
  TokenUsage,
  ToolCall,
} from "./types.js";

interface OpenAIChoiceDelta {
  role?: string;
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: {
      name?: string;
      arguments?: string;
    };
  }>;
}

interface OpenAIStreamChunk {
  id?: string;
  choices?: Array<{
    delta?: OpenAIChoiceDelta;
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    prompt_tokens_details?: {
      cached_tokens?: number;
    };
  };
}

export interface OpenAICompatibleChatConfig {
  apiKey: string;
  baseURL: string;
  model: string;
  modelID?: string;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

function normalizeBaseURL(baseURL: string): string {
  const trimmed = baseURL.replace(/\/+$/, "");
  if (trimmed.endsWith("/v1")) return trimmed;
  return `${trimmed}/v1`;
}

function toAPIMessage(message: Message): Record<string, unknown> {
  const base: Record<string, unknown> = {
    role: message.role,
  };
  if (message.name) base.name = message.name;
  if (message.tool_call_id) base.tool_call_id = message.tool_call_id;
  if (message.reasoning_content) base.reasoning_content = message.reasoning_content;

  const toolCalls = message.tool_calls;
  if (toolCalls && toolCalls.length > 0) {
    base.tool_calls = toolCalls.map((tc) => ({
      id: tc.id,
      type: tc.type,
      function: {
        name: tc.function.name,
        arguments: tc.function.arguments,
      },
    }));
  }

  const parts: MessageContentPart[] = [];
  if (message.content) {
    parts.push({ type: "text", text: message.content });
  }
  for (const image of message.images ?? []) {
    parts.push({ type: "image_url", image_url: { url: image } });
  }
  for (const part of message.multi_content ?? []) {
    parts.push(part);
  }

  if (parts.length > 1 || message.images?.length) {
    base.content = parts.map((part) => {
      if (part.type === "text") return { type: "text", text: part.text ?? "" };
      return { type: "image_url", image_url: part.image_url };
    });
  } else {
    base.content = message.content;
  }

  return base;
}

function parseUsage(raw: OpenAIStreamChunk["usage"]): TokenUsage | undefined {
  if (!raw) return undefined;
  return {
    prompt_tokens: raw.prompt_tokens ?? 0,
    completion_tokens: raw.completion_tokens ?? 0,
    total_tokens: raw.total_tokens ?? 0,
    cached_tokens: raw.prompt_tokens_details?.cached_tokens,
  };
}

function sseLines(buffer: string): { events: string[]; rest: string } {
  const events: string[] = [];
  let searchStart = 0;
  for (;;) {
    const idx = buffer.indexOf("\n\n", searchStart);
    if (idx < 0) break;
    events.push(buffer.slice(0, idx));
    buffer = buffer.slice(idx + 2);
    searchStart = 0;
  }
  return { events, rest: buffer };
}

function decodeSSEEvent(eventText: string): string[] {
  return eventText
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
}

function mergeToolCallDelta(target: Map<number, LLMToolCall>, delta: NonNullable<OpenAIChoiceDelta["tool_calls"]>[number]): void {
  const current =
    target.get(delta.index) ??
    ({
      id: delta.id ?? "",
      type: delta.type ?? "function",
      function: { name: "", arguments: "" },
    } satisfies LLMToolCall);

  if (delta.id) current.id = delta.id;
  if (delta.type) current.type = delta.type;
  if (delta.function?.name) current.function.name += delta.function.name;
  if (delta.function?.arguments) current.function.arguments += delta.function.arguments;
  target.set(delta.index, current);
}

export class OpenAICompatibleChatClient implements ChatClient {
  private readonly apiKey: string;
  private readonly baseURL: string;
  private readonly model: string;
  private readonly modelID: string;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;

  constructor(config: OpenAICompatibleChatConfig) {
    this.apiKey = config.apiKey;
    this.baseURL = normalizeBaseURL(config.baseURL);
    this.model = config.model;
    this.modelID = config.modelID ?? config.model;
    this.timeoutMs = config.timeoutMs ?? 120_000;
    this.headers = config.headers ?? {};
  }

  getModelName(): string {
    return this.model;
  }

  getModelID(): string {
    return this.modelID;
  }

  async *chatStream(messages: Message[], opts: ChatOptions): AsyncIterable<StreamResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error("LLM call timed out")), this.timeoutMs);

    const body: Record<string, unknown> = {
      model: this.model,
      messages: messages.map(toAPIMessage),
      temperature: opts.temperature,
      stream: true,
      stream_options: { include_usage: true },
    };
    if (opts.tools && opts.tools.length > 0) body.tools = opts.tools;
    if (opts.tool_choice) body.tool_choice = opts.tool_choice;
    if (opts.parallel_tool_calls !== undefined) body.parallel_tool_calls = opts.parallel_tool_calls;
    if (opts.max_tokens) body.max_tokens = opts.max_tokens;
    if (opts.max_completion_tokens) body.max_completion_tokens = opts.max_completion_tokens;
    if (opts.top_p) body.top_p = opts.top_p;
    if (opts.frequency_penalty) body.frequency_penalty = opts.frequency_penalty;
    if (opts.presence_penalty) body.presence_penalty = opts.presence_penalty;
    if (opts.format) body.response_format = opts.format;

    let response: Response;
    try {
      response = await fetch(`${this.baseURL}/chat/completions`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
          ...this.headers,
        },
        body: JSON.stringify(body),
      });
    } catch (error) {
      clearTimeout(timer);
      const message = error instanceof Error ? error.message : String(error);
      yield { response_type: "error", content: message, done: true };
      return;
    }

    if (!response.ok || !response.body) {
      clearTimeout(timer);
      const text = await response.text().catch(() => "");
      yield {
        response_type: "error",
        content: `LLM request failed: HTTP ${response.status}${text ? ` ${text}` : ""}`,
        done: true,
      };
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const toolCalls = new Map<number, LLMToolCall>();

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = sseLines(buffer);
        buffer = parsed.rest;

        for (const eventText of parsed.events) {
          for (const data of decodeSSEEvent(eventText)) {
            if (!data || data === "[DONE]") continue;
            let chunk: OpenAIStreamChunk;
            try {
              chunk = JSON.parse(data) as OpenAIStreamChunk;
            } catch {
              continue;
            }

            const usage = parseUsage(chunk.usage);
            if (usage) {
              yield { response_type: "answer", content: "", done: false, usage };
            }

            const choice = chunk.choices?.[0];
            const delta = choice?.delta;
            if (delta?.reasoning_content) {
              yield {
                id: chunk.id,
                response_type: "thinking",
                content: delta.reasoning_content,
                done: false,
                finish_reason: choice?.finish_reason ?? undefined,
              };
            }
            if (delta?.content) {
              yield {
                id: chunk.id,
                response_type: "answer",
                content: delta.content,
                done: false,
                finish_reason: choice?.finish_reason ?? undefined,
              };
            }
            if (delta?.tool_calls?.length) {
              for (const tc of delta.tool_calls) mergeToolCallDelta(toolCalls, tc);
              yield {
                id: chunk.id,
                response_type: "tool_call",
                content: "",
                done: false,
                tool_calls: [...toolCalls.entries()]
                  .sort(([a], [b]) => a - b)
                  .map(([, tc]) => tc),
                data: {
                  tool_call_id: delta.tool_calls[0]?.id,
                  tool_name: delta.tool_calls[0]?.function?.name,
                },
                finish_reason: choice?.finish_reason ?? undefined,
              };
            }

            if (choice?.finish_reason) {
              yield {
                id: chunk.id,
                response_type: "answer",
                content: "",
                done: true,
                tool_calls: [...toolCalls.entries()]
                  .sort(([a], [b]) => a - b)
                  .map(([, tc]) => tc),
                usage,
                finish_reason: choice.finish_reason,
              };
            }
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      yield { response_type: "error", content: message, done: true };
    } finally {
      clearTimeout(timer);
      reader.releaseLock();
    }
  }
}

export function toAssistantMessage(content: string, toolCalls?: ToolCall[], reasoningContent?: string): Message {
  return {
    role: "assistant",
    content,
    tool_calls: toolCalls,
    reasoning_content: reasoningContent,
  };
}

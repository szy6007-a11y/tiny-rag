import { AgentEventBus } from "./events.js";
import { buildRuntimeContextBlock, buildSystemPrompt } from "./prompt.js";
import { ToolRegistry } from "./tool-registry.js";
import type {
  AgentConfig,
  AgentContext,
  AgentState,
  AgentStep,
  ChatClient,
  ChatResponse,
  ChatTool,
  ExecutedToolCall,
  LLMToolCall,
  Message,
  StreamResponse,
  TokenUsage,
} from "./types.js";
import {
  compressContext,
  estimateMessagesTokens,
  generateEventID,
  isTransientError,
  parseJSONRecord,
  sanitizeMessages,
  sleep,
  stripThinkBlocks,
  ThinkStreamSplitter,
} from "./utils.js";

const DEFAULT_LLM_CALL_TIMEOUT_MS = 120_000;
const MAX_LLM_RETRIES = 2;
const MAX_EMPTY_RESPONSE_RETRIES = 2;
const MAX_REPEATED_RESPONSE_ROUNDS = 2;

interface StreamLLMResult {
  content: string;
  reasoningContent: string;
  toolCalls: LLMToolCall[];
  usage?: TokenUsage;
  finishReason: string;
  streamError: string;
}

type IterOutcome = "next" | "continue" | "break";

export class AgentEngine {
  private lastUsage?: TokenUsage;
  private lastSentMsgCount = 0;

  constructor(
    private readonly config: AgentConfig,
    private readonly chatModel: ChatClient,
    private readonly toolRegistry: ToolRegistry,
    private readonly eventBus: AgentEventBus,
    private readonly context: AgentContext,
  ) {}

  private getLLMCallTimeout(): number {
    return this.config.llm_call_timeout && this.config.llm_call_timeout > 0
      ? this.config.llm_call_timeout * 1000
      : DEFAULT_LLM_CALL_TIMEOUT_MS;
  }

  private estimateCurrentTokens(messages: Message[]): number {
    if (
      this.lastUsage?.total_tokens &&
      this.lastSentMsgCount > 0 &&
      this.lastSentMsgCount < messages.length
    ) {
      return this.lastUsage.total_tokens + estimateMessagesTokens(messages.slice(this.lastSentMsgCount));
    }
    return estimateMessagesTokens(messages);
  }

  private manageContextWindow(messages: Message[], currentTokens: number): Message[] {
    const maxTokens = this.config.max_context_tokens ?? 0;
    if (maxTokens <= 0) return messages;
    return compressContext(messages, maxTokens, currentTokens);
  }

  private buildMessagesWithLLMContext(systemPrompt: string, currentQuery: string, imageURLs: string[] = []): Message[] {
    const messages: Message[] = [{ role: "system", content: systemPrompt }];

    const llmContext = this.context.history ?? [];
    for (const msg of llmContext) {
      if (msg.role === "system") continue;
      if (msg.role === "user" || msg.role === "assistant" || msg.role === "tool") {
        messages.push(msg);
      }
    }

    const runtimeContext = buildRuntimeContextBlock(
      this.context.session_id,
      this.context.knowledge_bases,
      this.context.selected_documents,
    );
    messages.push({
      role: "user",
      content: `${runtimeContext}\n\n${currentQuery}`,
      images: imageURLs,
    });

    return messages;
  }

  private buildToolsForLLM(): ChatTool[] {
    return this.toolRegistry.getFunctionDefinitions();
  }

  private async streamLLMToEventBus(
    messages: Message[],
    tools: ChatTool[],
    emitFunc: (chunk: StreamResponse, fullContent: string) => void,
  ): Promise<StreamLLMResult> {
    const result: StreamLLMResult = {
      content: "",
      reasoningContent: "",
      toolCalls: [],
      finishReason: "",
      streamError: "",
    };

    const opts = {
      temperature: this.config.temperature,
      tools,
      thinking: this.config.thinking,
      parallel_tool_calls: true,
    };

    for await (const chunk of this.chatModel.chatStream(messages, opts)) {
      if (chunk.response_type === "error") {
        result.streamError = chunk.content;
        continue;
      }

      if (chunk.content) {
        const source = chunk.data?.source;
        if (!source) {
          if (chunk.response_type === "thinking") {
            result.reasoningContent += chunk.content;
          } else {
            result.content += chunk.content;
          }
        }
      }

      if (chunk.tool_calls?.length) {
        result.toolCalls = chunk.tool_calls;
      }
      if (chunk.usage) {
        result.usage = chunk.usage;
      }
      if (chunk.finish_reason) {
        result.finishReason = chunk.finish_reason;
      }

      emitFunc(chunk, result.content);
    }

    if (result.streamError && result.content === "" && result.toolCalls.length === 0) {
      throw new Error(`LLM stream error: ${result.streamError}`);
    }

    return result;
  }

  private async streamThinkingToEventBus(
    messages: Message[],
    tools: ChatTool[],
    iteration: number,
  ): Promise<ChatResponse> {
    const thinkingID = generateEventID("thinking");
    const answerID = generateEventID("answer");
    const splitter = new ThinkStreamSplitter();
    let thinkingOpen = false;
    let answerStreamed = false;

    const emitThought = (content: string, done: boolean) => {
      if (!content && !done) return;
      this.eventBus.emit("thought", { content, iteration, done });
    };

    const closeThinking = () => {
      if (thinkingOpen) {
        emitThought("", true);
        thinkingOpen = false;
      }
    };

    const emitAnswer = (content: string) => {
      if (!content) return;
      if (!answerStreamed && content.trim() === "") return;
      closeThinking();
      answerStreamed = true;
      this.eventBus.emit("answer", { content, done: false });
    };

    const llmResult = await this.streamLLMToEventBus(messages, tools, (chunk) => {
      if (chunk.response_type === "tool_call" && chunk.data) {
        const toolCallID = String(chunk.data.tool_call_id ?? "");
        const toolName = String(chunk.data.tool_name ?? "");
        if (toolCallID && toolName) {
          this.eventBus.emit("tool_call", {
            tool_call_id: toolCallID,
            tool_name: toolName,
            iteration,
          });
        }
      }

      if (chunk.response_type === "thinking") {
        if (chunk.content) {
          thinkingOpen = true;
          emitThought(chunk.content, false);
        } else if (chunk.done && thinkingOpen) {
          closeThinking();
        }
        return;
      }

      if (chunk.content) {
        const { think, answer } = splitter.feed(chunk.content);
        if (think) {
          thinkingOpen = true;
          emitThought(think, false);
        }
        emitAnswer(answer);
      }

      if (chunk.done) {
        const { think, answer } = splitter.flush();
        if (think) {
          thinkingOpen = true;
          emitThought(think, false);
        }
        emitAnswer(answer);
        closeThinking();
      }
    });

    const finishReason = llmResult.finishReason || "stop";
    const response: ChatResponse = {
      content: stripThinkBlocks(llmResult.content),
      reasoning_content: llmResult.reasoningContent,
      tool_calls: llmResult.toolCalls,
      finish_reason: finishReason,
      usage: llmResult.usage,
      answer_streamed: answerStreamed,
    };
    if (answerStreamed) response.answer_event_id = answerID || thinkingID;
    return response;
  }

  private countTotalToolCalls(steps: AgentStep[]): number {
    return steps.reduce((total, step) => total + step.tool_calls.length, 0);
  }

  private analyzeResponse(response: ChatResponse, step: AgentStep, iteration: number): {
    isDone: boolean;
    finalAnswer: string;
    emptyContent: boolean;
    step: AgentStep;
  } {
    if (response.finish_reason === "content_filter" && (response.tool_calls?.length ?? 0) === 0) {
      const answer =
        response.content ||
        "Sorry, this request was blocked by the content safety policy. Please try rephrasing your question.";
      this.eventBus.emit("answer", { content: answer, done: false });
      this.eventBus.emit("answer", { content: "", done: true });
      return { isDone: true, finalAnswer: answer, emptyContent: false, step };
    }

    if (response.finish_reason === "stop" && (response.tool_calls?.length ?? 0) === 0) {
      const answer = stripThinkBlocks(response.content);
      if (!response.answer_streamed && answer) {
        this.eventBus.emit("answer", { content: answer, done: false });
      }
      this.eventBus.emit("answer", { content: "", done: true });
      return { isDone: true, finalAnswer: answer, emptyContent: answer === "", step };
    }

    return { isDone: false, finalAnswer: "", emptyContent: false, step };
  }

  private async callLLMWithRetry(
    messages: Message[],
    tools: ChatTool[],
    state: AgentState,
    query: string,
    iteration: number,
  ): Promise<ChatResponse | null> {
    let sanitized = sanitizeMessages(messages);
    let response: ChatResponse | null = null;
    let lastError: unknown;

    for (let attempt = 0; attempt <= MAX_LLM_RETRIES; attempt++) {
      try {
        response = await this.streamThinkingToEventBus(sanitized, tools, iteration);
        return response;
      } catch (error) {
        lastError = error;
        if (!isTransientError(error) || attempt === MAX_LLM_RETRIES) break;
        await sleep((attempt + 1) * 1000);
        sanitized = sanitizeMessages(messages);
      }
    }

    if (this.countTotalToolCalls(state.round_steps) > 0) {
      await this.streamFinalAnswerToEventBus(query, state);
      state.is_complete = true;
      return null;
    }

    const message = lastError instanceof Error ? lastError.message : String(lastError);
    throw new Error(`LLM call failed: ${message}`);
  }

  private async executeToolCalls(response: ChatResponse, step: AgentStep, iteration: number): Promise<void> {
    const calls = response.tool_calls ?? [];
    if (calls.length === 0) return;

    const run = async (tc: LLMToolCall, index: number): Promise<ExecutedToolCall> => {
      const id = tc.id || `${tc.function.name}-${index}`;
      let args: Record<string, unknown>;
      try {
        args = parseJSONRecord(tc.function.arguments || "{}");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return {
          id,
          name: tc.function.name,
          args: { _raw: tc.function.arguments },
          duration: 0,
          provider_metadata: tc.provider_metadata,
          result: {
            success: false,
            output: "",
            error: `Failed to parse tool arguments: ${message}\n\n[Analyze the error above and try a different approach.]`,
          },
        };
      }

      this.eventBus.emit("tool_call", {
        tool_call_id: id,
        tool_name: tc.function.name,
        arguments: args,
        iteration,
      });

      const started = Date.now();
      const result = await this.toolRegistry.executeTool(tc.function.name, args);
      const duration = Date.now() - started;

      this.eventBus.emit("tool_result", {
        tool_call_id: id,
        tool_name: tc.function.name,
        output: result.output,
        error: result.error,
        success: result.success,
        duration,
        iteration,
        data: result.data,
      });

      return {
        id,
        name: tc.function.name,
        args,
        result,
        duration,
        provider_metadata: tc.provider_metadata,
      };
    };

    if (this.config.parallel_tool_calls && calls.length >= 2) {
      step.tool_calls.push(...(await Promise.all(calls.map(run))));
      return;
    }

    for (let i = 0; i < calls.length; i++) {
      step.tool_calls.push(await run(calls[i], i));
    }
  }

  private appendToolResults(messages: Message[], step: AgentStep): Message[] {
    if (step.thought || step.tool_calls.length > 0 || step.reasoning_content) {
      const assistantMsg: Message = {
        role: "assistant",
        content: step.thought,
        reasoning_content: step.reasoning_content,
      };
      if (step.tool_calls.length > 0) {
        assistantMsg.tool_calls = step.tool_calls.map((tc) => ({
          id: tc.id,
          type: "function",
          provider_metadata: tc.provider_metadata,
          function: {
            name: tc.name,
            arguments: JSON.stringify(tc.args),
          },
        }));
      }
      messages.push(assistantMsg);
    }

    for (const toolCall of step.tool_calls) {
      const resultContent = toolCall.result.success
        ? toolCall.result.output
        : `Error: ${toolCall.result.error ?? ""}`;
      messages.push({
        role: "tool",
        content: resultContent,
        tool_call_id: toolCall.id,
        name: toolCall.name,
      });
    }

    return messages;
  }

  private async streamFinalAnswerToEventBus(query: string, state: AgentState): Promise<void> {
    const systemPrompt = buildSystemPrompt({
      knowledgeBases: this.context.knowledge_bases,
      selectedDocs: this.context.selected_documents,
      language: this.context.language,
      systemPromptTemplate: this.config.system_prompt,
    });

    const messages: Message[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: query },
    ];

    for (const step of state.round_steps) {
      for (const toolCall of step.tool_calls) {
        messages.push({
          role: "user",
          content: `Tool ${toolCall.name} returned: ${toolCall.result.output}`,
        });
      }
    }

    messages.push({
      role: "user",
      content: `Based on the above tool call results, generate a complete answer for the user's question.

User question: ${query}

Requirements:
1. Answer based on the actually retrieved content
2. Clearly cite information sources (chunk_id, document name)
3. Organize the answer in a structured format
4. If information is insufficient, honestly state so
5. IMPORTANT: Respond in the same language as the user's question

Now generate the final answer:`,
    });

    let fullAnswer = "";
    const result = await this.streamLLMToEventBus(messages, [], (chunk) => {
      if (chunk.response_type === "thinking") return;
      if (chunk.content) {
        fullAnswer += chunk.content;
        this.eventBus.emit("answer", { content: chunk.content, done: false });
      }
    });
    if (!result.finishReason) {
      this.eventBus.emit("answer", { content: "", done: true });
    }
    state.final_answer = stripThinkBlocks(result.content || fullAnswer);
  }

  private async handleMaxIterations(query: string, state: AgentState): Promise<void> {
    await this.streamFinalAnswerToEventBus(query, state).catch(() => {
      state.final_answer = "Sorry, I was unable to generate a complete answer.";
    });
    state.is_complete = true;
  }

  private async runReActIteration(params: {
    state: AgentState;
    messages: Message[];
    tools: ChatTool[];
    query: string;
    emptyRetries: { value: number };
    consecutiveSameContent: { value: number };
    lastResponseContent: { value: string };
  }): Promise<{ outcome: IterOutcome; messages: Message[] }> {
    const { state, tools, query, emptyRetries, consecutiveSameContent, lastResponseContent } = params;
    let messages = params.messages;
    const round = state.current_round + 1;

    let currentTokens = this.estimateCurrentTokens(messages);
    const beforeLen = messages.length;
    messages = this.manageContextWindow(messages, currentTokens);
    if (messages.length < beforeLen) {
      currentTokens = estimateMessagesTokens(messages);
    }

    this.eventBus.emit("status", {
      message: `Round ${round}/${this.config.max_iterations}, messages=${messages.length}, tools=${tools.length}, est_tokens=${currentTokens}`,
    });

    this.lastSentMsgCount = messages.length;
    const response = await this.callLLMWithRetry(messages, tools, state, query, state.current_round);
    if (!response) return { outcome: "break", messages };

    if (response.usage?.total_tokens) this.lastUsage = response.usage;

    if ((response.tool_calls?.length ?? 0) === 0 && response.content) {
      if (response.content === lastResponseContent.value) {
        consecutiveSameContent.value++;
      } else {
        consecutiveSameContent.value = 0;
      }
      lastResponseContent.value = response.content;
      if (consecutiveSameContent.value >= MAX_REPEATED_RESPONSE_ROUNDS) {
        state.final_answer = response.content;
        state.is_complete = true;
        return { outcome: "break", messages };
      }
    } else {
      consecutiveSameContent.value = 0;
      lastResponseContent.value = "";
    }

    const step: AgentStep = {
      iteration: state.current_round,
      thought: response.content,
      reasoning_content: response.reasoning_content,
      tool_calls: [],
      timestamp: new Date().toISOString(),
    };

    const verdict = this.analyzeResponse(response, step, state.current_round);
    if (verdict.isDone) {
      if (verdict.emptyContent) {
        emptyRetries.value++;
        if (emptyRetries.value <= MAX_EMPTY_RESPONSE_RETRIES) {
          messages.push({
            role: "user",
            content: "Please provide your complete answer now as plain text.",
          });
          return { outcome: "continue", messages };
        }
        state.final_answer = "I'm sorry, I was unable to generate a response. Please try again.";
        state.is_complete = true;
        state.round_steps.push(verdict.step);
        return { outcome: "break", messages };
      }
      state.final_answer = verdict.finalAnswer;
      state.is_complete = true;
      state.round_steps.push(verdict.step);
      return { outcome: "break", messages };
    }

    await this.executeToolCalls(response, step, state.current_round);
    state.round_steps.push(step);
    messages = this.appendToolResults(messages, step);
    return { outcome: "next", messages };
  }

  async execute(query: string, imageURLs: string[] = []): Promise<AgentState> {
    const started = Date.now();
    const state: AgentState = {
      current_round: 0,
      round_steps: [],
      is_complete: false,
      final_answer: "",
      knowledge_refs: [],
    };

    try {
      const systemPrompt = buildSystemPrompt({
        knowledgeBases: this.context.knowledge_bases,
        selectedDocs: this.context.selected_documents,
        language: this.context.language,
        systemPromptTemplate: this.config.system_prompt,
      });
      let messages = this.buildMessagesWithLLMContext(systemPrompt, query, imageURLs);
      const tools = this.buildToolsForLLM();

      const emptyRetries = { value: 0 };
      const consecutiveSameContent = { value: 0 };
      const lastResponseContent = { value: "" };

      while (state.current_round < this.config.max_iterations) {
        const result = await this.runReActIteration({
          state,
          messages,
          tools,
          query,
          emptyRetries,
          consecutiveSameContent,
          lastResponseContent,
        });
        messages = result.messages;
        if (result.outcome === "continue") continue;
        if (result.outcome === "break") break;
        state.current_round++;
      }

      if (!state.is_complete) {
        await this.handleMaxIterations(query, state);
      }

      return state;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.eventBus.emit("error", { error: message, stage: "agent_execution" });
      throw error;
    } finally {
      await this.toolRegistry.cleanup();
      this.eventBus.emit("complete", { state, duration_ms: Date.now() - started });
    }
  }
}

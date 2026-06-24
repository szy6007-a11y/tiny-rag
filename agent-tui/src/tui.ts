import { StringDecoder } from "node:string_decoder";
import { AgentEngine } from "./engine.js";
import { AgentEventBus } from "./events.js";
import { appendHistory, clearHistory, loadHistory } from "./history.js";
import { OpenAICompatibleChatClient } from "./chat-client.js";
import { registerRagTools } from "./rag-tools.js";
import { ToolRegistry } from "./tool-registry.js";
import type { AgentContext, Message } from "./types.js";
import type { RuntimeConfig } from "./config.js";

type TranscriptRole = "user" | "assistant" | "thought" | "status" | "error" | "system";

interface TranscriptItem {
  role: TranscriptRole;
  content: string;
}

interface TranscriptViewport {
  lines: string[];
  totalLines: number;
  scrollTop: number;
  maxScrollTop: number;
}

interface TranscriptRender {
  lines: string[];
  itemStartLines: number[];
}

interface TerminalLayout {
  width: number;
  height: number;
  bodyHeight: number;
}

const ESC = "\x1b[";

function enterAlternateScreen(): void {
  process.stdout.write(`${ESC}?1049h${ESC}?25l${ESC}2J${ESC}H`);
}

function leaveAlternateScreen(): void {
  process.stdout.write(`${ESC}?25h${ESC}?1049l`);
}

function enableMouseReporting(): void {
  process.stdout.write(`${ESC}?1000h${ESC}?1006h`);
}

function disableMouseReporting(): void {
  process.stdout.write(`${ESC}?1000l${ESC}?1002l${ESC}?1003l${ESC}?1006l`);
}

function clearScreen(): void {
  process.stdout.write(`${ESC}?25l${ESC}2J${ESC}H`);
}

function showCursor(): void {
  process.stdout.write(`${ESC}?25h`);
}

function color(code: number, text: string): string {
  return `${ESC}${code}m${text}${ESC}0m`;
}

function stripAnsi(value: string): string {
  return value.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, "");
}

function visibleLength(value: string): number {
  return Array.from(stripAnsi(value)).reduce((sum, ch) => sum + (ch.charCodeAt(0) > 255 ? 2 : 1), 0);
}

function padRight(value: string, width: number): string {
  const len = visibleLength(value);
  if (len >= width) return value;
  return value + " ".repeat(width - len);
}

function truncateVisible(value: string, width: number): string {
  let out = "";
  let len = 0;
  for (const ch of Array.from(value)) {
    const w = ch.charCodeAt(0) > 255 ? 2 : 1;
    if (len + w > width) break;
    out += ch;
    len += w;
  }
  return out;
}

function wrapLine(line: string, width: number): string[] {
  if (width <= 4) return [line];
  const out: string[] = [];
  let current = "";
  let currentLen = 0;
  for (const ch of Array.from(line)) {
    const w = ch.charCodeAt(0) > 255 ? 2 : 1;
    if (currentLen + w > width) {
      out.push(current);
      current = ch;
      currentLen = w;
    } else {
      current += ch;
      currentLen += w;
    }
  }
  out.push(current);
  return out;
}

function wrapText(text: string, width: number): string[] {
  const lines = text.split(/\r?\n/);
  return lines.flatMap((line) => wrapLine(line, width));
}

export class AgentTUI {
  private readonly eventBus = new AgentEventBus();
  private readonly toolRegistry = new ToolRegistry();
  private readonly chatClient: OpenAICompatibleChatClient;
  private readonly transcript: TranscriptItem[] = [];
  private input = "";
  private busy = false;
  private closed = false;
  private currentAssistantIndex: number | null = null;
  private currentThoughtIndex: number | null = null;
  private context: AgentContext;
  private ragEnabled = false;
  private registeredTools: string[] = [];
  private ragCleanup?: () => void;
  private alternateScreenActive = false;
  private transcriptScrollTop = 0;
  private transcriptStickToBottom = true;
  private pendingScrollToTranscriptIndex: number | null = null;
  private readonly stdinDecoder = new StringDecoder("utf8");
  private pendingInputBytes = "";

  constructor(private readonly runtime: RuntimeConfig) {
    this.chatClient = new OpenAICompatibleChatClient({
      apiKey: runtime.apiKey,
      baseURL: runtime.baseURL,
      model: runtime.model,
      timeoutMs: (runtime.agentConfig.llm_call_timeout ?? 120) * 1000,
    });
    this.context = {
      ...runtime.context,
      history: [
        ...(runtime.context.history ?? []),
        ...loadHistory(runtime.historyPath, runtime.agentConfig.history_turns),
      ],
    };
    this.setupRagTools();
    this.attachEvents();
  }

  private setupRagTools(): void {
    const result = registerRagTools(this.toolRegistry, this.runtime.ragConfig);
    this.ragEnabled = result.enabled;
    this.registeredTools = result.tools;
    this.ragCleanup = result.cleanup;
    if (result.knowledgeBase) {
      const existing = new Set(this.context.knowledge_bases.map((kb) => kb.id));
      if (!existing.has(result.knowledgeBase.id)) {
        this.context.knowledge_bases.push(result.knowledgeBase);
      }
    }
    if (result.selectedDocuments.length > 0) {
      const existing = new Set(this.context.selected_documents.map((doc) => doc.knowledge_id));
      for (const doc of result.selectedDocuments) {
        if (!existing.has(doc.knowledge_id)) {
          this.context.selected_documents.push(doc);
        }
      }
    }
    if (this.ragEnabled && result.knowledgeBase) {
      this.runtime.agentConfig.knowledge_bases = [result.knowledgeBase.id];
      this.runtime.agentConfig.knowledge_ids = result.selectedDocuments.map((doc) => doc.knowledge_id);
      this.runtime.agentConfig.allowed_tools = this.registeredTools;
    }
  }

  private attachEvents(): void {
    this.eventBus.on("status", ({ message }) => {
      this.upsertStatus(message);
      this.render();
    });

    this.eventBus.on("thought", ({ content, done }) => {
      if (done) {
        this.currentThoughtIndex = null;
        return;
      }
      if (this.currentThoughtIndex == null) {
        this.currentThoughtIndex = this.transcript.push({ role: "thought", content: "" }) - 1;
      }
      this.transcript[this.currentThoughtIndex].content += content;
      this.render();
    });

    this.eventBus.on("answer", ({ content, done }) => {
      if (done) {
        if (this.currentAssistantIndex != null && this.transcriptStickToBottom) {
          this.pendingScrollToTranscriptIndex = this.currentAssistantIndex;
        }
        this.currentAssistantIndex = null;
        this.render();
        return;
      }
      if (this.currentAssistantIndex == null) {
        this.currentAssistantIndex = this.transcript.push({ role: "assistant", content: "" }) - 1;
      }
      this.transcript[this.currentAssistantIndex].content += content;
      this.render();
    });

    this.eventBus.on("tool_call", ({ tool_name }) => {
      this.transcript.push({ role: "status", content: `Tool call requested: ${tool_name}` });
      this.render();
    });

    this.eventBus.on("tool_result", ({ tool_name, success, error, data }) => {
      if (data?.unavailable_tool) {
        this.transcript.push({
          role: "status",
          content: `Tool skipped: ${tool_name} is not available in this session`,
        });
        this.render();
        return;
      }
      this.transcript.push({
        role: success ? "status" : "error",
        content: success ? `Tool completed: ${tool_name}` : `Tool failed: ${tool_name}: ${error ?? ""}`,
      });
      this.render();
    });

    this.eventBus.on("error", ({ error }) => {
      this.transcript.push({ role: "error", content: error });
      this.render();
    });
  }

  private upsertStatus(content: string): void {
    const last = this.transcript[this.transcript.length - 1];
    if (last?.role === "status") {
      last.content = content;
    } else {
      this.transcript.push({ role: "status", content });
    }
  }

  private roleLabel(role: TranscriptRole): string {
    switch (role) {
      case "user":
        return color(36, "You");
      case "assistant":
        return color(32, "Assistant");
      case "thought":
        return color(35, "Thought");
      case "status":
        return color(90, "Status");
      case "error":
        return color(31, "Error");
      case "system":
        return color(33, "System");
    }
  }

  private currentLayout(): TerminalLayout {
    const width = process.stdout.columns || 100;
    const height = process.stdout.rows || 30;
    const headerHeight = 3;
    const footerHeight = 3;
    return {
      width,
      height,
      bodyHeight: Math.max(5, height - headerHeight - footerHeight),
    };
  }

  private buildTranscript(width: number): TranscriptRender {
    const bodyWidth = Math.max(20, width - 4);
    const lines: string[] = [];
    const itemStartLines: number[] = [];
    for (const item of this.transcript) {
      itemStartLines.push(lines.length);
      const label = this.roleLabel(item.role);
      const wrapped = wrapText(item.content || " ", bodyWidth - 10);
      lines.push(`${label}: ${wrapped[0] ?? ""}`);
      for (const line of wrapped.slice(1)) lines.push(`          ${line}`);
      lines.push("");
    }
    return { lines, itemStartLines };
  }

  private buildTranscriptLines(width: number): string[] {
    return this.buildTranscript(width).lines;
  }

  private renderTranscript(width: number, height: number): TranscriptViewport {
    const rendered = this.buildTranscript(width);
    const lines = rendered.lines;
    const maxScrollTop = Math.max(0, lines.length - height);
    if (this.pendingScrollToTranscriptIndex != null) {
      const requestedTop = rendered.itemStartLines[this.pendingScrollToTranscriptIndex] ?? maxScrollTop;
      this.transcriptScrollTop = Math.max(0, Math.min(requestedTop, maxScrollTop));
      this.transcriptStickToBottom = this.transcriptScrollTop >= maxScrollTop;
      this.pendingScrollToTranscriptIndex = null;
    } else if (this.transcriptStickToBottom) {
      this.transcriptScrollTop = maxScrollTop;
    } else {
      this.transcriptScrollTop = Math.max(0, Math.min(this.transcriptScrollTop, maxScrollTop));
      if (this.transcriptScrollTop >= maxScrollTop) {
        this.transcriptStickToBottom = true;
      }
    }
    return {
      lines: lines.slice(this.transcriptScrollTop, this.transcriptScrollTop + height),
      totalLines: lines.length,
      scrollTop: this.transcriptScrollTop,
      maxScrollTop,
    };
  }

  private scrollTranscript(delta: number): void {
    const { width, bodyHeight } = this.currentLayout();
    const maxScrollTop = Math.max(0, this.buildTranscriptLines(width).length - bodyHeight);
    if (maxScrollTop <= 0) {
      this.transcriptScrollTop = 0;
      this.transcriptStickToBottom = true;
      this.render();
      return;
    }

    const currentTop = this.transcriptStickToBottom ? maxScrollTop : this.transcriptScrollTop;
    this.transcriptScrollTop = Math.max(0, Math.min(currentTop + delta, maxScrollTop));
    this.transcriptStickToBottom = this.transcriptScrollTop >= maxScrollTop;
    this.render();
  }

  private scrollTranscriptPage(direction: -1 | 1): void {
    this.scrollTranscript(direction * Math.max(1, this.currentLayout().bodyHeight - 1));
  }

  private jumpTranscriptTop(): void {
    this.transcriptScrollTop = 0;
    this.transcriptStickToBottom = false;
    this.render();
  }

  private jumpTranscriptBottom(): void {
    this.transcriptStickToBottom = true;
    this.render();
  }

  private scrollHelp(viewport: TranscriptViewport, bodyHeight: number): string {
    if (viewport.totalLines <= bodyHeight) {
      return "Enter send | Ctrl+C quit";
    }
    const visibleStart = viewport.scrollTop + 1;
    const visibleEnd = Math.min(viewport.scrollTop + bodyHeight, viewport.totalLines);
    const location =
      viewport.scrollTop <= 0
        ? "top"
        : viewport.scrollTop >= viewport.maxScrollTop
          ? "bottom"
          : `${visibleStart}-${visibleEnd}/${viewport.totalLines}`;
    return `Enter send | Mouse wheel or Up/Down scroll | PgUp/PgDn page | ${location}`;
  }

  private commandHelp(): string {
    return [
      "/help                 show commands",
      "/clear                clear screen transcript",
      "/clear-history        clear persisted history",
      "/model                show current model config",
      "/context              show current context summary",
      "Mouse wheel           scroll transcript",
      "Up/Down               scroll transcript one line",
      "PageUp/PageDown       scroll transcript one page",
      "Home/End              jump transcript top/bottom",
      "/quit                 exit",
    ].join("\n");
  }

  private async handleCommand(command: string): Promise<void> {
    const cmd = command.trim();
    switch (cmd) {
      case "/help":
        this.transcript.push({ role: "system", content: this.commandHelp() });
        break;
      case "/clear":
        this.transcript.length = 0;
        this.transcriptScrollTop = 0;
        this.transcriptStickToBottom = true;
        this.pendingScrollToTranscriptIndex = null;
        break;
      case "/clear-history":
        clearHistory(this.runtime.historyPath);
        this.context.history = [];
        this.transcript.push({ role: "system", content: `History cleared: ${this.runtime.historyPath}` });
        break;
      case "/model":
        this.transcript.push({
          role: "system",
          content: `model=${this.runtime.model}\nbase_url=${this.runtime.baseURL}\nhistory=${this.runtime.historyPath}\nrag=${this.ragEnabled ? "enabled" : "disabled"}\ntools=${this.registeredTools.length}${this.registeredTools.length ? ` (${this.registeredTools.join(", ")})` : ""}`,
        });
        break;
      case "/context":
        this.transcript.push({
          role: "system",
          content: `session=${this.context.session_id}\nlanguage=${this.context.language ?? ""}\nknowledge_bases=${this.context.knowledge_bases.length}\nselected_documents=${this.context.selected_documents.length}\nhistory_messages=${this.context.history.length}`,
        });
        break;
      case "/quit":
      case "/exit":
        this.close();
        return;
      default:
        this.transcript.push({ role: "error", content: `Unknown command: ${cmd}` });
    }
    this.transcriptStickToBottom = true;
    this.pendingScrollToTranscriptIndex = null;
    this.render();
  }

  private async submitInput(): Promise<void> {
    const query = this.input.trim();
    this.input = "";
    if (!query || this.busy) {
      this.render();
      return;
    }
    if (query.startsWith("/")) {
      await this.handleCommand(query);
      return;
    }

    this.busy = true;
    this.currentAssistantIndex = null;
    this.currentThoughtIndex = null;
    this.transcript.push({ role: "user", content: query });
    this.transcriptStickToBottom = true;
    this.pendingScrollToTranscriptIndex = null;
    this.render();

    const contextForTurn: AgentContext = {
      ...this.context,
      history: [...this.context.history],
    };
    const engine = new AgentEngine(
      this.runtime.agentConfig,
      this.chatClient,
      this.toolRegistry,
      this.eventBus,
      contextForTurn,
    );

    try {
      const state = await engine.execute(query);
      const userMessage: Message = { role: "user", content: query };
      const assistantMessage: Message = { role: "assistant", content: state.final_answer };
      this.context.history.push(userMessage, assistantMessage);
      appendHistory(this.runtime.historyPath, userMessage);
      appendHistory(this.runtime.historyPath, assistantMessage);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.transcript.push({ role: "error", content: message });
    } finally {
      this.busy = false;
      this.render();
    }
  }

  private render(): void {
    if (this.closed) return;
    const { width, bodyHeight } = this.currentLayout();

    clearScreen();
    const title = ` Tiny RAG Agent TUI | model ${this.runtime.model} | tools ${this.registeredTools.length} | ${this.busy ? "running" : "ready"} `;
    process.stdout.write(color(1, truncateVisible(title, width)) + "\n");
    const modeLine = this.ragEnabled
      ? `RAG enabled: ${this.context.selected_documents.length} document(s). Commands: /help /quit`
      : "Pure Agent core, no RAG tools. Add documents under ./knowledge to enable retrieval.";
    process.stdout.write(color(90, padRight(modeLine, width)) + "\n");
    process.stdout.write(color(90, "-".repeat(width)) + "\n");

    const viewport = this.renderTranscript(width, bodyHeight);
    const lines = viewport.lines;
    for (let i = 0; i < bodyHeight; i++) {
      process.stdout.write(padRight(lines[i] ?? "", width) + "\n");
    }

    process.stdout.write(color(90, "-".repeat(width)) + "\n");
    const prompt = this.busy ? color(90, "waiting> ") : color(36, "you> ");
    const inputLine = truncateVisible(this.input, Math.max(1, width - visibleLength(stripAnsi(prompt)) - 1));
    process.stdout.write(prompt + inputLine + "\n");
    process.stdout.write(color(90, truncateVisible(this.scrollHelp(viewport, bodyHeight), width)) + "\n");
  }

  private handleStdinData = async (chunk: Buffer | string) => {
    this.pendingInputBytes += typeof chunk === "string" ? chunk : this.stdinDecoder.write(chunk);
    let shouldRender = false;

    for (;;) {
      const token = this.consumeInputToken();
      if (token === "incomplete") break;
      if (token === "close") return;
      if (token === "submit") {
        if (shouldRender) {
          this.render();
          shouldRender = false;
        }
        await this.submitInput();
        continue;
      }
      if (token === "render") {
        shouldRender = true;
      }
    }

    if (shouldRender) this.render();
  };

  private consumeInputToken(): "render" | "submit" | "close" | "incomplete" | "none" {
    const input = this.pendingInputBytes;
    if (!input) return "incomplete";

    const consume = (length: number) => {
      this.pendingInputBytes = this.pendingInputBytes.slice(length);
    };

    const consumeRender = (length: number): "render" => {
      consume(length);
      return "render";
    };

    if (input.startsWith("\x03")) {
      consume(1);
      this.close();
      return "close";
    }
    if (input.startsWith("\r") || input.startsWith("\n")) {
      consume(1);
      return "submit";
    }
    if (input.startsWith("\x7f") || input.startsWith("\b")) {
      this.input = Array.from(this.input).slice(0, -1).join("");
      return consumeRender(1);
    }

    const sgrMouse = input.match(/^\x1b\[<(\d+);(\d+);(\d+)([mM])/);
    if (sgrMouse) {
      this.handleMouseButtonCode(Number(sgrMouse[1]));
      return consumeRender(sgrMouse[0].length);
    }
    if (input.startsWith("\x1b[M")) {
      if (input.length < 6) return "incomplete";
      this.handleMouseButtonCode(input.charCodeAt(3) - 32);
      return consumeRender(6);
    }

    const escapeActions: Array<[string, () => void]> = [
      ["\x1b[A", () => this.scrollTranscriptWithoutRender(-1)],
      ["\x1b[B", () => this.scrollTranscriptWithoutRender(1)],
      ["\x1b[5~", () => this.scrollTranscriptPageWithoutRender(-1)],
      ["\x1b[6~", () => this.scrollTranscriptPageWithoutRender(1)],
      ["\x1b[H", () => this.jumpTranscriptTopWithoutRender()],
      ["\x1b[1~", () => this.jumpTranscriptTopWithoutRender()],
      ["\x1bOH", () => this.jumpTranscriptTopWithoutRender()],
      ["\x1b[F", () => this.jumpTranscriptBottomWithoutRender()],
      ["\x1b[4~", () => this.jumpTranscriptBottomWithoutRender()],
      ["\x1bOF", () => this.jumpTranscriptBottomWithoutRender()],
    ];
    for (const [sequence, action] of escapeActions) {
      if (input.startsWith(sequence)) {
        action();
        return consumeRender(sequence.length);
      }
    }
    if (this.isPotentialEscapePrefix(input)) {
      return "incomplete";
    }
    if (input.startsWith("\x1b")) {
      this.input = "";
      return consumeRender(1);
    }

    const [ch] = Array.from(input);
    if (!ch) return "incomplete";
    consume(ch.length);
    if (ch >= " " && ch !== "\x7f") {
      this.input += ch;
      return "render";
    }
    return "none";
  }

  private isPotentialEscapePrefix(input: string): boolean {
    if (input === "\x1b[" || input === "\x1b[<" || input === "\x1b[M" || input === "\x1bO") {
      return true;
    }
    return /^\x1b\[<\d*(?:;\d*){0,2}$/.test(input);
  }

  private handleMouseButtonCode(buttonCode: number): boolean {
    if ((buttonCode & 64) === 0) return false;
    const direction = (buttonCode & 1) === 0 ? -1 : 1;
    const step = Math.max(3, Math.floor(this.currentLayout().bodyHeight / 4));
    this.scrollTranscriptWithoutRender(direction * step);
    return true;
  }

  private scrollTranscriptWithoutRender(delta: number): void {
    const { width, bodyHeight } = this.currentLayout();
    const maxScrollTop = Math.max(0, this.buildTranscriptLines(width).length - bodyHeight);
    if (maxScrollTop <= 0) {
      this.transcriptScrollTop = 0;
      this.transcriptStickToBottom = true;
      return;
    }

    const currentTop = this.transcriptStickToBottom ? maxScrollTop : this.transcriptScrollTop;
    this.transcriptScrollTop = Math.max(0, Math.min(currentTop + delta, maxScrollTop));
    this.transcriptStickToBottom = this.transcriptScrollTop >= maxScrollTop;
  }

  private scrollTranscriptPageWithoutRender(direction: -1 | 1): void {
    this.scrollTranscriptWithoutRender(direction * Math.max(1, this.currentLayout().bodyHeight - 1));
  }

  private jumpTranscriptTopWithoutRender(): void {
    this.transcriptScrollTop = 0;
    this.transcriptStickToBottom = false;
  }

  private jumpTranscriptBottomWithoutRender(): void {
    this.transcriptStickToBottom = true;
  }

  private onResize = () => {
    this.render();
  };

  private onProcessExit = () => {
    this.restoreTerminal();
  };

  private onSignal = () => {
    this.close(0);
  };

  run(): void {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
      throw new Error("TUI requires an interactive terminal");
    }

    enterAlternateScreen();
    this.alternateScreenActive = true;
    enableMouseReporting();
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.on("data", this.handleStdinData);
    process.stdout.on("resize", this.onResize);
    process.once("exit", this.onProcessExit);
    process.once("SIGTERM", this.onSignal);
    process.once("SIGHUP", this.onSignal);
    this.transcript.push({
      role: "system",
      content: this.ragEnabled
        ? `Started. History: ${this.runtime.historyPath}\nRAG tools registered: ${this.registeredTools.join(", ")}\nIndexed documents: ${this.context.selected_documents.map((doc) => doc.title || doc.file_name || doc.knowledge_id).join(", ")}`
        : `Started. History: ${this.runtime.historyPath}\nNo RAG tools are registered. Add documents under ./knowledge to build the local index.`,
    });
    this.render();
  }

  private restoreTerminal(): void {
    process.stdin.off("data", this.handleStdinData);
    process.stdout.off("resize", this.onResize);
    process.off("exit", this.onProcessExit);
    process.off("SIGTERM", this.onSignal);
    process.off("SIGHUP", this.onSignal);
    this.ragCleanup?.();
    this.ragCleanup = undefined;
    disableMouseReporting();
    if (process.stdin.isTTY) process.stdin.setRawMode(false);
    if (this.alternateScreenActive) {
      this.alternateScreenActive = false;
      leaveAlternateScreen();
      return;
    }
    showCursor();
  }

  close(exitCode = 0): void {
    if (this.closed) return;
    this.closed = true;
    this.restoreTerminal();
    process.exit(exitCode);
  }
}

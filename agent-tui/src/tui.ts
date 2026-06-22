import readline from "node:readline";
import { AgentEngine } from "./engine.js";
import { AgentEventBus } from "./events.js";
import { appendHistory, clearHistory, loadHistory } from "./history.js";
import { OpenAICompatibleChatClient } from "./chat-client.js";
import { ToolRegistry } from "./tool-registry.js";
import type { AgentContext, Message } from "./types.js";
import type { RuntimeConfig } from "./config.js";

type TranscriptRole = "user" | "assistant" | "thought" | "status" | "error" | "system";

interface TranscriptItem {
  role: TranscriptRole;
  content: string;
}

const ESC = "\x1b[";

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
    this.attachEvents();
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

    this.eventBus.on("tool_result", ({ tool_name, success, error }) => {
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

  private renderTranscript(width: number, height: number): string[] {
    const bodyWidth = Math.max(20, width - 4);
    const rendered: string[] = [];
    for (const item of this.transcript) {
      const label = this.roleLabel(item.role);
      const wrapped = wrapText(item.content || " ", bodyWidth - 10);
      rendered.push(`${label}: ${wrapped[0] ?? ""}`);
      for (const line of wrapped.slice(1)) rendered.push(`          ${line}`);
      rendered.push("");
    }
    return rendered.slice(-height);
  }

  private commandHelp(): string {
    return [
      "/help                 show commands",
      "/clear                clear screen transcript",
      "/clear-history        clear persisted history",
      "/model                show current model config",
      "/context              show current context summary",
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
        break;
      case "/clear-history":
        clearHistory(this.runtime.historyPath);
        this.context.history = [];
        this.transcript.push({ role: "system", content: `History cleared: ${this.runtime.historyPath}` });
        break;
      case "/model":
        this.transcript.push({
          role: "system",
          content: `model=${this.runtime.model}\nbase_url=${this.runtime.baseURL}\nhistory=${this.runtime.historyPath}\ntools=0`,
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
    const width = process.stdout.columns || 100;
    const height = process.stdout.rows || 30;
    const headerHeight = 3;
    const footerHeight = 3;
    const bodyHeight = Math.max(5, height - headerHeight - footerHeight);

    clearScreen();
    const title = ` Tiny RAG Agent TUI | model ${this.runtime.model} | tools 0 | ${this.busy ? "running" : "ready"} `;
    process.stdout.write(color(1, truncateVisible(title, width)) + "\n");
    process.stdout.write(color(90, padRight("Pure Agent core, no built-in tools. Commands: /help /quit", width)) + "\n");
    process.stdout.write(color(90, "-".repeat(width)) + "\n");

    const lines = this.renderTranscript(width, bodyHeight);
    for (let i = 0; i < bodyHeight; i++) {
      process.stdout.write(padRight(lines[i] ?? "", width) + "\n");
    }

    process.stdout.write(color(90, "-".repeat(width)) + "\n");
    const prompt = this.busy ? color(90, "waiting> ") : color(36, "you> ");
    const inputLine = truncateVisible(this.input, Math.max(1, width - visibleLength(stripAnsi(prompt)) - 1));
    process.stdout.write(prompt + inputLine + "\n");
    process.stdout.write(color(90, "Enter send | Ctrl+C quit") + "\n");
  }

  private onKeypress = async (str: string, key: readline.Key) => {
    if (key.ctrl && key.name === "c") {
      this.close();
      return;
    }
    if (key.name === "return") {
      await this.submitInput();
      return;
    }
    if (key.name === "backspace") {
      this.input = Array.from(this.input).slice(0, -1).join("");
      this.render();
      return;
    }
    if (key.name === "escape") {
      this.input = "";
      this.render();
      return;
    }
    if (str && !key.ctrl && !key.meta && key.name !== "up" && key.name !== "down" && key.name !== "left" && key.name !== "right") {
      this.input += str;
      this.render();
    }
  };

  run(): void {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
      throw new Error("TUI requires an interactive terminal");
    }

    readline.emitKeypressEvents(process.stdin);
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.on("keypress", this.onKeypress);
    process.stdout.on("resize", () => this.render());
    this.transcript.push({
      role: "system",
      content: `Started. History: ${this.runtime.historyPath}\nNo tools are registered in this build.`,
    });
    this.render();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    process.stdin.off("keypress", this.onKeypress);
    if (process.stdin.isTTY) process.stdin.setRawMode(false);
    showCursor();
    process.stdout.write(`${ESC}2J${ESC}H`);
    process.exit(0);
  }
}

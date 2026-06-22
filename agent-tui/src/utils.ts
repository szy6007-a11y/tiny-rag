import type { Message } from "./types.js";

const thinkOpenTag = "<think>";
const thinkCloseTag = "</think>";

export function generateEventID(suffix: string): string {
  return `${crypto.randomUUID().slice(0, 8)}-${suffix}`;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function stripThinkBlocks(content: string): string {
  return content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function holdBackPartialTag(value: string, tag: string): [safe: string, hold: string] {
  const maxK = Math.min(tag.length - 1, value.length);
  for (let k = maxK; k >= 1; k--) {
    if (value.endsWith(tag.slice(0, k))) {
      return [value.slice(0, value.length - k), value.slice(value.length - k)];
    }
  }
  return [value, ""];
}

export class ThinkStreamSplitter {
  private inThink = false;
  private pending = "";

  feed(value: string): { think: string; answer: string } {
    if (!value) return { think: "", answer: "" };
    this.pending += value;

    let think = "";
    let answer = "";

    for (;;) {
      if (this.inThink) {
        const idx = this.pending.indexOf(thinkCloseTag);
        if (idx >= 0) {
          think += this.pending.slice(0, idx);
          this.pending = this.pending.slice(idx + thinkCloseTag.length);
          this.inThink = false;
          continue;
        }
        const [safe, hold] = holdBackPartialTag(this.pending, thinkCloseTag);
        think += safe;
        this.pending = hold;
        return { think, answer };
      }

      const idx = this.pending.indexOf(thinkOpenTag);
      if (idx >= 0) {
        answer += this.pending.slice(0, idx);
        this.pending = this.pending.slice(idx + thinkOpenTag.length);
        this.inThink = true;
        continue;
      }

      const [safe, hold] = holdBackPartialTag(this.pending, thinkOpenTag);
      answer += safe;
      this.pending = hold;
      return { think, answer };
    }
  }

  flush(): { think: string; answer: string } {
    const rest = this.pending;
    this.pending = "";
    if (!rest) return { think: "", answer: "" };
    if (this.inThink) return { think: rest, answer: "" };
    return { think: "", answer: rest };
  }
}

function hasMatchingToolCall(messages: Message[], toolCallID: string): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "assistant") {
      for (const tc of msg.tool_calls ?? []) {
        if (tc.id === toolCallID) return true;
      }
    }
  }
  return false;
}

export function sanitizeMessages(messages: Message[]): Message[] {
  if (messages.length === 0) return messages;

  const result: Message[] = [];
  messages.forEach((original, index) => {
    const msg: Message = { ...original };
    if (msg.content === "" && msg.role !== "system" && msg.role !== "tool" && !(msg.tool_calls?.length)) {
      return;
    }

    if (result.length > 0 && msg.role !== "tool") {
      const prev = result[result.length - 1];
      if (prev.role === msg.role) {
        prev.content += `\n\n${msg.content}`;
        return;
      }
    }

    if (msg.role === "tool" && msg.tool_call_id) {
      if (!hasMatchingToolCall(messages.slice(0, index), msg.tool_call_id)) {
        msg.role = "system";
        msg.content = `[Tool result for ${msg.name ?? ""}]: ${msg.content}`;
        delete msg.tool_call_id;
        delete msg.name;
      }
    }

    result.push(msg);
  });

  return result;
}

export function estimateMessageTokens(message: Message): number {
  const text = [
    message.role,
    message.content,
    message.reasoning_content ?? "",
    JSON.stringify(message.tool_calls ?? []),
  ].join("\n");
  return Math.ceil(Array.from(text).length / 4) + 8;
}

export function estimateMessagesTokens(messages: Message[]): number {
  return messages.reduce((sum, msg) => sum + estimateMessageTokens(msg), 0);
}

function groupToolMessages(messages: Message[]): Message[][] {
  const groups: Message[][] = [];
  let i = 0;
  while (i < messages.length) {
    const msg = messages[i];
    if (msg.role === "assistant" && (msg.tool_calls?.length ?? 0) > 0) {
      const group = [msg];
      i++;
      while (i < messages.length && messages[i].role === "tool") {
        group.push(messages[i]);
        i++;
      }
      groups.push(group);
      continue;
    }
    groups.push([msg]);
    i++;
  }
  return groups;
}

export function compressContext(messages: Message[], maxTokens: number, currentTokens: number): Message[] {
  if (maxTokens <= 0 || messages.length <= 2) return messages;

  const threshold = Math.floor(maxTokens * 0.8);
  if (currentTokens <= threshold) return messages;

  const systemMsg = messages[0];
  let lastUserIdx = messages.length - 1;
  for (let i = messages.length - 1; i >= 1; i--) {
    if (messages[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }

  const history = messages.slice(1, lastUserIdx);
  const tail = messages.slice(lastUserIdx);
  if (history.length === 0) return messages;

  const groups = groupToolMessages(history);
  const tokensToFree = currentTokens - threshold;
  let freed = 0;
  let removeUpTo = 0;

  for (let i = 0; i < groups.length; i++) {
    const groupTokens = groups[i].reduce((sum, msg) => sum + estimateMessageTokens(msg), 0);
    freed += groupTokens;
    removeUpTo = i + 1;
    if (freed >= tokensToFree) break;
  }

  return [systemMsg, ...groups.slice(removeUpTo).flat(), ...tail];
}

export function truncateRunes(value: string, max: number): string {
  if (max <= 0 || !value) return value;
  const runes = Array.from(value);
  if (runes.length <= max) return value;
  return `${runes.slice(0, max).join("")}…`;
}

export function parseJSONRecord(value: string): Record<string, unknown> {
  const parsed = JSON.parse(value) as unknown;
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return parsed as Record<string, unknown>;
  }
  return {};
}

export function isTransientError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const text = err.message.toLowerCase();
  return [
    "429",
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "server error",
    "temporarily unavailable",
  ].some((marker) => text.includes(marker));
}

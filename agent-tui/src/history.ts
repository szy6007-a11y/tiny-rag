import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { Message } from "./types.js";

interface HistoryRecord {
  timestamp: string;
  message: Message;
}

export function loadHistory(path: string, maxTurns: number): Message[] {
  if (!existsSync(path)) return [];
  const lines = readFileSync(path, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const messages: Message[] = [];
  for (const line of lines) {
    try {
      const record = JSON.parse(line) as HistoryRecord;
      if (record.message) messages.push(record.message);
    } catch {
      // Keep loading best-effort, matching the agent's graceful degradation style.
    }
  }

  if (maxTurns <= 0) return messages;
  return messages.slice(-maxTurns * 2);
}

export function appendHistory(path: string, message: Message): void {
  const dir = dirname(path);
  if (dir && dir !== ".") mkdirSync(dir, { recursive: true });
  const record: HistoryRecord = {
    timestamp: new Date().toISOString(),
    message,
  };
  appendFileSync(path, `${JSON.stringify(record)}\n`);
}

export function clearHistory(path: string): void {
  const dir = dirname(path);
  if (dir && dir !== ".") mkdirSync(dir, { recursive: true });
  writeFileSync(path, "");
}

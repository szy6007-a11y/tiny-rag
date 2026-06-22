import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  DEFAULT_AGENT_MAX_ITERATIONS,
  DEFAULT_AGENT_TEMPERATURE,
  DEFAULT_MAX_CONTEXT_TOKENS,
  type AgentConfig,
  type AgentContext,
} from "./types.js";

export interface RuntimeConfig {
  apiKey: string;
  baseURL: string;
  model: string;
  historyPath: string;
  contextPath?: string;
  agentConfig: AgentConfig;
  context: AgentContext;
}

function readJSONFile<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function parseDotEnv(path: string): Record<string, string> {
  if (!existsSync(path)) return {};
  const out: Record<string, string> = {};
  for (const rawLine of readFileSync(path, "utf8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function findUp(fileName: string, startDir = process.cwd()): string | undefined {
  let current = resolve(startDir);
  for (;;) {
    const candidate = resolve(current, fileName);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(current);
    if (parent === current) return undefined;
    current = parent;
  }
}

function resolveDefaultEnvPath(): string | undefined {
  return findUp(".env");
}

function argValue(args: string[], name: string): string | undefined {
  const idx = args.indexOf(name);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  const prefix = `${name}=`;
  const hit = args.find((arg) => arg.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : undefined;
}

export function defaultAgentConfig(): AgentConfig {
  return {
    max_iterations: DEFAULT_AGENT_MAX_ITERATIONS,
    allowed_tools: [],
    temperature: DEFAULT_AGENT_TEMPERATURE,
    knowledge_bases: [],
    knowledge_ids: [],
    use_custom_system_prompt: false,
    multi_turn_enabled: true,
    history_turns: 20,
    retain_retrieval_history: false,
    skills_enabled: false,
    max_tool_output_chars: 16_000,
    max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
    parallel_tool_calls: false,
  };
}

export function defaultAgentContext(): AgentContext {
  return {
    session_id: `local-${new Date().toISOString()}`,
    language: "Chinese (Simplified)",
    knowledge_bases: [],
    selected_documents: [],
    history: [],
  };
}

export function loadRuntimeConfig(argv = process.argv.slice(2)): RuntimeConfig {
  const envPath = argValue(argv, "--env") ?? process.env.AGENT_TUI_ENV ?? resolveDefaultEnvPath();
  const fileEnv = envPath ? parseDotEnv(envPath) : {};
  const contextPath = argValue(argv, "--context") ?? process.env.AGENT_TUI_CONTEXT;
  const historyPath = resolve(argValue(argv, "--history") ?? process.env.AGENT_TUI_HISTORY ?? ".agent-tui-history.jsonl");

  const apiKey =
    argValue(argv, "--api-key") ??
    process.env.LLM_API_KEY ??
    fileEnv.LLM_API_KEY ??
    process.env.OPENAI_API_KEY ??
    fileEnv.OPENAI_API_KEY ??
    "";
  const baseURL =
    argValue(argv, "--base-url") ??
    process.env.LLM_BASE_URL ??
    fileEnv.LLM_BASE_URL ??
    process.env.OPENAI_BASE_URL ??
    fileEnv.OPENAI_BASE_URL ??
    "https://api.openai.com/v1";
  const model =
    argValue(argv, "--model") ??
    process.env.LLM_MODEL_NAME ??
    fileEnv.LLM_MODEL_NAME ??
    process.env.OPENAI_MODEL ??
    fileEnv.OPENAI_MODEL ??
    process.env.AGENT_MODEL ??
    "gpt-4o-mini";

  let agentConfig = defaultAgentConfig();
  let context = defaultAgentContext();
  if (contextPath && existsSync(contextPath)) {
    const loaded = readJSONFile<Partial<AgentContext> & { agent_config?: Partial<AgentConfig> }>(contextPath);
    context = {
      ...context,
      ...loaded,
      knowledge_bases: loaded.knowledge_bases ?? [],
      selected_documents: loaded.selected_documents ?? [],
      history: loaded.history ?? [],
    };
    agentConfig = {
      ...agentConfig,
      ...(loaded.agent_config ?? {}),
      allowed_tools: [],
    };
  }

  return {
    apiKey,
    baseURL,
    model,
    historyPath,
    contextPath,
    agentConfig,
    context,
  };
}

export function printHelp(): string {
  return `Usage:
  npm run dev -- [options]

Options:
  --context <file>    Load agent context JSON
  --history <file>    Read/write history JSONL
  --model <name>      OpenAI-compatible model name
  --base-url <url>    OpenAI-compatible base URL
  --api-key <key>     API key
  --env <file>        Load a tiny-rag .env file

Environment:
  LLM_API_KEY         Preferred default
  LLM_BASE_URL        Preferred default
  LLM_MODEL_NAME      Preferred default
  OPENAI_*            Fallbacks
`;
}

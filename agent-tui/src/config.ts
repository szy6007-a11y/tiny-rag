import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  DEFAULT_AGENT_MAX_ITERATIONS,
  DEFAULT_AGENT_TEMPERATURE,
  DEFAULT_MAX_CONTEXT_TOKENS,
  type AgentConfig,
  type AgentContext,
} from "./types.js";
import { defaultBridgeScript, defaultIngestScript, type RagConfig } from "./rag-tools.js";

export interface RuntimeConfig {
  apiKey: string;
  baseURL: string;
  model: string;
  historyPath: string;
  contextPath?: string;
  ragConfig: RagConfig;
  agentConfig: AgentConfig;
  context: AgentContext;
}

type ContextRagConfig = Partial<RagConfig> & {
  enabled?: boolean;
};

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

function isDisabled(value: string | undefined): boolean {
  return value?.trim().toLowerCase() === "1" || value?.trim().toLowerCase() === "true";
}

function defaultRepoRoot(): string {
  return dirname(defaultIngestScript());
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
  let contextRag: ContextRagConfig = {};
  if (contextPath && existsSync(contextPath)) {
    const loaded = readJSONFile<
      Partial<AgentContext> & {
        agent_config?: Partial<AgentConfig>;
        rag?: ContextRagConfig;
      }
    >(contextPath);
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
    contextRag = loaded.rag ?? {};
  }

  const ragDisabled =
    argv.includes("--no-rag") ||
    isDisabled(process.env.AGENT_TUI_NO_RAG) ||
    contextRag.enabled === false;
  const ragSyncDisabled =
    argv.includes("--no-rag-sync") ||
    isDisabled(process.env.AGENT_TUI_NO_RAG_SYNC) ||
    contextRag.autoSync === false;
  const ragWatchDisabled =
    argv.includes("--no-rag-watch") ||
    isDisabled(process.env.AGENT_TUI_NO_RAG_WATCH) ||
    contextRag.watch === false;

  const repoRoot = defaultRepoRoot();
  const ragDbPath = resolve(
    argValue(argv, "--rag-db") ??
      process.env.AGENT_TUI_RAG_DB ??
      contextRag.dbPath ??
      resolve(repoRoot, ".agent-tui-rag.sqlite"),
  );
  const ragDocsDir = resolve(
    argValue(argv, "--docs-dir") ??
      process.env.AGENT_TUI_KNOWLEDGE_DIR ??
      contextRag.docsDir ??
      resolve(repoRoot, "knowledge"),
  );
  const ragWatchInterval = Number(
    argValue(argv, "--rag-watch-interval") ??
      process.env.AGENT_TUI_RAG_WATCH_INTERVAL ??
      contextRag.watchInterval ??
      1.5,
  );
  const ragConfig: RagConfig = {
    dbPath: ragDbPath,
    pythonCommand:
      argValue(argv, "--rag-python") ??
      process.env.AGENT_TUI_RAG_PYTHON ??
      contextRag.pythonCommand ??
      "uv run python",
    bridgeScript: resolve(
      argValue(argv, "--rag-bridge") ??
        process.env.AGENT_TUI_RAG_BRIDGE ??
        contextRag.bridgeScript ??
        defaultBridgeScript(),
    ),
    ingestScript: resolve(
      argValue(argv, "--rag-ingest") ??
        process.env.AGENT_TUI_RAG_INGEST ??
        contextRag.ingestScript ??
        defaultIngestScript(),
    ),
    docsDir: ragDocsDir,
    enabled: !ragDisabled,
    autoSync: !ragDisabled && !ragSyncDisabled,
    watch: !ragDisabled && !ragSyncDisabled && !ragWatchDisabled,
    watchInterval: Number.isFinite(ragWatchInterval) && ragWatchInterval > 0 ? ragWatchInterval : 1.5,
  };

  return {
    apiKey,
    baseURL,
    model,
    historyPath,
    contextPath,
    ragConfig,
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
  --rag-db <file>     SQLite path for an existing local TUI RAG index
  --rag-python <cmd>  Python runner for the bridge, default: "uv run python"
  --docs-dir <dir>    Local knowledge directory, default: repo-root ./knowledge
  --no-rag-sync       Do not sync repo-root ./knowledge on startup
  --no-rag-watch      Do not watch repo-root ./knowledge while the TUI is running
  --no-rag            Start pure chat mode without opening a local index

Environment:
  LLM_API_KEY         Preferred default
  LLM_BASE_URL        Preferred default
  LLM_MODEL_NAME      Preferred default
  OPENAI_*            Fallbacks
  AGENT_TUI_RAG_DB    Existing SQLite index path, default: repo-root .agent-tui-rag.sqlite
  AGENT_TUI_KNOWLEDGE_DIR
                      Knowledge directory, default: repo-root ./knowledge
  AGENT_TUI_NO_RAG    Set to true/1 to disable local RAG tools

Offline ingest:
  Put documents under repo-root ./knowledge; the TUI syncs and watches it automatically.
`;
}

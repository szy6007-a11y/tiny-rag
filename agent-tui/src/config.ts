import { existsSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import {
  DEFAULT_AGENT_MAX_ITERATIONS,
  DEFAULT_AGENT_TEMPERATURE,
  DEFAULT_MAX_CONTEXT_TOKENS,
  type AgentConfig,
  type AgentContext,
} from "./types.js";
import { defaultBridgeScript, type RagConfig } from "./rag-tools.js";

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
  document?: string | false;
  documents?: string[] | false;
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

function argValues(args: string[], name: string): string[] {
  const out: string[] = [];
  const prefix = `${name}=`;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === name && i + 1 < args.length) {
      out.push(args[i + 1]);
      i++;
      continue;
    }
    if (arg.startsWith(prefix)) {
      out.push(arg.slice(prefix.length));
    }
  }
  return out;
}

function intArg(args: string[], name: string, envValue: string | undefined, fallback: number): number {
  const raw = argValue(args, name) ?? envValue ?? "";
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function splitEnvList(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function resolveFrom(baseDir: string, path: string): string {
  return isAbsolute(path) ? path : resolve(baseDir, path);
}

function defaultKnowledgeDocument(): string {
  return resolve(dirname(defaultBridgeScript()), "../agent-tui/examples/default_kb.md");
}

function isDisabled(value: string | undefined): boolean {
  return value?.trim().toLowerCase() === "1" || value?.trim().toLowerCase() === "true";
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
  const contextDir = contextPath ? dirname(resolve(contextPath)) : process.cwd();
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

  const contextDocs = [
    ...(Array.isArray(contextRag.documents) ? contextRag.documents : []),
    ...(typeof contextRag.document === "string" ? [contextRag.document] : []),
  ];
  const explicitRagDocuments = [
    ...contextDocs.map((path) => resolveFrom(contextDir, path)),
    ...splitEnvList(process.env.AGENT_TUI_RAG_DOCUMENTS).map((path) => resolve(path)),
    ...argValues(argv, "--rag-document").map((path) => resolve(path)),
    ...argValues(argv, "--rag-doc").map((path) => resolve(path)),
  ];
  const ragDisabled =
    argv.includes("--no-rag") ||
    isDisabled(process.env.AGENT_TUI_NO_RAG) ||
    contextRag.enabled === false ||
    contextRag.documents === false ||
    contextRag.document === false;
  const ragDocuments = ragDisabled
    ? []
    : explicitRagDocuments.length > 0
      ? explicitRagDocuments
      : [defaultKnowledgeDocument()];

  const ragDbPath = resolve(
    argValue(argv, "--rag-db") ??
      process.env.AGENT_TUI_RAG_DB ??
      contextRag.dbPath ??
      ".agent-tui-rag.sqlite",
  );
  const ragConfig: RagConfig = {
    documents: [...new Set(ragDocuments)],
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
    tenantId: intArg(argv, "--rag-tenant-id", process.env.AGENT_TUI_RAG_TENANT_ID, contextRag.tenantId ?? 1),
    knowledgeBaseId:
      argValue(argv, "--rag-kb-id") ??
      process.env.AGENT_TUI_RAG_KB_ID ??
      contextRag.knowledgeBaseId ??
      "local-tui-kb",
    knowledgeBaseName:
      argValue(argv, "--rag-kb-name") ??
      process.env.AGENT_TUI_RAG_KB_NAME ??
      contextRag.knowledgeBaseName ??
      "Local TUI Knowledge Base",
    chunkSize: intArg(argv, "--rag-chunk-size", process.env.AGENT_TUI_RAG_CHUNK_SIZE, contextRag.chunkSize ?? 512),
    chunkOverlap: intArg(
      argv,
      "--rag-chunk-overlap",
      process.env.AGENT_TUI_RAG_CHUNK_OVERLAP,
      contextRag.chunkOverlap ?? 80,
    ),
    embeddingDimensions: intArg(
      argv,
      "--rag-embedding-dimensions",
      process.env.AGENT_TUI_RAG_EMBEDDING_DIMENSIONS,
      contextRag.embeddingDimensions ?? 128,
    ),
    maxToolOutputChars: agentConfig.max_tool_output_chars ?? 16000,
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
  --rag-document <f>  Use local document(s) instead of the default KB (repeatable)
  --rag-db <file>     SQLite path for the local TUI RAG index
  --rag-python <cmd>  Python runner for the bridge, default: "uv run python"
  --no-rag            Start pure chat mode without the default KB

Environment:
  LLM_API_KEY         Preferred default
  LLM_BASE_URL        Preferred default
  LLM_MODEL_NAME      Preferred default
  OPENAI_*            Fallbacks
  AGENT_TUI_RAG_DOCUMENTS  Comma-separated local documents replacing the default KB
  AGENT_TUI_NO_RAG         Set to true/1 to disable default RAG mode
`;
}

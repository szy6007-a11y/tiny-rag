import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ChatTool, KnowledgeBaseInfo, RuntimeTool, SelectedDocumentInfo, ToolResult } from "./types.js";
import { ToolRegistry } from "./tool-registry.js";

export interface RagConfig {
  dbPath: string;
  pythonCommand: string;
  bridgeScript: string;
  ingestScript: string;
  docsDir: string;
  enabled: boolean;
  autoSync: boolean;
  watch: boolean;
  watchInterval: number;
}

export interface RagInitResult {
  enabled: boolean;
  knowledgeBase?: KnowledgeBaseInfo;
  selectedDocuments: SelectedDocumentInfo[];
  tools: string[];
  cleanup?: () => void;
}

interface BridgeDefinitionsPayload {
  success: boolean;
  error?: string;
  knowledge_base?: KnowledgeBaseInfo;
  selected_documents?: SelectedDocumentInfo[];
  tools?: ChatTool[];
}

const DEFAULT_BRIDGE_SCRIPT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../scripts/tui_rag_bridge.py",
);
const DEFAULT_INGEST_SCRIPT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../ingest.py",
);

export function defaultBridgeScript(): string {
  return DEFAULT_BRIDGE_SCRIPT;
}

export function defaultIngestScript(): string {
  return DEFAULT_INGEST_SCRIPT;
}

export function registerRagTools(registry: ToolRegistry, config: RagConfig): RagInitResult {
  if (!config.enabled) {
    return { enabled: false, selectedDocuments: [], tools: [] };
  }

  if (config.autoSync) {
    runIngestSync(config, ["--allow-empty"]);
  }

  if (!existingIndexAvailable(config.dbPath)) {
    return { enabled: false, selectedDocuments: [], tools: [] };
  }

  const definitionsPayload = runBridgeSync(config, [
    "definitions",
    "--db",
    config.dbPath,
  ]) as BridgeDefinitionsPayload;

  if (!definitionsPayload.success) {
    throw new Error(definitionsPayload.error || "failed to load local RAG bridge definitions");
  }

  const tools = definitionsPayload.tools ?? [];
  for (const tool of tools) {
    registry.registerTool(new PythonRagTool(config, tool));
  }

  const watcher = config.autoSync && config.watch ? startIngestWatcher(config) : undefined;
  return {
    enabled: true,
    knowledgeBase: definitionsPayload.knowledge_base,
    selectedDocuments: definitionsPayload.selected_documents ?? [],
    tools: tools.map((tool) => tool.function.name).sort(),
    cleanup: watcher
      ? () => {
          watcher.kill("SIGTERM");
        }
      : undefined,
  };
}

function existingIndexAvailable(dbPath: string): boolean {
  return existsSync(dbPath) && existsSync(`${dbPath}.manifest.json`);
}

class PythonRagTool implements RuntimeTool {
  constructor(
    private readonly config: RagConfig,
    private readonly definition: ChatTool,
  ) {}

  name(): string {
    return this.definition.function.name;
  }

  description(): string {
    return this.definition.function.description;
  }

  parameters(): unknown {
    return this.definition.function.parameters;
  }

  execute(args: unknown, signal?: AbortSignal): Promise<ToolResult> {
    return runBridgeAsync(
      this.config,
      ["execute", "--db", this.config.dbPath, "--tool", this.name()],
      JSON.stringify(args ?? {}),
      signal,
    ) as Promise<ToolResult>;
  }
}

function splitCommand(command: string): string[] {
  return command.trim().split(/\s+/).filter(Boolean);
}

function bridgeArgs(config: RagConfig, args: string[]): string[] {
  return [config.bridgeScript, ...args];
}

function ingestArgs(config: RagConfig, args: string[]): string[] {
  return [
    config.ingestScript,
    "--docs-dir",
    config.docsDir,
    "--db",
    config.dbPath,
    "--json",
    ...args,
  ];
}

function parseBridgeOutput(stdout: string, stderr: string): unknown {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error(stderr.trim() || "bridge returned no output");
  }
  try {
    return JSON.parse(trimmed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`failed to parse bridge output: ${message}\n${trimmed}`);
  }
}

function runBridgeSync(config: RagConfig, args: string[]): unknown {
  return runPythonSync(config, bridgeArgs(config, args), "bridge");
}

function runIngestSync(config: RagConfig, args: string[]): unknown {
  return runPythonSync(config, ingestArgs(config, args), "ingest");
}

function runPythonSync(config: RagConfig, args: string[], label: string): unknown {
  const command = splitCommand(config.pythonCommand);
  if (command.length === 0) throw new Error("RAG python command is empty");
  const [bin, ...prefixArgs] = command;
  const result = spawnSync(bin, [...prefixArgs, ...args], {
    encoding: "utf8",
    cwd: resolve(dirname(config.bridgeScript), ".."),
    env: process.env,
    maxBuffer: 10 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  const payload = parseBridgeOutput(result.stdout || "", result.stderr || "");
  if (result.status !== 0) {
    const pythonError =
      typeof payload === "object" && payload && "error" in payload
        ? String((payload as { error?: unknown }).error || "")
        : "";
    throw new Error(pythonError || result.stderr || `${label} exited with status ${result.status}`);
  }
  return payload;
}

function startIngestWatcher(config: RagConfig): ChildProcess {
  const command = splitCommand(config.pythonCommand);
  if (command.length === 0) throw new Error("RAG python command is empty");
  const [bin, ...prefixArgs] = command;
  return spawn(
    bin,
    [
      ...prefixArgs,
      ...ingestArgs(config, [
        "--watch",
        "--allow-empty",
        "--watch-interval",
        String(config.watchInterval),
      ]),
    ],
    {
      cwd: resolve(dirname(config.bridgeScript), ".."),
      env: process.env,
      stdio: "ignore",
    },
  );
}

function runBridgeAsync(
  config: RagConfig,
  args: string[],
  input: string,
  signal?: AbortSignal,
): Promise<unknown> {
  const command = splitCommand(config.pythonCommand);
  if (command.length === 0) return Promise.reject(new Error("RAG python command is empty"));
  const [bin, ...prefixArgs] = command;

  return new Promise((resolvePromise, reject) => {
    const child = spawn(bin, [...prefixArgs, ...bridgeArgs(config, args)], {
      cwd: resolve(dirname(config.bridgeScript), ".."),
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";

    const abort = () => {
      child.kill("SIGTERM");
      reject(new Error("RAG tool execution aborted"));
    };
    if (signal?.aborted) {
      abort();
      return;
    }
    signal?.addEventListener("abort", abort, { once: true });

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      signal?.removeEventListener("abort", abort);
      reject(error);
    });
    child.on("close", (code) => {
      signal?.removeEventListener("abort", abort);
      try {
        const payload = parseBridgeOutput(stdout, stderr);
        if (code !== 0) {
          const bridgeError =
            typeof payload === "object" && payload && "error" in payload
              ? String((payload as { error?: unknown }).error || "")
              : "";
          reject(new Error(bridgeError || stderr || `bridge exited with status ${code}`));
          return;
        }
        resolvePromise(payload);
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(input);
  });
}

import { spawn, spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ChatTool, KnowledgeBaseInfo, RuntimeTool, SelectedDocumentInfo, ToolResult } from "./types.js";
import { ToolRegistry } from "./tool-registry.js";

export interface RagConfig {
  documents: string[];
  dbPath: string;
  pythonCommand: string;
  bridgeScript: string;
  tenantId: number;
  knowledgeBaseId: string;
  knowledgeBaseName: string;
  chunkSize: number;
  chunkOverlap: number;
  embeddingDimensions: number;
  maxToolOutputChars: number;
}

export interface RagInitResult {
  enabled: boolean;
  knowledgeBase?: KnowledgeBaseInfo;
  selectedDocuments: SelectedDocumentInfo[];
  tools: string[];
}

interface BridgeInitPayload {
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

export function defaultBridgeScript(): string {
  return DEFAULT_BRIDGE_SCRIPT;
}

export function registerRagTools(registry: ToolRegistry, config: RagConfig): RagInitResult {
  if (config.documents.length === 0) {
    return { enabled: false, selectedDocuments: [], tools: [] };
  }

  const initPayload = runBridgeSync(config, [
    "init",
    "--db",
    config.dbPath,
    "--tenant-id",
    String(config.tenantId),
    "--knowledge-base-id",
    config.knowledgeBaseId,
    "--knowledge-base-name",
    config.knowledgeBaseName,
    "--chunk-size",
    String(config.chunkSize),
    "--chunk-overlap",
    String(config.chunkOverlap),
    "--embedding-dimensions",
    String(config.embeddingDimensions),
    "--max-tool-output-chars",
    String(config.maxToolOutputChars),
    ...config.documents.flatMap((document) => ["--document", document]),
  ]) as BridgeInitPayload;

  if (!initPayload.success) {
    throw new Error(initPayload.error || "failed to initialize local RAG bridge");
  }

  const tools = initPayload.tools ?? [];
  for (const tool of tools) {
    registry.registerTool(new PythonRagTool(config, tool));
  }

  return {
    enabled: true,
    knowledgeBase: initPayload.knowledge_base,
    selectedDocuments: initPayload.selected_documents ?? [],
    tools: tools.map((tool) => tool.function.name).sort(),
  };
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
  const command = splitCommand(config.pythonCommand);
  if (command.length === 0) throw new Error("RAG python command is empty");
  const [bin, ...prefixArgs] = command;
  const result = spawnSync(bin, [...prefixArgs, ...bridgeArgs(config, args)], {
    encoding: "utf8",
    cwd: resolve(dirname(config.bridgeScript), ".."),
    env: process.env,
    maxBuffer: 10 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  const payload = parseBridgeOutput(result.stdout || "", result.stderr || "");
  if (result.status !== 0) {
    const bridgeError =
      typeof payload === "object" && payload && "error" in payload
        ? String((payload as { error?: unknown }).error || "")
        : "";
    throw new Error(bridgeError || result.stderr || `bridge exited with status ${result.status}`);
  }
  return payload;
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

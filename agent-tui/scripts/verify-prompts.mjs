import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const vendored = resolve(root, "prompts/agent_system_prompt.yaml");

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function extractPrompt(file, mode) {
  const lines = readFileSync(file, "utf8").split(/\r?\n/);
  let inTargetTemplate = false;
  let inContent = false;
  const content = [];
  for (const line of lines) {
    if (line.startsWith("  - id: ")) {
      inTargetTemplate = false;
      inContent = false;
    }
    if (line.trim() === `mode: "${mode}"`) {
      inTargetTemplate = true;
      continue;
    }
    if (inTargetTemplate && line.trim() === "content: |") {
      inContent = true;
      continue;
    }
    if (inContent) {
      if (line === "" || line.startsWith("      ")) {
        content.push(line.startsWith("      ") ? line.slice(6) : line);
        continue;
      }
      break;
    }
  }
  if (content.length === 0) fail(`Prompt block not found for mode=${mode}`);
  return `${content.join("\n")}\n`;
}

if (!existsSync(vendored)) {
  fail(`Vendored prompt file missing: ${vendored}`);
}

const promptModule = await import("../dist/prompt.js");
const checks = [
  ["pure", promptModule.PURE_AGENT_SYSTEM_PROMPT],
  ["rag", promptModule.PROGRESSIVE_RAG_SYSTEM_PROMPT],
];

for (const [mode, actual] of checks) {
  const expected = extractPrompt(vendored, mode);
  if (Buffer.from(actual, "utf8").compare(Buffer.from(expected, "utf8")) !== 0) {
    fail(`Runtime prompt for mode=${mode} is not byte-for-byte identical to vendored YAML content block.`);
  }
}

process.stdout.write("Prompt verification passed: local vendored YAML and runtime prompt blocks are byte-for-byte identical.\n");

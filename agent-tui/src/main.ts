#!/usr/bin/env node
import { loadRuntimeConfig, printHelp } from "./config.js";
import { AgentTUI } from "./tui.js";

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
  process.stdout.write(printHelp());
  process.exit(0);
}

const runtime = loadRuntimeConfig(args);
if (!runtime.apiKey) {
  process.stderr.write("Missing API key. Pass --api-key, set LLM_API_KEY, or set OPENAI_API_KEY.\n");
  process.exit(1);
}

const app = new AgentTUI(runtime);
app.run();

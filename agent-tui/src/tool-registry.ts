import type { ChatTool, RuntimeTool, ToolResult } from "./types.js";

const TOOL_ERROR_HINT = "\n\n[Analyze the error above and try a different approach.]";

export class ToolRegistry {
  private tools = new Map<string, RuntimeTool>();

  registerTool(tool: RuntimeTool): void {
    const name = tool.name();
    if (this.tools.has(name)) return;
    this.tools.set(name, tool);
  }

  listTools(): string[] {
    return [...this.tools.keys()].sort();
  }

  hasTool(name: string): boolean {
    return this.tools.has(name);
  }

  getFunctionDefinitions(): ChatTool[] {
    return this.listTools().map((name) => {
      const tool = this.tools.get(name);
      if (!tool) throw new Error(`tool not found: ${name}`);
      return {
        type: "function",
        function: {
          name: tool.name(),
          description: tool.description(),
          parameters: tool.parameters(),
        },
      };
    });
  }

  async executeTool(name: string, args: unknown, signal?: AbortSignal): Promise<ToolResult> {
    const tool = this.tools.get(name);
    if (!tool) {
      return {
        success: false,
        output: "",
        error: `tool not found: ${name}${TOOL_ERROR_HINT}`,
      };
    }

    try {
      return await tool.execute(args, signal);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { success: false, output: "", error: message };
    }
  }

  async cleanup(): Promise<void> {
    for (const tool of this.tools.values()) {
      await tool.cleanup?.();
    }
  }
}

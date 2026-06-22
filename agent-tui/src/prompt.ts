import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { KnowledgeBaseInfo, RecentDocInfo, SelectedDocumentInfo } from "./types.js";

const PROMPT_FILE = resolve(dirname(fileURLToPath(import.meta.url)), "../prompts/agent_system_prompt.yaml");
const BLOCK_INDENT = "      ";

function loadPromptTemplate(mode: "pure" | "rag"): string {
  const lines = readFileSync(PROMPT_FILE, "utf8").split(/\r?\n/);
  let inTargetTemplate = false;
  let inContent = false;
  const content: string[] = [];

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
      if (line === "" || line.startsWith(BLOCK_INDENT)) {
        content.push(line.startsWith(BLOCK_INDENT) ? line.slice(BLOCK_INDENT.length) : line);
        continue;
      }
      break;
    }
  }

  if (content.length === 0) {
    throw new Error(`Prompt template not found for mode: ${mode}`);
  }
  return `${content.join("\n")}\n`;
}

export const PURE_AGENT_SYSTEM_PROMPT = loadPromptTemplate("pure");
export const PROGRESSIVE_RAG_SYSTEM_PROMPT = loadPromptTemplate("rag");

function formatFileSize(size: number): string {
  const kb = 1024;
  const mb = 1024 * kb;
  const gb = 1024 * mb;
  if (size < kb) return `${size} B`;
  if (size < mb) return `${(size / kb).toFixed(2)} KB`;
  if (size < gb) return `${(size / mb).toFixed(2)} MB`;
  return `${(size / gb).toFixed(2)} GB`;
}

function formatRFC3339(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hour = pad(date.getHours());
  const minute = pad(date.getMinutes());
  const second = pad(date.getSeconds());
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const offsetHour = pad(Math.floor(abs / 60));
  const offsetMinute = pad(abs % 60);
  return `${year}-${month}-${day}T${hour}:${minute}:${second}${sign}${offsetHour}:${offsetMinute}`;
}

function formatDocSummary(summary: string | undefined, maxLen: number): string {
  const cleaned = (summary ?? "").trim().replace(/[\n\r]/g, " ").replace(/\s+/g, " ");
  if (!cleaned) return "-";
  const runes = Array.from(cleaned);
  if (runes.length <= maxLen) return cleaned;
  return `${runes.slice(0, maxLen).join("").trim()}...`;
}

function xmlAttr(value: string | undefined): string {
  return (value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderFAQDoc(doc: RecentDocInfo): string {
  const question = doc.faq_standard_question || doc.file_name || "";
  const lines = [
    `<faq chunk_id="${doc.chunk_id ?? ""}" knowledge_id="${doc.knowledge_id}" created_at="${doc.created_at ?? ""}">`,
    `<question>${question}</question>`,
  ];
  for (const answer of doc.faq_answers ?? []) {
    lines.push(`<answer>${answer}</answer>`);
  }
  lines.push("</faq>");
  return lines.join("\n");
}

function renderRecentDoc(doc: RecentDocInfo): string {
  const name = doc.title || doc.file_name || "";
  const fileSize = formatFileSize(doc.file_size ?? 0);
  const lines = [
    `<document knowledge_id="${doc.knowledge_id}" type="${doc.type ?? ""}" file_size="${fileSize}" created_at="${doc.created_at ?? ""}">`,
    `<name>${name}</name>`,
  ];
  if (doc.description) {
    lines.push(`<summary>${formatDocSummary(doc.description, 120)}</summary>`);
  }
  lines.push("</document>");
  return lines.join("\n");
}

export function formatKnowledgeBaseList(kbInfos: KnowledgeBaseInfo[]): string {
  if (kbInfos.length === 0) return "<knowledge_bases />";

  const lines = ["<knowledge_bases>"];
  for (const kb of kbInfos) {
    const kbType = kb.type || "document";
    const capabilities = kb.capabilities?.length ? ` capabilities="${kb.capabilities.join(",")}"` : "";
    lines.push(`<knowledge_base id="${kb.id}" name="${kb.name}" type="${kbType}" doc_count="${kb.doc_count ?? 0}"${capabilities}>`);
    if (kb.description) lines.push(`<description>${kb.description}</description>`);

    const recentDocs = kb.recent_docs ?? [];
    if (recentDocs.length > 0) {
      if (kbType === "faq") {
        lines.push("<faq_entries>");
        for (const doc of recentDocs.slice(0, 10)) lines.push(renderFAQDoc(doc));
        lines.push("</faq_entries>");
      } else {
        lines.push("<recent_documents>");
        for (const doc of recentDocs.slice(0, 10)) lines.push(renderRecentDoc(doc));
        lines.push("</recent_documents>");
      }
    }
    lines.push("</knowledge_base>");
  }
  lines.push("</knowledge_bases>");
  return lines.join("\n");
}

function renderPromptPlaceholders(template: string, knowledgeBases: KnowledgeBaseInfo[]): string {
  let result = template;
  if (result.includes("{{knowledge_bases}}")) {
    const replacement =
      knowledgeBases.length === 0
        ? "(no knowledge bases bound to this session)"
        : "(see `<bound_knowledge_bases>` inside the user message's `<runtime_context>` for the current bound KB list and their capabilities)";
    result = result.replaceAll("{{knowledge_bases}}", replacement);
  }
  return result;
}

function formatSelectedDocuments(docs: SelectedDocumentInfo[]): string {
  if (docs.length === 0) return "";
  const lines = [
    "",
    "### User Selected Documents (via @ mention)",
    "The user has explicitly selected the following documents. **You should prioritize searching and retrieving information from these documents when answering.**",
    "Use `list_knowledge_chunks` with the provided Knowledge IDs to fetch their content.",
    "",
    "| # | Document Name | Type | Knowledge ID |",
    "|---|---------------|------|---------------|",
  ];

  docs.forEach((doc, index) => {
    const title = doc.title || doc.file_name || "";
    const fileType = doc.file_type || "-";
    lines.push(`| ${index + 1} | ${title} | ${fileType} | \`${doc.knowledge_id}\` |`);
  });
  lines.push("");
  return lines.join("\n");
}

export function buildSystemPrompt(params: {
  knowledgeBases: KnowledgeBaseInfo[];
  selectedDocs: SelectedDocumentInfo[];
  language?: string;
  systemPromptTemplate?: string;
}): string {
  const template =
    params.systemPromptTemplate && params.systemPromptTemplate.trim()
      ? params.systemPromptTemplate
      : params.knowledgeBases.length === 0
        ? PURE_AGENT_SYSTEM_PROMPT
        : PROGRESSIVE_RAG_SYSTEM_PROMPT;

  let prompt = renderPromptPlaceholders(template, params.knowledgeBases);
  prompt = prompt
    .replaceAll("{{current_time}}", formatRFC3339(new Date()))
    .replaceAll("{{language}}", params.language ?? "")
    .replaceAll("{{skills}}", "");

  if (params.selectedDocs.length > 0) {
    prompt += formatSelectedDocuments(params.selectedDocs);
  }

  return prompt;
}

function indentLines(text: string, indent: string): string {
  if (!text) return "";
  return text
    .split("\n")
    .map((line) => (line ? `${indent}${line}` : line))
    .join("\n");
}

export function buildRuntimeContextBlock(
  sessionID: string,
  knowledgeBases: KnowledgeBaseInfo[],
  selectedDocs: SelectedDocumentInfo[],
): string {
  const lines = [
    '<runtime_context note="turn metadata; follow communication_instruction and answer_instruction">',
    `  <current_time>${formatRFC3339(new Date())}</current_time>`,
    `  <session>${xmlAttr(sessionID)}</session>`,
  ];

  if (knowledgeBases.length > 0) {
    lines.push("  <bound_knowledge_bases>");
    lines.push(indentLines(formatKnowledgeBaseList(knowledgeBases), "    "));
    lines.push("  </bound_knowledge_bases>");
  }

  if (selectedDocs.length > 0) {
    lines.push('  <pinned_documents scope="authoritative_for_this_turn">');
    for (const doc of selectedDocs) {
      const title = doc.title || doc.file_name || doc.knowledge_id;
      lines.push(`    <document knowledge_id="${xmlAttr(doc.knowledge_id)}" title="${xmlAttr(title)}" />`);
    }
    lines.push("  </pinned_documents>");
    lines.push(
      "  <note>The pinned-document set above is authoritative for THIS turn. If an earlier turn in this conversation analysed a different document, do NOT reuse that analysis — re-query against the current scope.</note>",
    );
  }

  lines.push(
    '  <communication_instruction>Do not use internal tool names or identifiers in your answers or in Thought. Say "keyword retrieval" instead of grep_chunks, "semantic retrieval" instead of knowledge_search, "browse full document" instead of list_knowledge_chunks; likewise never expose chunk_id, knowledge_id, or other internal IDs—refer to documents by title or name.</communication_instruction>',
  );
  lines.push(
    "  <answer_instruction>When you have gathered enough information, write your complete user-facing answer as your reply and stop—do not request any more tools in that final message. Until then, keep using tools; do not give a partial answer mid-investigation.</answer_instruction>",
  );
  lines.push("</runtime_context>");
  return lines.join("\n");
}

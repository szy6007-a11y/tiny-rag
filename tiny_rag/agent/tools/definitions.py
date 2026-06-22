from __future__ import annotations

maxFunctionNameLength = 64

ToolThinking = "thinking"
ToolTodoWrite = "todo_write"
ToolGrepChunks = "grep_chunks"
ToolKnowledgeSearch = "knowledge_search"
ToolListKnowledgeChunks = "list_knowledge_chunks"
ToolGetDocumentInfo = "get_document_info"
ToolDatabaseQuery = "database_query"
ToolDataAnalysis = "data_analysis"
ToolDataSchema = "data_schema"
ToolExecuteSkillScript = "execute_skill_script"
ToolReadSkill = "read_skill"


def available_tool_definitions() -> list[dict[str, str]]:
    return [
        {"name": ToolGrepChunks, "label": "关键词搜索", "description": "快速定位包含特定关键词的文档和分块"},
        {"name": ToolKnowledgeSearch, "label": "语义搜索", "description": "理解问题并查找语义相关内容"},
        {"name": ToolListKnowledgeChunks, "label": "查看文档分块", "description": "获取文档完整分块内容"},
        {"name": ToolGetDocumentInfo, "label": "获取文档信息", "description": "查看文档元数据和分块范围"},
    ]


def default_allowed_tools() -> list[str]:
    return [
        ToolKnowledgeSearch,
        ToolGrepChunks,
        ToolListKnowledgeChunks,
        ToolGetDocumentInfo,
    ]

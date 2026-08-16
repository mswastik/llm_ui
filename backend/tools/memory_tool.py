"""
Memory tools for the agent (Phase 2).

Exposes memory_write / memory_read / memory_search / memory_delete to the LLM
so the agent can persist and retrieve durable facts across sessions. Search is
embedding-backed (reuses the RAG embedding pipeline) with a keyword fallback
when the embedding model is unavailable.
"""
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from database.memory_crud import (
    create_memory_entry, delete_memory_entry, get_memory_entry,
    list_memory_entries,
)
from database.models import get_db

MEMORY_WRITE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory_write",
        "description": (
            "Persist a durable fact or preference to long-term memory. Use for "
            "information that should be remembered ACROSS conversations (user "
            "preferences, project facts, decisions, learned tricks). Scope "
            "'global' for everything, 'agent:<name>' for agent-specific memory. "
            "Keep each entry concise and standalone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact or preference to remember (concise, standalone)"},
                "scope": {"type": "string", "description": "'global' or 'agent:<name>' (default 'global')"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"}
            },
            "required": ["content"]
        }
    }
}

MEMORY_READ_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory_read",
        "description": (
            "List recent memory entries. Optional scope filter "
            "('global', 'agent:<name>', or 'conversation:<id>')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Scope filter (optional)"},
                "limit": {"type": "integer", "description": "Max entries (default 50)"}
            }
        }
    }
}

MEMORY_SEARCH_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Semantic search over long-term memory entries. Use when you need "
            "to recall something specific from past sessions. Returns the most "
            "relevant entries with their IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "top_k": {"type": "integer", "description": "Max results (default 5)"}
            },
            "required": ["query"]
        }
    }
}

MEMORY_DELETE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "memory_delete",
        "description": "Delete a memory entry by its id (from memory_search/memory_read results).",
        "parameters": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "Memory entry id"}
            },
            "required": ["entry_id"]
        }
    }
}

MEMORY_TOOL_DEFINITIONS = [
    MEMORY_WRITE_DEFINITION, MEMORY_READ_DEFINITION,
    MEMORY_SEARCH_DEFINITION, MEMORY_DELETE_DEFINITION,
]


class MemoryTool:
    def __init__(self, rag_service=None):
        self.rag_service = rag_service  # optional; used for embeddings

    async def _embed(self, text: str):
        """Embedding via RAG pipeline; returns None if unavailable."""
        if not self.rag_service:
            return None
        try:
            return await self.rag_service._get_embedding(text)
        except Exception as e:
            print(f"[MEMORY] embedding failed: {e}")
            return None

    async def _search(self, query: str, top_k: int = 5, limit: int = 200) -> List[Dict]:
        async with get_db() as db:
            entries = await list_memory_entries(db, limit=limit)
        query_emb = await self._embed(query)

        if query_emb is not None:
            import numpy as np
            scored = []
            for e in entries:
                emb = await self._embed(e["content"])
                if emb is None:
                    continue
                score = float(np.dot(query_emb, emb) / (
                    (np.linalg.norm(query_emb) * np.linalg.norm(emb)) or 1.0))
                scored.append((score, e))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [e for _, e in scored[:max(1, min(top_k, len(scored)))]]
            if results:
                return results

        # Keyword fallback
        q_lower = query.lower()
        terms = [t for t in q_lower.split() if len(t) > 2]
        scored = []
        for e in entries:
            c = e["content"].lower()
            score = sum(1 for t in terms if t in c)
            if score:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncGenerator[Dict, None]:
        try:
            if tool_name == "memory_write":
                content = str(arguments.get("content", "")).strip()
                if not content:
                    yield {"type": "tool_error", "tool": tool_name, "error": "Empty content"}
                    return
                scope = str(arguments.get("scope") or "global").strip()
                if not scope.startswith(("global", "agent:", "conversation:")):
                    scope = "global"
                tags = arguments.get("tags") or []
                async with get_db() as db:
                    entry = await create_memory_entry(
                        db, content, scope=scope, tags=tags, source="manual"
                    )
                yield {"type": "tool_progress", "tool": tool_name, "status": "Saved to memory",
                       "progress": 100,
                       "result": {"entry_id": entry["id"], "scope": entry["scope"], "content": entry["content"]}}

            elif tool_name == "memory_read":
                scope = arguments.get("scope") or None
                limit = int(arguments.get("limit") or 50)
                async with get_db() as db:
                    entries = await list_memory_entries(db, scope=scope, limit=limit)
                result = {
                    "count": len(entries),
                    "entries": [{"id": e["id"], "scope": e["scope"], "content": e["content"],
                                 "source": e["source"], "tags": e["tags"]} for e in entries]
                }
                yield {"type": "tool_progress", "tool": tool_name, "status": f"{len(entries)} memory entries",
                       "progress": 100, "result": result}

            elif tool_name == "memory_search":
                query = str(arguments.get("query", "")).strip()
                top_k = int(arguments.get("top_k") or 5)
                if not query:
                    yield {"type": "tool_error", "tool": tool_name, "error": "Empty query"}
                    return
                entries = await self._search(query, top_k=top_k)
                result = {
                    "query": query,
                    "count": len(entries),
                    "entries": [{"id": e["id"], "scope": e["scope"], "content": e["content"],
                                 "source": e["source"]} for e in entries]
                }
                yield {"type": "tool_progress", "tool": tool_name, "status": f"{len(entries)} matches",
                       "progress": 100, "result": result}

            elif tool_name == "memory_delete":
                entry_id = str(arguments.get("entry_id", "")).strip()
                if not entry_id:
                    yield {"type": "tool_error", "tool": tool_name, "error": "Missing entry_id"}
                    return
                async with get_db() as db:
                    deleted = await delete_memory_entry(db, entry_id)
                if not deleted:
                    yield {"type": "tool_error", "tool": tool_name, "error": f"Entry {entry_id} not found"}
                    return
                yield {"type": "tool_progress", "tool": tool_name, "status": "Deleted",
                       "progress": 100, "result": {"deleted": True, "entry_id": entry_id}}
            else:
                yield {"type": "tool_error", "tool": tool_name, "error": f"Unknown memory tool: {tool_name}"}
        except Exception as e:
            yield {"type": "tool_error", "tool": tool_name, "error": str(e)}

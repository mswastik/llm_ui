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
    create_memory_entry, delete_memory_entry, find_near_duplicate,
    get_memory_entry, list_memory_entries,
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
            "Keep each entry concise and standalone. Duplicates are rejected — "
            "if the existing entry is outdated, memory_delete it first, then "
            "write the new fact. Never store secret values (API keys, passwords)."
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
            "Full-text search (FTS5, BM25, porter stemming) over long-term memory. "
            "No embedding model, no VRAM — instant. Use query for keywords/phrases, "
            "tags for explicit label filter (e.g. `project-x`)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "top_k": {"type": "integer", "description": "Max results (default 5)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tag filter — only entries containing at least one tag are returned"}
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
        self.rag_service = rag_service  # kept for compat, not used for memory (FTS5 now VRAM-free)

    async def _search(self, query: str, top_k: int = 5, limit: int = 200, tags: Optional[List[str]] = None) -> List[Dict]:
        """
        FTS5 full-text search (porter, BM25) — no embedding model, no VRAM.
        Benefits over embedding for memory:
        * No Qwen3-4B-Embedding load → 35B stays in VRAM → next turn KV hits
        * Instant, deterministic, no network, works offline
        * Good for small, curated facts (exact terms, tags, project names)
        * Embedding helps only for heavy paraphrase/synonym (e.g. “favourite colour”
          vs “I like blue”) which is rare for memory; FTS5 stemming covers most.
        * If you have large paraphrased memory and enough VRAM, set
          memory_search_use_embedding=true to re-enable semantic fallback.
        """
        from database.memory_crud import fts_search_memory
        from database.models import get_db as _get_db
        # Tag-aware FTS5 search (VRAM-free)
        try:
            async with _get_db() as db:
                results = await fts_search_memory(db, query, top_k=top_k, tags=tags)
            if results:
                return results
        except Exception as e:
            print(f"[MEMORY] FTS search failed, falling back: {e}")

        # Fallback: if FTS found nothing and embedding is explicitly enabled, try semantic
        try:
            from backend.settings import settings_manager as _sm
            use_emb = bool(_sm.get_settings().get("memory_search_use_embedding", False))
        except Exception:
            use_emb = False
        if not use_emb:
            return []
        # Embedding fallback (loads Qwen3-4B-Embedding → evicts 35B on limited VRAM)
        if not self.rag_service:
            return []
        try:
            query_emb = await self.rag_service._get_embedding(query)
        except Exception as e:
            print(f"[MEMORY] embedding fallback failed: {e}")
            return []
        if query_emb is None:
            return []
        # Score remaining entries via cosine (costly, VRAM-heavy)
        async with _get_db() as db:
            from database.memory_crud import list_memory_entries as _list
            entries = await _list(db, limit=limit)
        if tags:
            tag_set = set(t.strip() for t in tags if t.strip())
            if tag_set:
                entries = [e for e in entries if tag_set & set(e.get("tags") or [])]
        import numpy as np
        scored = []
        for e in entries:
            try:
                emb = await self.rag_service._get_embedding(e["content"])
            except Exception:
                continue
            if emb is None:
                continue
            score = float(np.dot(query_emb, emb) / ((np.linalg.norm(query_emb) * np.linalg.norm(emb)) or 1.0))
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:max(1, min(top_k, len(scored)))]]

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
                    dup = await find_near_duplicate(db, content)
                    if dup:
                        yield {"type": "tool_progress", "tool": tool_name,
                               "status": "Skipped — near-duplicate already in memory", "progress": 100,
                               "result": {"existing_id": dup["id"], "existing_content": dup["content"],
                                          "hint": "To update an outdated fact: memory_delete the existing id, then memory_write the new one."}}
                        return
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
                tags = arguments.get("tags") or None
                if not query:
                    yield {"type": "tool_error", "tool": tool_name, "error": "Empty query"}
                    return
                entries = await self._search(query, top_k=top_k, tags=tags)
                result = {
                    "query": query,
                    "count": len(entries),
                    "entries": [{"id": e["id"], "scope": e["scope"], "content": e["content"],
                                 "source": e["source"], "tags": e["tags"]} for e in entries]
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

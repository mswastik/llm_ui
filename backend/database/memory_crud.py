"""
Async CRUD helpers for the agent memory store (Phase 2).

Entries live in the `memory_entries` table with scope:
  'global' | 'agent:<id>' | 'conversation:<id>'
"""
from typing import Dict, List, Optional

from database.models import MemoryEntry

MAX_INJECTION_CHARS = 3000


async def _fts_sync_insert(db, entry_id: str, content: str, tags: List[str], scope: str):
    """Keep memory_fts in sync — best-effort, never raises."""
    try:
        from sqlalchemy import text
        tags_str = " ".join(t for t in (tags or []) if isinstance(t, str))
        await db.execute(
            text("INSERT OR REPLACE INTO memory_fts (id, content, tags, scope) VALUES (:id, :content, :tags, :scope)"),
            {"id": entry_id, "content": content or "", "tags": tags_str, "scope": scope or "global"},
        )
    except Exception as e:
        # FTS table may not exist yet (old DB before migration) — non-fatal
        print(f"[FTS] sync insert failed for {entry_id[:8]}: {e}")


async def _fts_sync_delete(db, entry_id: str):
    try:
        from sqlalchemy import text
        await db.execute(text("DELETE FROM memory_fts WHERE id = :id"), {"id": entry_id})
    except Exception as e:
        print(f"[FTS] sync delete failed for {entry_id[:8]}: {e}")


def _row(entry: MemoryEntry) -> Dict:
    return {
        "id": entry.id,
        "scope": entry.scope,
        "content": entry.content,
        "tags": entry.tags or [],
        "source": entry.source,
        "importance": entry.importance,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }


async def create_memory_entry(
    db, content: str, scope: str = "global", tags: Optional[List[str]] = None,
    source: str = "manual", importance: float = 0.5
) -> Dict:
    entry = MemoryEntry(
        content=content, scope=scope, tags=tags or [],
        source=source, importance=importance
    )
    db.add(entry)
    await db.flush()
    await _fts_sync_insert(db, entry.id, content, tags or [], scope)
    return _row(entry)


async def list_memory_entries(db, scope: Optional[str] = None, limit: int = 200) -> List[Dict]:
    from sqlalchemy import select
    stmt = select(MemoryEntry).order_by(MemoryEntry.updated_at.desc()).limit(limit)
    if scope:
        stmt = stmt.where(MemoryEntry.scope == scope)
    result = await db.execute(stmt)
    return [_row(e) for e in result.scalars().all()]


async def get_memory_entry(db, entry_id: str) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(MemoryEntry).where(MemoryEntry.id == entry_id))
    entry = result.scalar_one_or_none()
    return _row(entry) if entry else None


async def update_memory_entry(db, entry_id: str, **fields) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(MemoryEntry).where(MemoryEntry.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        return None
    for key in ("content", "scope", "tags", "importance"):
        if key in fields:
            setattr(entry, key, fields[key])
    await db.flush()
    await _fts_sync_insert(db, entry.id, entry.content, entry.tags or [], entry.scope)
    return _row(entry)


async def delete_memory_entry(db, entry_id: str) -> bool:
    from sqlalchemy import select
    result = await db.execute(select(MemoryEntry).where(MemoryEntry.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        return False
    await db.delete(entry)
    await db.flush()
    await _fts_sync_delete(db, entry_id)
    return True


async def get_memory_tags(db, agent_id: Optional[int] = None, conversation_id: Optional[str] = None) -> List[str]:
    """Return distinct tags in use (for injection as lightweight index)."""
    from sqlalchemy import select
    scopes = []
    if conversation_id:
        scopes.append(f"conversation:{conversation_id}")
    if agent_id is not None:
        scopes.append(f"agent:{agent_id}")
    scopes.append("global")
    tags: set = set()
    for scope in scopes:
        result = await db.execute(select(MemoryEntry).where(MemoryEntry.scope == scope))
        for e in result.scalars().all():
            for t in (e.tags or []):
                if isinstance(t, str) and t.strip():
                    tags.add(t.strip())
    return sorted(tags)


async def fts_search_memory(
    db, query: str, top_k: int = 5, tags: Optional[List[str]] = None, scope: Optional[str] = None
) -> List[Dict]:
    """
    FTS5 full-text search over memory_entries (VRAM-free, BM25 ranked).
    Replaces the embedding cosine search — no Qwen3-4B-Embedding model load,
    so the 35B main model stays in VRAM and KV cache stays hot.

    * `query` is tokenized with porter; FTS5 does stemming + ranking.
    * `tags` and `scope` are post-filtered in Python (memory is small).
    * Falls back to LIKE search if FTS table is missing/corrupt.
    """
    from sqlalchemy import text, select

    # Tag pre-filter via Python after FTS — keep FTS query simple
    tag_set = set(t.strip() for t in (tags or []) if t.strip()) if tags else None

    # Build FTS5 query: quote each term, OR them. Escape double quotes.
    terms = [t for t in query.split() if len(t) > 1]
    if not terms:
        terms = [query]
    fts_q = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms if t.strip())
    if not fts_q:
        return []

    try:
        # FTS5 bm25 ranking; search all indexed columns (content, tags)
        # Use a generous limit for post-filtering, then trim to top_k
        sql = text("SELECT id, rank FROM memory_fts WHERE memory_fts MATCH :q ORDER BY rank LIMIT :lim")
        result = await db.execute(sql, {"q": fts_q, "lim": max(top_k * 4, 20)})
        rows = result.fetchall()
        if not rows:
            # No FTS hits — fall back to LIKE (covers substring, edge tokenization)
            raise ValueError("no FTS hits")
        ids_in_rank_order = [r[0] for r in rows]
        # Fetch full entries in rank order
        stmt = select(MemoryEntry).where(MemoryEntry.id.in_(ids_in_rank_order))
        if scope:
            stmt = stmt.where(MemoryEntry.scope == scope)
        res = await db.execute(stmt)
        entries = {e.id: e for e in res.scalars().all()}
        ranked = []
        for _id in ids_in_rank_order:
            e = entries.get(_id)
            if not e:
                continue
            if tag_set and not (tag_set & set(e.tags or [])):
                continue
            if scope and e.scope != scope:
                continue
            ranked.append(_row(e))
            if len(ranked) >= top_k:
                break
        if ranked:
            return ranked
        # If post-filter removed everything, fall through to LIKE
        raise ValueError("post-filter empty")
    except Exception as e:
        # Fallback: Python LIKE / term counting (no VRAM, deterministic)
        # This also covers the case where memory_fts doesn't exist yet
        if "no FTS hits" not in str(e) and "post-filter" not in str(e):
            print(f"[FTS] search fallback (LIKE) due to: {e}")
        # Fallback to old python scoring (fast for <1k rows)
        entries = await list_memory_entries(db, limit=500)
        if tag_set:
            entries = [en for en in entries if tag_set & set(en.get("tags") or [])]
        if scope:
            entries = [en for en in entries if en.get("scope") == scope]
        q_lower = query.lower()
        q_terms = [t for t in q_lower.split() if len(t) > 2]
        scored = []
        for en in entries:
            c = en["content"].lower()
            # BM25-ish: count term hits, boost tag hits
            score = sum(1 for t in q_terms if t in c)
            # Also check tags text
            tag_text = " ".join(en.get("tags") or []).lower()
            score += sum(2 for t in q_terms if t in tag_text)
            if score:
                scored.append((score, en))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [en for _, en in scored[:top_k]]


async def get_memory_for_injection(
    db, agent_id: Optional[int] = None, conversation_id: Optional[str] = None,
    max_chars: int = MAX_INJECTION_CHARS
) -> str:
    """Build the compact memory block injected into the system prompt.

    For local single-user use we DO NOT dump full memory every turn (token waste
    + context bloat). Instead:
      - conversation-scoped entries are injected verbatim (short, task-relevant)
      - for agent/global: inject only a tag index + usage hint, model pulls
        via memory_search(query, tags=[...]) or memory_read when relevant.
    """
    from sqlalchemy import select
    chunks = []
    # Conversation-scoped memory is NOT injected verbatim (would change system
    # prompt every time that conversation gains a fact → KV miss for that
    # conversation's next turn, which on long 27k prompts costs 80s).
    # Keep system prompt static; the model can pull conversation facts via
    # memory_search/memory_read when needed, same as global.

    # Global/agent: static hint — never changes with counts/tags
    # (dynamic counts/tags caused KV cache miss on every memory write because
    # system prompt is at the very start of the prompt; any change there
    # invalidates the entire prefix. Keep the hint static so KV reuses.)
    # We still need to know if there is ANY memory to decide whether to inject.
    from sqlalchemy import func as _func
    has_any = False
    for scope in ([f"agent:{agent_id}"] if agent_id is not None else []) + ["global"]:
        r = await db.execute(select(_func.count()).select_from(MemoryEntry).where(MemoryEntry.scope == scope))
        if (r.scalar() or 0) > 0:
            has_any = True
            break
    if has_any:
        # Static — does not include counts or tag list, so KV stays hot
        hint = (
            "### Persistent memory is available.\n"
            "Use memory_search(query, top_k=5) to recall past facts, preferences, "
            "project details or decisions. You can also call memory_read(scope=\"global\", limit=20). "
            "Call memory_search when the user references a past preference, project, or person."
        )
        chunks.append(hint)

    return "\n\n".join(chunks) if chunks else ""

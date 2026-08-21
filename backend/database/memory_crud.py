"""
Async CRUD helpers for the agent memory store (Phase 2).

Entries live in the `memory_entries` table with scope:
  'global' | 'agent:<id>' | 'conversation:<id>'
"""
from typing import Dict, List, Optional

from database.models import MemoryEntry

MAX_INJECTION_CHARS = 3000


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
    return _row(entry)


async def delete_memory_entry(db, entry_id: str) -> bool:
    from sqlalchemy import select
    result = await db.execute(select(MemoryEntry).where(MemoryEntry.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        return False
    await db.delete(entry)
    await db.flush()
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
    # 1. Conversation scope: small, directly relevant — keep verbatim (capped 800 chars)
    if conversation_id:
        result = await db.execute(
            select(MemoryEntry)
            .where(MemoryEntry.scope == f"conversation:{conversation_id}")
            .order_by(MemoryEntry.importance.desc(), MemoryEntry.updated_at.desc())
        )
        entries = result.scalars().all()
        if entries:
            lines = [f"- {e.content}" for e in entries]
            block = "### Memory (this conversation):\n" + "\n".join(lines)
            # cap conversation block separately
            if len(block) > 800:
                block = block[:800] + " …"
            chunks.append(block)

    # 2. Global/agent: expose tag index, not full dump
    tags = await get_memory_tags(db, agent_id=agent_id, conversation_id=None)
    # count per scope for hint
    counts = {}
    for scope in ([f"agent:{agent_id}"] if agent_id is not None else []) + ["global"]:
        r = await db.execute(select(MemoryEntry).where(MemoryEntry.scope == scope))
        counts[scope] = len(r.scalars().all())
    total = sum(counts.values())
    if total > 0:
        tag_line = ", ".join(f"`{t}`" for t in tags[:20]) if tags else "(no tags yet)"
        if len(tags) > 20:
            tag_line += f" +{len(tags)-20} more"
        scope_detail = f" ({counts.get('global',0)} global"
        if agent_id is not None:
            scope_detail += f", {counts.get(f'agent:{agent_id}',0)} for this agent"
        scope_detail += ")"
        hint = (
            f"### Persistent memory: {total} entries{scope_detail}\n"
            + f"Tags in use: {tag_line}\n"
            + "Use memory_search(query, top_k=5) for semantic search. "
            + "You can also call memory_read(scope=\"global\", limit=20) and filter by tags. "
            + "Call memory_search whenever the user references a past preference, project, or person."
        )
        chunks.append(hint)

    return "\n\n".join(chunks) if chunks else ""

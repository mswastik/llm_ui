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


async def get_memory_for_injection(
    db, agent_id: Optional[int] = None, conversation_id: Optional[str] = None,
    max_chars: int = MAX_INJECTION_CHARS
) -> str:
    """Build the memory block injected into the system prompt.

    Order of precedence: conversation-scoped, then agent-scoped, then global.
    Scopes with no entries are skipped; total output is capped at max_chars.
    """
    from sqlalchemy import select
    scopes = []
    if conversation_id:
        scopes.append(f"conversation:{conversation_id}")
    if agent_id is not None:
        scopes.append(f"agent:{agent_id}")
    scopes.append("global")

    chunks = []
    used = 0
    for scope in scopes:
        stmt = (
            select(MemoryEntry)
            .where(MemoryEntry.scope == scope)
            .order_by(MemoryEntry.importance.desc(), MemoryEntry.updated_at.desc())
        )
        result = await db.execute(stmt)
        entries = result.scalars().all()
        if not entries:
            continue
        lines = [f"- {e.content}" for e in entries]
        block = f"### Memory ({scope}):\n" + "\n".join(lines)
        if used + len(block) > max_chars:
            room = max_chars - used
            if room > 80:
                chunks.append(block[:room])
            used = max_chars
            break
        chunks.append(block)
        used += len(block)

    return "\n\n".join(chunks) if chunks else ""

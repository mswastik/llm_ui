"""
CRUD for LLM providers (multi-provider support).

A provider is an OpenAI-compatible endpoint (base_url + optional api_key).
Models are auto-fetched from the provider and cached in the `models` column.
"""
from typing import Any, Dict, List, Optional

from database.models import LLMProvider


def _row(p: LLMProvider, include_api_key: bool = False) -> Dict:
    row = {
        "id": p.id,
        "name": p.name,
        "base_url": p.base_url,
        "models": p.models or [],
        "enabled": p.enabled,
        "is_default": p.is_default,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }
    if include_api_key:
        row["api_key"] = p.api_key
    return row


async def create_provider(db, name: str, base_url: str, api_key: Optional[str] = None,
                          models: Optional[List[Dict]] = None,
                          is_default: int = 0, enabled: int = 1) -> Dict:
    provider = LLMProvider(name=name, base_url=base_url, api_key=api_key,
                           models=models or [], is_default=is_default, enabled=enabled)
    db.add(provider)
    await db.flush()
    return _row(provider)


async def list_providers(db, include_api_key: bool = False) -> List[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(LLMProvider).order_by(LLMProvider.is_default.desc(), LLMProvider.created_at))
    return [_row(p, include_api_key=include_api_key) for p in result.scalars().all()]


async def get_provider(db, provider_id: str, include_api_key: bool = False) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    p = result.scalar_one_or_none()
    return _row(p, include_api_key=include_api_key) if p else None


async def get_provider_by_name(db, name: str) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(LLMProvider).where(LLMProvider.name == name))
    p = result.scalar_one_or_none()
    return _row(p) if p else None


async def get_default_provider(db, include_api_key: bool = False) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(LLMProvider).where(LLMProvider.is_default == 1).limit(1))
    p = result.scalar_one_or_none()
    if p is None:
        result = await db.execute(select(LLMProvider).where(LLMProvider.enabled == 1).limit(1))
        p = result.scalar_one_or_none()
    return _row(p, include_api_key=include_api_key) if p else None


async def update_provider(db, provider_id: str, **fields) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    p = result.scalar_one_or_none()
    if not p:
        return None
    for key in ("name", "base_url", "api_key", "models", "enabled", "is_default"):
        if key in fields:
            setattr(p, key, fields[key])
    await db.flush()
    return _row(p)


async def set_default_provider(db, provider_id: str) -> Optional[Dict]:
    """Unset the current default and make provider_id the default."""
    from sqlalchemy import select, update
    await db.execute(update(LLMProvider).values(is_default=0))
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    p = result.scalar_one_or_none()
    if not p:
        return None
    p.is_default = 1
    p.enabled = 1
    await db.flush()
    return _row(p)


async def delete_provider(db, provider_id: str) -> bool:
    from sqlalchemy import select
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    p = result.scalar_one_or_none()
    if not p:
        return False
    await db.delete(p)
    await db.flush()
    return True

"""
CRUD for the skill usage log (Phase 4).

skill_runs records every load_skill execution so the reflection pass can
propose improvements when a skill underperforms.
"""
from typing import Dict, List, Optional

from database.models import SkillRun


async def record_skill_run(db, skill_name: str, conversation_id: Optional[str] = None,
                           success: bool = True, user_correction: Optional[str] = None) -> Dict:
    row = SkillRun(
        skill_name=skill_name,
        conversation_id=conversation_id,
        success=1 if success else 0,
        user_correction=user_correction,
    )
    db.add(row)
    await db.flush()
    return {
        "id": row.id,
        "skill_name": row.skill_name,
        "conversation_id": row.conversation_id,
        "success": row.success,
        "user_correction": row.user_correction,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def list_skill_runs(db, skill_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
    from sqlalchemy import select
    stmt = select(SkillRun).order_by(SkillRun.created_at.desc()).limit(limit)
    if skill_name:
        stmt = stmt.where(SkillRun.skill_name == skill_name)
    result = await db.execute(stmt)
    rows = []
    for r in result.scalars().all():
        rows.append({
            "id": r.id,
            "skill_name": r.skill_name,
            "conversation_id": r.conversation_id,
            "success": r.success,
            "user_correction": r.user_correction,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return rows

"""
CRUD for on-demand job runs (Phase 5).

job_runs records every run_job execution: status, params, output file,
and the conversation the job ran in.
"""
from typing import Dict, List, Optional

from database.models import JobRun


async def create_job_run(db, job_name: str, params: Optional[dict] = None,
                         conversation_id: Optional[str] = None) -> Dict:
    row = JobRun(
        job_name=job_name,
        params=params or {},
        conversation_id=conversation_id,
    )
    db.add(row)
    await db.flush()
    return _row(row)


async def list_job_runs(db, limit: int = 50) -> List[Dict]:
    from sqlalchemy import select
    stmt = select(JobRun).order_by(JobRun.started_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return [_row(r) for r in result.scalars().all()]


async def get_job_run(db, run_id: str) -> Optional[Dict]:
    from sqlalchemy import select
    result = await db.execute(select(JobRun).where(JobRun.id == run_id))
    row = result.scalar_one_or_none()
    return _row(row) if row else None


async def finish_job_run(db, run_id: str, status: str,
                         output_path: Optional[str] = None,
                         error: Optional[str] = None) -> Optional[Dict]:
    from sqlalchemy import select
    from datetime import datetime
    result = await db.execute(select(JobRun).where(JobRun.id == run_id))
    row = result.scalar_one_or_none()
    if not row:
        return None
    row.status = status
    row.finished_at = datetime.utcnow()
    if output_path:
        row.output_path = output_path
    if error:
        row.error = error
    await db.flush()
    return _row(row)


def _row(r: JobRun) -> Dict:
    return {
        "id": r.id,
        "job_name": r.job_name,
        "params": r.params or {},
        "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "output_path": r.output_path,
        "conversation_id": r.conversation_id,
        "error": r.error,
    }

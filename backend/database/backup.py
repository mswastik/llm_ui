"""
Database backup service with automatic scheduled backups.
"""

import asyncio
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.settings import settings_manager


def _resolve_db_path() -> Optional[str]:
    """Resolve the actual SQLite database file path from the database URL."""
    db_url = settings_manager.settings.database_url
    # sqlite+aiosqlite:///./llm_ui.db -> ./llm_ui.db
    match = re.match(r'sqlite(?:\+[a-z]+)?:///(.+)', db_url)
    if match:
        path = match.group(1)
        return os.path.abspath(path)
    return None


def _backup_filename() -> str:
    """Generate a timestamped backup filename."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"llm_ui_backup_{ts}.db"


def _cleanup_old_backups(backup_dir: str, max_keep: int):
    """Remove oldest backup files exceeding max_keep count."""
    try:
        files = sorted(
            [f for f in Path(backup_dir).iterdir() if f.name.startswith("llm_ui_backup_") and f.suffix == ".db"],
            key=lambda f: f.stat().st_mtime
        )
        while len(files) > max_keep:
            oldest = files.pop(0)
            oldest.unlink(missing_ok=True)
    except Exception as e:
        print(f"[BACKUP] Cleanup error: {e}")


def _get_backup_metadata_path(backup_dir: str) -> str:
    return os.path.join(backup_dir, "backup_metadata.json")


def _load_metadata(backup_dir: str) -> dict:
    meta_path = _get_backup_metadata_path(backup_dir)
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_metadata(backup_dir: str, metadata: dict):
    meta_path = _get_backup_metadata_path(backup_dir)
    try:
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[BACKUP] Metadata save error: {e}")


def backup_database() -> dict:
    """Perform a single database backup. Returns result dict."""
    db_path = _resolve_db_path()
    if not db_path or not os.path.exists(db_path):
        return {"success": False, "error": "Database file not found"}

    backup_dir = settings_manager.settings.backup_path
    max_keep = settings_manager.settings.backup_max_keep

    try:
        os.makedirs(backup_dir, exist_ok=True)
        dest = os.path.join(backup_dir, _backup_filename())
        shutil.copy2(db_path, dest)

        size_bytes = os.path.getsize(dest)
        _cleanup_old_backups(backup_dir, max_keep)

        metadata = _load_metadata(backup_dir)
        metadata["last_backup"] = datetime.now(timezone.utc).isoformat()
        metadata["last_backup_file"] = os.path.basename(dest)
        metadata["last_backup_size"] = size_bytes
        _save_metadata(backup_dir, metadata)

        print(f"[BACKUP] Created: {dest} ({size_bytes / 1024:.1f} KB)")
        return {"success": True, "file": dest, "size": size_bytes}
    except Exception as e:
        print(f"[BACKUP] Error: {e}")
        return {"success": False, "error": str(e)}


def get_backup_status() -> dict:
    """Get backup status information."""
    backup_dir = settings_manager.settings.backup_path
    metadata = {}

    if os.path.exists(backup_dir):
        metadata = _load_metadata(backup_dir)

    backup_files = []
    if os.path.exists(backup_dir):
        for f in sorted(Path(backup_dir).iterdir()):
            if f.name.startswith("llm_ui_backup_") and f.suffix == ".db":
                backup_files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat()
                })

    return {
        "enabled": settings_manager.settings.backup_enabled,
        "path": settings_manager.settings.backup_path,
        "interval_hours": settings_manager.settings.backup_interval_hours,
        "max_keep": settings_manager.settings.backup_max_keep,
        "last_backup": metadata.get("last_backup"),
        "last_backup_file": metadata.get("last_backup_file"),
        "last_backup_size": metadata.get("last_backup_size"),
        "backup_files": sorted(backup_files, key=lambda x: x["modified"], reverse=True)
    }


class BackupScheduler:
    """Background scheduler for automatic database backups."""

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self):
        """Start the backup scheduler background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print("[BACKUP] Scheduler started")

    async def stop(self):
        """Stop the backup scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        print("[BACKUP] Scheduler stopped")

    def restart(self):
        """Restart the scheduler (e.g., after settings change)."""
        if self._task:
            self._task.cancel()
        self._running = False
        self.start()

    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                if settings_manager.settings.backup_enabled:
                    interval = settings_manager.settings.backup_interval_hours * 3600
                    result = backup_database()
                    if not result.get("success"):
                        print(f"[BACKUP] Scheduled backup failed: {result.get('error')}")
                else:
                    interval = 3600  # Check every hour if backup gets enabled
            except Exception as e:
                print(f"[BACKUP] Scheduler error: {e}")
                interval = 3600

            # Sleep in small increments to allow clean shutdown
            for _ in range(max(1, interval // 5)):
                if not self._running:
                    return
                await asyncio.sleep(5)


# Global scheduler instance
backup_scheduler = BackupScheduler()

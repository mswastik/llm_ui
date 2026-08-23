"""Regression check for the backup scheduler:
- restart() must fully stop the old loop before starting a new one
  (no overlapping scheduler tasks → no same-time backup pile-ups).
- _run_loop must NOT back up instantly on start; the first backup
  only happens after the full interval has elapsed.

Deterministic: `asyncio.sleep` is stubbed to a bare yield, and the number of
elapsed 5s-chunks is counted rather than measuring real time, so the check is
flakiness-free.

Run: PYTHONPATH=. python tests/test_backup_scheduler.py
"""

import asyncio
import unittest
from unittest.mock import patch

from backend.database import backup as backup_mod
from backend.database.backup import BackupScheduler

REAL_SLEEP = asyncio.sleep
CHUNKS_PER_INTERVAL = 50  # interval size in 5s chunks


class BackupSchedulerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Shrink the interval: CHUNKS_PER_INTERVAL chunks instead of 4320
        backup_mod.settings_manager.settings.backup_interval_hours = \
            CHUNKS_PER_INTERVAL * 5 / 3600
        backup_mod.settings_manager.settings.backup_enabled = True
        self.scheduler = BackupScheduler()
        self.chunks = [0]

    def _patch_loop(self):
        """Replace the 5s chunk sleep with a one-shot yield, counting chunks."""
        async def yield_sleep(delay):
            self.chunks[0] += 1
            await REAL_SLEEP(0)
        return patch.object(asyncio, "sleep", yield_sleep)

    async def test_restart_leaves_single_live_loop(self):
        calls = []
        with self._patch_loop(), \
             patch.object(backup_mod, "backup_database",
                          side_effect=lambda: calls.append(1) or {"success": True}), \
             patch.object(backup_mod, "_load_metadata", lambda _: {}):
            self.scheduler.start()
            tasks = [self.scheduler._task]
            for _ in range(3):
                await self.scheduler.restart()
                tasks.append(self.scheduler._task)

            # Every previous loop must have finished; exactly one lives on
            self.assertTrue(all(t.done() for t in tasks[:-1]))
            self.assertFalse(tasks[-1].done())
            self.assertEqual(len(calls), 0, "restart must not fire a backup")
            await self.scheduler.stop()

    async def test_first_backup_only_after_interval(self):
        calls = []
        fired_at = []
        def fake_backup():
            fired_at.append(self.chunks[0])
            calls.append(1)
            return {"success": True}
        with self._patch_loop(), patch.object(backup_mod, "backup_database", fake_backup), \
             patch.object(backup_mod, "_load_metadata", lambda _: {}):
            self.scheduler.start()
            await REAL_SLEEP(0)  # let the scheduler run a few chunks
            # We are still well short of the interval → no backup yet
            self.assertLess(self.chunks[0], CHUNKS_PER_INTERVAL)
            self.assertEqual(len(calls), 0, "no backup before interval elapses")

            # Let the loop spin past the interval → backups fire, but every
            # single one only after a full interval of chunks has elapsed
            await REAL_SLEEP(0.05)
            self.assertGreater(len(calls), 0, "backups fire after the interval")
            self.assertTrue(all(c >= CHUNKS_PER_INTERVAL for c in fired_at),
                            "backup fired before the interval elapsed")
            await self.scheduler.stop()



if __name__ == "__main__":
    unittest.main(verbosity=2)
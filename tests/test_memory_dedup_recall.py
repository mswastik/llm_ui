"""Regression checks for the memory curation overhaul:

1. find_near_duplicate catches rephrased/re-run copies of a saved fact and
   leaves unrelated facts alone — regenerations cannot re-add the same memory.
2. create_memory_entry_dedup skips the duplicate (returning the existing row)
   and saves genuinely new content.
3. get_recall_for_query returns a <relevant_memories> block for matching user
   turns and "" when nothing is relevant.
4. _fts_sync_insert keeps exactly ONE index row per entry across updates
   (FTS5 has no unique key, so INSERT OR REPLACE used to accumulate stale rows).

Run: PYTHONPATH=backend:. python tests/test_memory_dedup_recall.py
"""

import asyncio
import os
import sys
import tempfile
import unittest

# Point the app at a throwaway DB BEFORE database.models builds its engine.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_backend = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, _backend)

import settings  # noqa: E402

settings.DATABASE_URL = f"sqlite+aiosqlite:///{tempfile.mktemp(suffix='_llmui_memtest.db')}"

from database.memory_crud import (  # noqa: E402
    create_memory_entry,
    create_memory_entry_dedup,
    find_near_duplicate,
    fts_search_memory,
    get_recall_for_query,
    update_memory_entry,
)
from database.models import get_db, init_db  # noqa: E402


class TestMemoryDedupRecall(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()

    async def test_find_near_duplicate(self):
        async with get_db() as db:
            await create_memory_entry(
                db, "Kokoro af_heart at speed 0.9 is the approved narration voice",
                scope="global", tags=["tts"], source="manual",
            )
            await db.commit()

        async with get_db() as db:
            # Rephrased near-copy of the same fact → duplicate detected
            dup = await find_near_duplicate(
                db, "Kokoro af_heart voice at speed 0.9 is the approved narration for videos"
            )
            self.assertIsNotNone(dup)
            # Unrelated fact → no duplicate
            none = await find_near_duplicate(
                db, "User runs a supply chain analytics company called Verit"
            )
            self.assertIsNone(none)

    async def test_create_dedup_skips_and_saves(self):
        async with get_db() as db:
            e1, dup1 = await create_memory_entry_dedup(
                db, "Buffer renders em-dash as unicode so always use double hyphen",
                scope="global", tags=["buffer"], source="auto",
            )
            await db.commit()
            self.assertIsNotNone(e1)
            self.assertIsNone(dup1)

            # Same fact, minor reword → skipped, existing entry returned
            e2, dup2 = await create_memory_entry_dedup(
                db, "Buffer API renders em-dash as unicode — always use double hyphen in posts",
                scope="global", tags=["buffer"], source="auto",
            )
            await db.commit()
            self.assertIsNone(e2)
            self.assertEqual(dup2["id"], e1["id"])

    async def test_recall_block(self):
        async with get_db() as db:
            await create_memory_entry(
                db, "Investment portfolio target allocation is 60 percent equity 30 debt 10 gold",
                scope="global", tags=["finance"], source="manual",
            )
            await db.commit()

        async with get_db() as db:
            block = await get_recall_for_query(db, "what is my portfolio allocation target again?")
            self.assertIn("<relevant_memories>", block)
            self.assertIn("60 percent equity", block)
            empty = await get_recall_for_query(db, "zzz nonexistent qqq wwww")
            self.assertEqual(empty, "")

    async def test_fts_one_row_per_entry_after_update(self):
        async with get_db() as db:
            entry = await create_memory_entry(
                db, "Original fact about the staging deployment server",
                scope="global", source="manual",
            )
            await db.commit()
        async with get_db() as db:
            await update_memory_entry(db, entry["id"], content="Updated fact about the staging deployment server")
            await db.commit()
        async with get_db() as db:
            from sqlalchemy import text
            n = (await db.execute(text("SELECT count(*) FROM memory_fts WHERE id = :i"), {"i": entry["id"]})).scalar()
            self.assertEqual(n, 1)
            hits = await fts_search_memory(db, "Updated fact staging deployment", top_k=5)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["content"], "Updated fact about the staging deployment server")
            # Stale content must no longer be findable
            stale = await fts_search_memory(db, "Original fact staging deployment", top_k=5)
            self.assertEqual([h for h in stale if h["id"] == entry["id"] and "Original" in h["content"]], [])


if __name__ == "__main__":
    unittest.main()

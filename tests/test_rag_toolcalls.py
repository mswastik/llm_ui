"""Regression checks for:
1. float16 embedding storage — new writes are half-size, old float32 rows
   still read correctly, and cosine similarity is unaffected in practice.
2. Slimmed tool_calls column — full tool results live in metadata.blocks only,
   not duplicated in the tool_calls column.

Run: PYTHONPATH=backend python tests/test_rag_toolcalls.py
"""

import asyncio
import sqlite3
import tempfile
import unittest
import uuid

import numpy as np

from tools.rag_service import EmbeddingStore, Chunk


class Float16EmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = f"{self.tmp}/test.db"
        # EmbeddingStore queries LEFT JOIN documents for the filename
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT)")
        conn.commit()
        conn.close()
        self.store = EmbeddingStore(self.db_path)

    def test_float16_write_and_legacy_float32_read(self):
        rng = np.random.default_rng(42)
        doc_id = str(uuid.uuid4())
        chunks = [
            Chunk(content="alpha beta gamma", start_char=0, end_char=17, section="s1"),
            Chunk(content="delta epsilon zeta", start_char=18, end_char=35, section="s1"),
        ]
        vecs = [rng.normal(size=2560).astype(np.float32) for _ in chunks]

        # Real write path: stores float16 blobs (half of float32 = 5120 bytes)
        self.store.store_chunks(doc_id, chunks, vecs)
        conn = sqlite3.connect(self.db_path)
        sizes = {r[0]: len(r[1]) for r in conn.execute(
            "SELECT chunk_id, embedding FROM document_embeddings")}
        conn.close()
        self.assertTrue(all(s == 2560 * 2 for s in sizes.values()),
                        f"expected float16 blobs (5120B), got {set(sizes.values())}")

        # Simulate a legacy row still stored as float32 (10 KB blob)
        legacy = rng.normal(size=2560).astype(np.float32)
        conn = sqlite3.connect(self.db_path)
        legacy_id = str(uuid.uuid4())
        conn.execute("INSERT INTO document_chunks (id, document_id, chunk_index, content, section) "
                     "VALUES (?, ?, 99, 'legacy row', 's1')", (legacy_id, doc_id))
        conn.execute("INSERT INTO document_embeddings (chunk_id, embedding) VALUES (?, ?)",
                     (legacy_id, legacy.tobytes()))
        conn.commit()
        conn.close()

        # Query against one of the stored vectors — both dtypes must be found
        q = vecs[0]
        results = self.store.search_similar(q, top_k=10, document_ids=[doc_id])
        found = {r["chunk_id"]: r["similarity"] for r in results}
        self.assertIn(legacy_id, found, "legacy float32 row must still be searchable")

        # Both stored dtypes must reproduce the exact float32 cosine
        def exact_cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        self.assertAlmostEqual(found[legacy_id], exact_cos(q, legacy), delta=1e-6,
                               msg="legacy float32 cosine must be exact")
        # The self-match (float16) must still rank ~1.0 — precision loss is minor
        self.assertAlmostEqual(max(found.values()), 1.0, delta=0.02,
                               msg="float16 cosine drift is too large")


class SlimToolCallsTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_calls_column_kept_slim(self):
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from database.models import Base, Message
        from database.crud import add_message
        from sqlalchemy import select

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        big_result = "x" * 100_000  # simulate a large tool payload
        blocks = [
            {"type": "tool_call", "name": "query_documents", "arguments": {"q": "test"},
             "result": big_result, "sources": ["src1"], "progress_history": [{"p": 50}]},
            {"type": "content", "content": "done"},
        ]
        async with Session() as db:
            saved = await add_message(db, "conv-1", "assistant", "done", blocks=blocks,
                                      extra_metadata={"model": "test"})
            self.assertEqual(saved["tool_calls"][0]["name"], "query_documents")
            self.assertNotIn("result", saved["tool_calls"][0],
                             "tool_calls column must not duplicate the result")
            self.assertNotIn("progress_history", saved["tool_calls"][0])
            self.assertEqual(saved["metadata"]["blocks"][0]["result"], big_result,
                             "full payload stays in metadata.blocks")

            # Reload from DB — the stored column is slim, blocks are intact
            row = (await db.execute(
                select(Message).where(Message.conversation_id == "conv-1"))).scalar_one()
            self.assertNotIn("result", row.tool_calls[0])
            self.assertEqual(row.extra_metadata["blocks"][0]["result"], big_result)
        await engine.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)

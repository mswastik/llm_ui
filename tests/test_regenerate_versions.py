"""Regression checks for regeneration + versioning:

1. Regenerating an assistant answer must NOT leak the previous version (or any
   later turn) into the LLM context — history replay is cut at the user prompt
   being re-answered, and every stored version of the response group is
   excluded.
2. All versions of a response share the superseded message's timeline slot
   (turn_index), so a mid-thread regeneration stays in place after reload.
3. Version numbers derive from MAX(version) of the group, so regenerating via a
   stale message id cannot mint duplicate (group, version) pairs.

Run: PYTHONPATH=backend:. python tests/test_regenerate_versions.py
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

settings.DATABASE_URL = f"sqlite+aiosqlite:///{tempfile.mktemp(suffix='_llmui_regtest.db')}"


class TestRegenerateVersioning(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import database.models as models

        self.models = models
        await models.init_db()

        from database.crud import create_conversation, add_message

        async with models.get_db() as db:
            conv = await create_conversation(db, "regtest")
            self.cid = conv["id"]
            u1 = await add_message(db, self.cid, "user", "question 1")
            a1 = await add_message(db, self.cid, "assistant", "answer v1")
            await db.commit()
        self.u1, self.a1 = u1, a1

    async def test_full_regen_flow(self):
        # ── First regeneration (mid-thread scenario: later turns exist) ──
        # Stamp v1 into a group and simulate an existing v2 + later turns,
        # exactly as a previously buggy run would have left them.
        import uuid as uuid_lib

        from sqlalchemy import select

        from database.crud import add_message, get_conversation_messages
        from database.models import Message

        async with self.models.get_db() as db:
            row = (
                await db.execute(select(Message).where(Message.id == self.a1["id"]))
            ).scalar_one_or_none()
            row.version_group = group = str(uuid_lib.uuid4())
            await db.commit()
            await add_message(db, self.cid, "assistant", "answer v2",
                              version=2, version_group=group,
                              turn_index=row.turn_index)
            await add_message(db, self.cid, "user", "question 2")
            await add_message(db, self.cid, "assistant", "later answer")
            await db.commit()

        from httpx import ASGITransport, AsyncClient

        from app.main import _core_stream_handler, app, llm_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            r = await client.post(
                f"/api/conversations/{self.cid}/regenerate",
                json={"message_id": self.a1["id"]},
            )
            self.assertEqual(r.status_code, 200)
            d = r.json()

        # Anchor points at the user prompt being re-answered; slot inherited.
        self.assertEqual(d["anchor_user_message_id"], self.u1["id"])
        self.assertEqual(d["version"], 3)          # MAX(version)=2 → 3
        self.assertEqual(d["version_group"], group)
        self.assertEqual(d["turn_index"], self.a1["turn_index"])

        # ── Context cut: stub the LLM and run the real stream handler ──
        calls = []

        async def fake_stream(messages, **kw):
            calls.append(list(messages))
            yield {"type": "content", "content": "fresh answer"}

        llm_client.stream_chat = fake_stream
        events = []
        async for ev in _core_stream_handler(
                "req-regtest", self.cid,
                enable_rag=False, model="stub-model",
                version=d["version"], version_group=d["version_group"],
                anchor_message_id=d["anchor_user_message_id"],
                turn_index=d["turn_index"]):
            events.append(ev)

        replay = calls[0]
        blob = "".join(str(m.get("content")) for m in replay)
        body = [m for m in replay if m["role"] != "system"]
        self.assertTrue(body, "history must still contain the anchor prompt")
        self.assertTrue(all(m["role"] == "user" for m in body),
                        "replay must stop at the anchor user message")
        self.assertIn("question 1", blob)
        self.assertNotIn("answer v1", blob, "prior version leaked into context")
        self.assertNotIn("answer v2", blob, "prior version leaked into context")
        self.assertNotIn("question 2", blob, "later turn leaked into context")
        self.assertNotIn("later answer", blob, "later turn leaked into context")

        # The streamed answer was persisted into the inherited slot, v3.
        async with self.models.get_db() as db:
            latest = await get_conversation_messages(db, self.cid)
            fresh = next(m for m in latest if m["content"] == "fresh answer")
            self.assertEqual(fresh["version"], 3)
            self.assertEqual(fresh["version_group"], group)
            self.assertEqual(fresh["turn_index"], self.a1["turn_index"])
            self.assertEqual([m["content"] for m in latest], [
                "question 1", "fresh answer", "question 2", "later answer"])

    async def test_turn_indices_are_ordered_slots(self):
        from database.crud import (
            add_message, create_conversation, get_conversation_messages)

        async with self.models.get_db() as db:
            conv = await create_conversation(db, "slots")
            cid = conv["id"]
            ids = []
            for role, text in [("user", "q"), ("assistant", "a"),
                               ("user", "q2"), ("assistant", "a2")]:
                m = await add_message(db, cid, role, text)
                ids.append(m["turn_index"])
            await db.commit()
        self.assertTrue(ids[0] < ids[1] < ids[2] < ids[3], ids)

    async def test_delete_specific_version(self):
        """DELETE /api/messages/{id}?version=N removes exactly that version row,
        resolved via the group from the displayed representative's id."""
        import uuid as uuid_lib

        from sqlalchemy import select

        from database.crud import add_message
        from database.models import Message
        from httpx import ASGITransport, AsyncClient

        async with self.models.get_db() as db:
            row = (
                await db.execute(select(Message).where(Message.id == self.a1["id"]))
            ).scalar_one_or_none()
            row.version_group = group = str(uuid_lib.uuid4())
            await db.commit()
            await add_message(db, self.cid, "assistant", "answer v2",
                              version=2, version_group=group,
                              turn_index=row.turn_index)
        from httpx import ASGITransport, AsyncClient

        from app.main import app
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
        async with client:
            # Delete v1 using only the representative id + ?version=1
            r = await client.delete(f"/api/messages/{self.a1['id']}?version=1")
            self.assertEqual(r.status_code, 200, r.text)
            # Unknown versions must 404
            r2 = await client.delete(f"/api/messages/{self.a1['id']}?version=99")
            self.assertEqual(r2.status_code, 404)

        async with self.models.get_db() as db:
            rows = (await db.execute(
                select(Message).where(Message.version_group == group))).scalars().all()
            self.assertEqual([m.content for m in rows], ["answer v2"])

    async def test_crash_midstream_still_persists_and_emits_done(self):
        """If the LLM stream raises mid-generation, the handler must persist the
        partial output and terminate with error + done — never close silently
        (that left clients spinning forever with unpatched placeholder ids)."""
        import json as _json

        from app.main import _core_stream_handler, llm_client
        from database.crud import add_message, get_conversation_messages

        async with self.models.get_db() as db:
            await add_message(db, self.cid, "user", "hello")
            await db.commit()

        async def exploding_stream(messages, **kw):
            yield {"type": "content", "content": "partial an"}
            raise RuntimeError("connection reset mid-stream")

        llm_client.stream_chat = exploding_stream

        events = []
        async for ev in _core_stream_handler(
                "req-crash", self.cid,
                enable_rag=False, model="stub-model",
                version=1, version_group=None,
                anchor_message_id=None, turn_index=None):
            events.append(ev)

        kinds = [_json.loads(e[6:]).get("type") for e in events if e.startswith("data: ")]
        self.assertIn("error", kinds)
        self.assertEqual(kinds[-1], "done", "stream must always end with done")
        done = _json.loads([e for e in events if e.startswith("data: ")][-1][6:])
        self.assertTrue(done.get("message_id"), done)

        async with self.models.get_db() as db:
            rows = await get_conversation_messages(db, self.cid)
        partial = [m for m in rows if m["content"] == "partial an"]
        self.assertTrue(partial, "partial generation must be persisted on crash")
        self.assertEqual(partial[0]["id"], done["message_id"])

if __name__ == "__main__":
    unittest.main(verbosity=2)

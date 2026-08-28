---
name: llm-ui-data-cleanup
description: Audit and safely clean up llm_ui app data: duplicate threads, orphaned uploads, stale agents/MCP configs, skill-run telemetry, with backup + verification steps.
---

<!-- reflection reason: Multi-step, repeatable procedure spanning DB, files, agents, MCP config, and skills with specific gotchas (WAL-safe backup, content-level dup detection, orphan rescan after deletes, chunk→embedding FK order). The existing conversation-audit skill covers only thread dedup and references a tools/conversation_audit.py script that does not exist; this session's full cleanup workflow deserves its own skill. -->

1. Safety first: snapshot the DB before any deletion — `sqlite3.connect('llm_ui.db')` then `dst.backup()` to the OneDrive backups folder as `llm_ui_PRE-CLEANUP_<date>.db` (handles WAL correctly; do NOT file-copy).
2. Inventory: list tables/row counts in llm_ui.db (conversations, messages, agents, mcp_servers, documents, document_chunks, document_embeddings, skill_runs, memory_entries); `du -sh uploads/ skills/ models/ outputs/`.
3. Duplicate threads: compare FIRST USER MESSAGE content (difflib ratio ≥0.9), not titles (forks/retries have different titles). Keep the higher depth score (assistant_chars/200 + assistant_msgs*15). Also flag `Forked:` chats whose original was deleted.
4. Orphaned uploads: regex-scan every upload filename against messages (content, thinking, tool_calls, metadata), documents.filepath, memory_entries, conversation titles. Orphans are usually leftover TTS .wav files — report count/size and non-audio orphans separately before deleting.
5. Agent/MCP config: delete agents with is_active=0 AND 0 attached conversations; remove entries from agents.enabled_mcp_servers that are not in mcp_servers.name (stale refs); report servers enabled-but-unreferenced (wasted startup processes) and referenced-but-disabled (broken agent features). Do NOT 'fix' profile dir names that look wrong — check ~/.browser-mcp/profiles.json mapping first.
6. Skill dir/name mismatches: compare each skills/<dir>/SKILL.md frontmatter `name:` to its directory name; recommend rename, never delete (agents reference the frontmatter name).
7. Memory: difflib near-dup scan (>0.85) of memory_entries.content; report groups for user review.
8. Present findings as tiers: Tier1 zero-risk (orphan files, skill_runs telemetry, failed documents rows + their chunks/embeddings via `document_embeddings.chunk_id IN (SELECT id FROM document_chunks WHERE document_id=?)`), Tier2 duplicates (keep/delete table), Tier3 user judgment. Never delete without explicit approval.
9. Execute in order: DB deletes (messages before conversations) → re-run orphan scan AFTER thread deletions (threads may free more files) → delete files → archive terminal_audit.jsonl → `VACUUM` → verify: PRAGMA integrity_check, orphan-message count 0, KEPT twins still present, final counts/sizes.

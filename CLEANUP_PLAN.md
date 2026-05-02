# Codebase Cleanup Plan

Generated after refactoring review of llm_ui.

---

## 1. Delete Dead Files

| # | File | Reason |
|---|------|--------|
| 1 | `frontend/static/css/styles.css` | Never loaded in any template. Overlaps with `theme.css`. |
| 2 | `migrate_db.py` | Schema is now managed by SQLAlchemy `init_db()`. One-shot migration, no longer needed. |
| 3 | `backend/tools/clear_tts_cache.py` | Orphaned utility script. Never imported, no CLI entry point. |
| 4 | `graphify-out/` (entire directory) | Build artifact from graphify skill. Add to `.gitignore`. |
| 5 | `backend/__init__.py` | Empty file. Not used as a package (all imports are direct file imports). |
| 6 | `backend/app/__init__.py` | Empty file. Same reason. |
| 7 | `backend/llm_client/__init__.py` | Contains only a comment, no exports. |
| 8 | `backend/mcp_client/__init__.py` | Contains only a comment, no exports. |

---

## 2. Fix Broken / Missing Imports

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| 1 | `backend/app/main.py` | 944 | `_agent_to_dict()` uses `Dict` in type hint but `Dict` is not imported (only `List, Optional` from `typing`). | Add `Dict` to the `from typing import` on line 7. |

---

## 3. Remove Unused Functions & Imports

| # | File | Item | Reason |
|---|------|------|--------|
| 1 | `backend/database/crud.py` | `get_default_agent()` (line 582) | Never called anywhere. |
| 2 | `backend/database/crud.py` | `hard_delete_agent()` (line 571) | Only soft-delete is used. |
| 3 | `backend/settings.py` | `get_config()` (line 294) | Backward-compat function, never called. |
| 4 | `backend/app/main.py` | Import `get_default_agent` (line 21) | Dead import — function above is dead. |
| 5 | `backend/app/main.py` | Import `get_all_mcp_servers` (line 473, local) | Should be at top with other CRUD imports. |
| 6 | `backend/app/main.py` | Import `QUERY_MODEL` (line 402, inside function) | Should be at module level. |

---

## 4. Remove Dead Code in Functions

| # | File | Item | Reason |
|---|------|------|--------|
| 1 | `backend/tools/base.py` | `SharedLLMUtils.cosine_similarity()` (line 84) | Defined but never called. `EmbeddingStore.search_similar()` computes cosine similarity inline. |
| 2 | `backend/tools/rag_service.py` | `RAGConfig.embeddings_api` (line 52) | Set in `__post_init__` but never read. |
| 3 | `backend/tools/rag_service.py` | `RAGConfig.rerank_api` (line 53) | Set in `__post_init__` but never read. |

---

## 5. Consolidate Duplicate Logic

| # | File A | File B | Issue | Recommendation |
|---|--------|--------|-------|----------------|
| 1 | `backend/tools/base.py:13` | `backend/tools/rag_service.py:48` | `SharedLLMUtils._get_base_url()` default = `http://localhost:8001/v3`; `RAGConfig.__post_init__` default = `http://localhost:8080/v1` (different port AND path). | Use a single source of truth (e.g., `settings.py` or `SharedLLMUtils`). Remove the duplicate from `RAGConfig`. |
| 2 | `frontend/static/css/styles.css` | `frontend/static/css/theme.css` | Both define: scrollbar styles, `[x-cloak]`, `.prose`, `pre`/`code`, `.citation-link`. | Delete `styles.css` (Section 1). Move its unique-only classes (`message-action-btn`, `copy-success`, `code-copy-btn`, `stop-btn-pulse`, `tts-playing`, `line-clamp-2`, `.toast-enter`) into `theme.css`. |

---

## 6. Minor Code Quality

| # | File | Line | Issue | Recommendation |
|---|------|------|-------|----------------|
| 1 | `backend/app/main.py` | 473 | `get_all_mcp_servers` imported inside endpoint function | Move to top-level imports with other CRUD functions |
| 2 | `backend/app/main.py` | 402 | `QUERY_MODEL` imported inside `_core_stream_handler` loop | Move to module-level import (line 12) |

---

## Execution Order

1. **Fix the broken import** (Section 2, #1) — `Dict` in `main.py` — this is a runtime bug
2. **Delete dead files** (Section 1)
3. **Remove unused functions & imports** (Section 3)
4. **Remove dead code inside functions** (Section 4)
5. **Consolidate duplicates** (Section 5)
6. **Move CSS-only classes** from `styles.css` to `theme.css` before deleting `styles.css`
7. **Verify the app still runs** — start the server and test all endpoints

---

## Files to Touch

| File | Action |
|------|--------|
| `frontend/static/css/styles.css` | Delete (after moving unique classes) |
| `migrate_db.py` | Delete |
| `backend/tools/clear_tts_cache.py` | Delete |
| `graphify-out/` | Delete or add to `.gitignore` |
| `backend/__init__.py` | Delete |
| `backend/app/__init__.py` | Delete |
| `backend/llm_client/__init__.py` | Delete |
| `backend/mcp_client/__init__.py` | Delete |
| `backend/app/main.py` | Fix import, move imports up, remove dead import |
| `backend/database/crud.py` | Remove `get_default_agent`, `hard_delete_agent` |
| `backend/settings.py` | Remove `get_config` |
| `backend/tools/base.py` | Remove `cosine_similarity` |
| `backend/tools/rag_service.py` | Remove `embeddings_api` and `rerank_api` from `RAGConfig` |
| `frontend/static/css/theme.css` | Add unique classes from `styles.css` |

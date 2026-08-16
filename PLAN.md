# PLAN: Upgrading llm_ui into an on-demand local agent platform

Status: Draft v1 · 2026-08-15 · No code changed yet.

## 1. Goal

Turn the existing llm_ui (FastAPI + Alpine.js chat app) into a local, on-demand
AI agent — an alternative to Hermes Agent for someone who does **not** need
24/7 availability (no gateway, no messaging platforms, no cron service).

Must-have agent capabilities, in priority order:

1. Execute commands in the system — with dangerous-command blocking, a
   command allowlist, a directory allowlist, and interactive approval.
2. Persistent memory across sessions (auto + explicit).
3. Agent skills (SKILL.md format) — loadable on demand, user-created and
   agent-created.
4. Self-improvement: the agent writes/improves skills from experience.
5. Jobs platform: on-demand runbooks (e.g. fetch news, social media
   engagement analysis, social post creation) created later by asking the
   app — designed as a platform so they plug in without core changes.

## 2. Non-goals (explicit user decisions)

- No 24/7 gateway, no messaging-platform bots, no cross-platform handoff.
- No sandboxing (no Docker/SSH/Modal) — safety comes from blocklist +
  allowlist + approval instead.
- No cron scheduler now; the design leaves a scheduler interface ready.
- No image generation, GUI computer_use, provider routing/failover,
  multi-profile isolation.

## 3. Current state (verified)

The agent loop already exists in `backend/app/main.py`:

- `_core_stream_handler()` (line ~258): loads conversation + agent config →
  builds system prompt → assembles messages → loops: stream LLM chunks,
  parse `pending_tool_calls`, execute via `tool_executor.execute_tool()`
  (line ~621), append tool result, repeat.
- Agent binding: `enabled_tools` / `enabled_mcp_servers` filter custom and
  MCP tools (`main.py:479-501`).
- Custom tools: `query_documents` (RAG), `generate_speech` (TTS) — dispatched
  in `backend/tools/tool_executor.py:execute_tool()`, definitions in
  `get_tool_definitions()`.
- MCP client: stdio / SSE / streamable-http (`MCPClientManager`), tools
  auto-discovered and exposed to the LLM.
- RAG: embeddings + chunking + retrieval already implemented
  (`backend/tools/rag_service.py`, raw-SQLite embedding store).
- STT already implemented (`backend/tools/stt_service.py`,
  `/api/stt/transcribe`).
- Background-task pattern exists: `BackupScheduler` (`database/backup.py`) —
  asyncio loop started in lifespan, restartable.
- DB migrations: `init_db()` with ALTER TABLE fallback (`models.py`).
- Frontend: Alpine stores defined in `templates/base.html`, components in
  `frontend/static/js/components/*.js`, modals with `settingsTab` pattern in
  `templates/partials/settings_modal.html`, tool-call blocks rendered in
  `main_chat.html`, SSE events handled in `components/chat.js:processStreamEvent()`.

## 4. Architecture principle

The app is already 80% of an agent. The upgrade adds **three primitives**
(terminal, memory, skills) and **two pipelines** (self-improvement, jobs) on
top of the existing stream loop. Dependency chain:

```
jobs (Phase 5)  →  skills (Phase 3)  →  terminal (Phase 1)
                 ↘  memory (Phase 2)  ↗
```

Each phase unlocks the next; jobs and self-improvement are thin once
terminal + memory + skills exist.

## 5. Native vs MCP decision

Rule: implement **natively** when the feature needs app internals (stream
loop, DB, UI, safety enforcement). Connect an **MCP server** when a
third-party server already does the job with zero code and no safety
conflicts.

| Feature | Decision | Rationale |
|---|---|---|
| Terminal execution | **Native** (`run_command`) | The safety model (blocklist + allowlist + dirs + pause-for-approval) must be app-owned. The app is a generic MCP client — it cannot pause an MCP tool call mid-execution for approval, and third-party terminal servers (e.g. Desktop Commander) enforce their own permission model designed for other clients. |
| Memory | **Native** store | App already has embeddings; memory needs system-prompt injection (stream-handler hook), auto-extraction (post-turn hook), per-agent scoping, and a management UI — all app-side. MCP memory servers are tool-call-only (no injection, no UI, data in their own store). |
| Skills | **Native** loader | Skill index injection + `load_skill` + drafts + approval + UI are app concerns. The SKILL.md format is open, so skills from the ecosystem are drop-in without any MCP involvement. |
| Jobs | **Native** | Thin layer over skills + `job_runs` table + UI. Not MCP. |
| Context panel | **Native** | Needs stream-handler internals (what the model sees). |
| Conversation search | **Native** (FTS5) | In-process SQLite; no server needed. |
| Web search | **MCP** (already connected) | Done. |
| Browser automation | **MCP** (Playwright/Puppeteer) | Mature third-party servers; zero code. |
| Filesystem read/write | **MCP** (official `server-filesystem`) | Safer than shell for file ops; recommended optional alongside `run_command`. |
| Structured reasoning | **MCP** (`server-sequential-thinking`) | Free reasoning boost, zero code. |
| Platform APIs (social/news) | **MCP** | Jobs that need platform access (engagement analysis, posting) connect platform-specific servers; jobs orchestrate them. |
| Memory quick-start (optional) | **MCP** (official `server-memory`, `basic-memory`) | Zero-code stopgap before Phase 2 ships; documented but not the platform path. |
| Skill registry import (later phase) | **MCP** (SkillMCP / Skill.sh-MCP) or fetch | Optional Phase 4.5; discovery only — loading stays native. |

Default posture: connect MCP servers for *capability*, keep *safety,
state, and context assembly* native.

## 6. Phase 1 — Terminal execution + safety (foundation) — ✅ DONE (2026-08-15)

> Implementation notes:
> - `backend/tools/terminal_tool.py` (new): `TerminalTool` + `ApprovalManager` +
>   `TERMINAL_TOOL_DEFINITION`. Layers: hard blocklist (HARD_BLOCKED_PATTERNS,
>   merged with settings, never removable) → directory allowlist → binary
>   allowlist → interactive approval. `cd <dir> && cmd` prefixes are parsed and
>   the dir goes through the same allowlist. Output truncated at 100 KB.
> - Approval gate is **pre-registered before the SSE event is yielded** —
>   removes a race where a fast decision arrived before the gate existed.
> - Audit log: `terminal_audit.jsonl` — verdicts: blocked / pending / approved /
>   denied / executed / timeout / cancelled.
> - `run_command` registered in `ToolExecutor` (`get_tool_definitions` +
>   `execute_tool`), per-agent filtering works through the existing
>   `enabled_tools` mechanism (`main.py:479-501`); `execute_tool` now takes
>   `call_key` so parallel tool calls in one turn get distinct approval keys.
> - Endpoint: `POST /api/tools/{request_id}/approve` (`main.py`).
> - Frontend: `tool_approval_required` SSE case + `respondApproval()` in
>   `components/chat.js`; "Needs approval" badge + Approve/Deny buttons +
>   command preview in `partials/main_chat.html`.
>
> ⚠️ Discovery (pre-existing bug, fixed): `_core_stream_handler` silently
> swallowed `{"type":"error"}` chunks from `llm_client` — an LLM outage or
> invalid model produced an empty "done" with no message saved, looking like a
> silent empty reply. The chunk loop now surfaces the error to the UI.
> ⚠️ Discovery: settings default model id `qwen36-35B` does not exist on the
> llama.cpp server (models are e.g. `Qwen3.6-35B-General`); the UI always
> sends the model param so the browser path is fine, but API/curl users must
> pass `?model=<valid-id>`. Consider fixing the default later.
>
> Verified: 19 unit-style checks (blocklist/allowlist/dir/approval/timeout/
> audit) + full SSE E2E via API (allowlisted run, approve flow, deny flow) +
> browser UI test (Approve button → execution → Done). Test conversations
> cleaned up.

New custom tool `run_command`:

- Args: `command` (string), `working_dir` (optional), `timeout` (default
  ~120s, cap ~600s).
- Registered in `ToolExecutor.get_tool_definitions()`, dispatched in
  `execute_tool()`; controlled per-agent through the existing
  `enabled_tools` filter (`main.py:479-501`).
- Run via `asyncio.create_subprocess_shell`, stdout/stderr captured and
  truncated (~100 KB) to protect context; progress streamed as
  `tool_progress` events.

Safety, four layers applied in order:

1. **Static blocklist** (hard-coded regexes, no user override): `rm -rf /`,
   `mkfs`, `dd if=...of=/dev`, fork bombs, `sudo`, `> /dev/sd*`,
   `chmod 777 /`, `curl|sh` style pipelines, shell reverse shells.
   Blocked → `tool_error` with reason surfaced in the tool block.
2. **Binary allowlist** (configurable): `python3`, `node`, `git`, `curl`,
   `grep`, `find`, `sed`, `awk`, `ls`, `cat`, `mkdir`, `cp`, `mv`,
   `date`, `jq`, … Binary not allowlisted → requires approval.
3. **Directory allowlist**: resolved `working_dir` (symlinks resolved)
   must be inside an allowed root (repo dir, `uploads/`, `outputs/`;
   configurable). Violation → blocked.
4. **Interactive approval**: commands failing 2–3 go to a pause. The SSE
   generator awaits an `asyncio.Event` keyed by `request_id`; the tool
   block renders Approve/Deny buttons; `POST /api/tools/{request_id}/approve`
   (or `/deny`) resumes or aborts. Reuse the disconnect-watcher pattern
   (`main.py:1644`) so a dropped client aborts the command.

Audit log: every command (approved or blocked) appended to
`terminal_audit.jsonl` — timestamp, command, working_dir, verdict,
conversation_id.

Settings additions (`settings.json` + `Settings` model + `_ENV_MAP`):
`terminal_allowed_dirs`, `terminal_allowed_commands`,
`terminal_blocked_patterns`, `terminal_require_approval` (default true),
`terminal_default_timeout`.

Caveat (stated, accepted): blocklist + approval protects against accidents
and off-script model behavior, not against a malicious local actor. That is
the agreed tradeoff for running unsandboxed.

## 7. Phase 2 — Persistent memory — ✅ DONE (2026-08-15)

> Implementation notes:
> - `memory_entries` table added to `models.py` — new tables are auto-created
>   by `create_all` on startup, so no ALTER migration was needed.
> - `backend/database/memory_crud.py` (new): CRUD + `get_memory_for_injection`
>   (conversation → agent → global precedence, capped at 3000 chars).
> - `backend/tools/memory_tool.py` (new): `memory_write` / `memory_read` /
>   `memory_search` (embedding-backed via RAG pipeline, keyword fallback) /
>   `memory_delete`. Registered in `ToolExecutor`; per-agent filtering works
>   through `enabled_tools`.
> - Injection: memory block appended to the system prompt in
>   `_core_stream_handler`.
> - Auto-extraction: `_extract_memory_from_exchange` runs every
>   `memory_auto_extract_interval` (default 3) assistant turns, extracts facts
>   with a low-token completion using the conversation's model, writes
>   `source=auto` entries. Wrapped in a 90s timeout; failures logged, never
>   propagated into the stream.
> - API: GET/POST `/api/memory`, PATCH/DELETE `/api/memory/{id}`.
> - Frontend: "Memory" tab in the settings modal (list/add/edit/delete,
>   global/all filter, auto-extract interval). `api.patch()` added to utils.
> - ⚠️ Discovery: `enabled_tools` filtering in `main.py` only excluded
>   `generate_speech`; `run_command` and the memory tools were therefore
>   always exposed even when an agent restricted tools. The filter now covers
>   all six custom tools.
> - ⚠️ Discovery (dev annoyance): ES module imports in `main.js` have no
>   cache-busting version query, so after editing a component file the browser
>   keeps serving the cached module. Hard-reload / disable cache to see
>   frontend changes; consider adding `?v=` to component imports later.
>
> Verified: CRUD API round-trip; injection E2E (new conversation answered
> "You prefer Rust" with zero mention in the prompt); auto-extraction E2E
> (auto entry appeared after the 3rd turn; model also used `memory_write`
> proactively); browser UI add → list render. Test data cleaned up.

New table `memory_entries`:

```
id, scope ('global'|'agent:<id>'|'conversation:<id>'), content (Text),
tags (JSON), source ('auto'|'manual'), importance (Float, default 0.5),
created_at, updated_at
```

Migration via existing `init_db()` ALTER TABLE pattern.

- **Read / injection**: in `_core_stream_handler`, after agent config load:
  fetch global + agent-scoped entries; small → append verbatim to system
  prompt; large → RAG-retrieve top-k via existing embeddings and inject.
  Scoped to `conversation:<id>` only when non-empty.
- **Write / extraction**: after each assistant turn, a QUERY_MODEL pass
  ("extract durable facts/preferences from this exchange") writes entries.
  Rate-limited (every N turns, default 3) to control cost. Explicit user
  "remember that …" → direct write.
- **LLM tools**: `memory_write`, `memory_search` (embedding-backed),
  `memory_read`, `memory_delete`. The agent can persist on its own — this
  is what self-improvement builds on.
- **UI**: new memory tab — browse, edit, delete, toggle auto-extraction.
- Optional quick-start before this phase ships: connect official
  `server-memory` or `basic-memory` MCP (zero code; no injection/UI).

## 8. Phase 3 — Skills — ✅ DONE (2026-08-15)

> Implementation notes:
> - `backend/tools/skills_tool.py` (new): SKILL.md frontmatter parser
>   (no yaml dependency), `list_skills` / `skill_index` / `get_skill` /
>   `write_skill` / `delete_skill`, plus Phase-4-ready `get_draft` /
>   `accept_draft` and `skills/_drafts/` support. Path traversal guarded via
>   `_safe_join`; names slugified.
> - Tools: `load_skill(name)` (returns full SKILL.md body + file manifest,
>   truncated at 60 KB) and `create_skill(name, description, instructions)`
>   (writes the skill; available next turn). Registered in `ToolExecutor`.
> - Injection: compact one-line-per-skill index appended to the system prompt
>   with a `load_skill` usage hint. Drafts are excluded from the index.
> - API: GET `/api/skills?include_drafts=`, GET/PUT/DELETE
>   `/api/skills/{name}`, POST `/api/skills`, POST
>   `/api/skills/drafts/{name}/accept`, DELETE `/api/skills/drafts/{name}`.
> - Frontend: new Skills modal (`partials/skills_modal.html` +
>   `components/skills.js` + store methods + sidebar button + `index.html`
>   include) following the app's modal pattern. Supports list / view /
>   create / edit / delete, and accept/reject for drafts.
> - ⚠️ Deviation from plan: per-agent `enabled_skills` toggle not yet added
>   to the Agents modal (deferred to Phase 6 with the other agent
>   capability toggles). Skills are currently available to all agents;
>   `enabled_tools` can still exclude them per agent.
> - ⚠️ Fix applied: bumped `?v=` cache-busting versions on `main.js`,
>   `utils.js`, `settings.js`, `chat.js`, `sidebar.js` imports + new
>   `skills.js` (this is the app's convention — the version was not being
>   bumped, which is the stale-module issue noted in Phase 2).
>
> Verified: CRUD API round-trip; E2E (model loaded `news-fetch` via
> `load_skill` and reported its first step, proving index injection + tool
> flow); browser modal opens from sidebar and lists skills. Test data
> cleaned up.

- **Discovery**: scan `skills/` at stream start, parse frontmatter.
  Compact index (name + one-line description) appended to the system
  prompt — a few lines, not full bodies.
- **`load_skill(name)` tool**: returns the full SKILL.md (+ file manifest)
  as a tool result when the LLM decides the skill applies. Lean context,
  same mechanism as Hermes.
- **`create_skill(name, description, content)` tool**: writes a skill dir
  (used by self-improvement and by explicit user requests).
- **UI**: skills modal — list installed skills, view SKILL.md, create/edit,
  toggle per-agent `enabled_skills`.

## 9. Phase 4 — Self-improvement (approval-first by design) — ✅ DONE (2026-08-15)

> Implementation notes:
> - `skill_runs` table (`models.py`) + `backend/database/skill_crud.py`:
>   every `load_skill` execution is logged (skill, conversation, success).
> - Reflection: `_maybe_reflect_and_propose_skill` in `main.py` runs after a
>   tool-using turn (same cadence as memory extraction), prompts the model to
>   output `{action: create|improve|none, name, description, instructions,
>   reason}` and writes a **draft** to `skills/_drafts/`. It never writes live
>   skills silently; "improve" is only accepted when the target skill exists.
> - Draft approval UI built in Phase 3 (Skills modal: Accept / Reject on
>   draft rows; `accept_draft` moves the dir into `skills/`, replacing an
>   existing skill of the same name).
> - Explicit path verified too: the model invoked the `create_skill` tool
>   directly when the user asked for a reusable procedure.
>
> ⚠️ Note: when a draft appears there is no toast/sidebar badge — the user
> sees it by opening the Skills modal. A small follow-up could add a badge.
>
> Verified: multi-step `run_command` task → reflection proposed
> `rename-files-extension` draft; accept moved it live; a later `load_skill`
> turn logged a `skill_runs` row (success=1). Cleanup done.

Design rule (addresses the user's #1 Hermes complaint — invisible background
work): the learning loop produces **proposals**, never silent changes.

1. **Explicit**: "create a skill for X" → `create_skill` tool; available
   next turn.
2. **Semi-auto reflection**: after a completed multi-tool task, a
   QUERY_MODEL pass (task summary + transcript + existing skill list)
   decides: create new skill / improve existing / nothing. Output lands in
   `skills/_drafts/` + UI notification. Accept → move to `skills/`;
   reject → delete. Never silent.
3. **Feedback loop**: skill usage logged (`skill_runs` table: skill_name,
   conversation_id, success, user_correction) → reflection proposes
   SKILL.md diffs, shown in UI for accept/reject.

Model caveat: skill-writing quality tracks model quality. Run reflection on
`QUERY_MODEL` or a stronger endpoint; `qwen3-4b` will write weak skills.

Phase 4.5 (optional, low effort): skill **import** from registries via MCP
(SkillMCP / Skill.sh-MCP) or plain fetch — discovery/install only; loading
stays native.

## 10. Phase 5 — Jobs platform (on-demand runbooks) — ✅ DONE (2026-08-15)

> Implementation notes:
> - `job_runs` table (`models.py`) + `backend/database/job_crud.py`:
>   create/list/finish with status `running | completed | failed`,
>   output_path, conversation_id.
> - `run_job(job, params)` tool (ToolExecutor): loads the job's skill
>   instructions, creates a tracked run, and tells the model to execute the
>   instructions with the available tools. The stream handler finalizes open
>   runs after the turn: writes `outputs/jobs/<run_id>.md` and marks
>   completed/failed.
> - `POST /api/jobs/run` (body: job, params, optional agent_id): runs a job
>   to completion inline — creates a "Job: <name>" conversation, picks a
>   model (`jobs_model` setting, else the first *loaded* model on the
>   llama.cpp server), runs the agent loop, writes the output file, returns
>   the run record + output. This is the future cron hook.
> - `GET /api/jobs` history endpoint; `outputs/` dir mounted at `/outputs`.
> - Frontend: Jobs modal (`partials/jobs_modal.html` + `components/jobs.js` +
>   store methods + sidebar button) — job picker with description, params
>   JSON, Run, history with status badges + output links + Re-run (same
>   params).
> - ⚠️ Known quirk: when the external endpoint runs a job, the model may
>   itself call the `run_job` tool (seeing the job instructions), creating a
>   second tracked run that the handler also finalizes. Both complete;
>   the endpoint's own run record is authoritative. Harmless redundancy.
> - ⚠️ Note: `POST /api/jobs/run` blocks until the job finishes (inline).
>   A background-task variant (or the later cron scheduler) should be added
>   before jobs become long-running; the `backup_scheduler` pattern applies.
>
> Verified: external `POST /api/jobs/run` completed with output file; in-chat
> `run_job` tool flow (run_job → run_command → finalize) completed and was
> logged; Jobs modal lists jobs + history with output links. Cleanup done.

A job = a skill with a defined input/output contract. The user's examples
(news fetch, social engagement analysis, social post creation) are all thin
skills orchestrating terminal + MCP tools + an analysis step. Jobs run in
the same stream loop, so they can call any MCP tool (search, platform APIs)
and `run_command`.

- **`run_job(name, params)` tool** → loads the job's skill, executes its
  instructions, streams progress, returns output.
- **`job_runs` table**: job_name, params (JSON), status, started_at,
  finished_at, output_path, conversation_id, error.
- **Jobs UI**: library list, run form, history (status, duration, output,
  link to the run's conversation), re-run with last params.
- **Outputs** to `outputs/`; rendered in chat like tool results; optional
  TTS-read on completion (TTS exists).
- **Future-proofing**: expose `POST /api/jobs/run` so jobs are triggerable
  externally. Cron later = thin scheduler on the `backup_scheduler`
  pattern — fires only while the app runs, no gateway needed.

## 11. Phase 6 — UX glue — ✅ DONE (2026-08-15)

> Implementation notes:
> - **Context panel**: `context_info` SSE event emitted by the stream
>   handler at the start of every turn (model, message count, tool list,
>   full system prompt incl. injected memory + skill index). Frontend:
>   collapsible "What the model sees" panel under the conversation title
>   (`chat.js` `contextInfo`/`contextOpen` + `main_chat.html`).
> - **Approval UI**: done in Phase 1 (command preview + Approve/Deny in the
>   tool block).
> - **Memory / Skills tabs + modals + Jobs modal**: done in Phases 2/3/5.
> - ⚠️ NOT done: per-agent capability toggles (`enable_terminal`,
>   `enable_memory`, `enabled_skills`, `auto_learn` columns on `agents`).
>   The existing `enabled_tools` list already gates all custom tools per
>   agent (fixed in Phase 2 to cover run_command + memory + skills + jobs),
>   so the dedicated toggles are cosmetic; left as future work.
> - ⚠️ Context panel shows the system prompt and tool list; token estimate
>   and RAG chunk count were not added (would need llama.cpp token counting;
>   low value vs. effort).

- **Context panel**: what the model sees — system prompt (incl. injected
  memory + skill index), RAG chunk count, tool list, token estimate.
  Collapsible; directly fixes the Hermes context-opacity complaint.
- **Approval UI** in tool blocks (Phase 1): pending command, Approve/Deny,
  audit trail link.
- New settings tabs / modals for memory, skills, jobs (follow the existing
  `settingsTab` + modal patterns).
- **Agent capability toggles**: `enable_terminal`, `enable_memory`,
  `enabled_skills`, `auto_learn` on the existing `agents` table — the Agent
  becomes a Hermes-style profile (persona + capability set).

## 12. Low-effort additions (recommended, beyond the core ask)

| # | Addition | Effort | Why |
|---|---|---|---|
| 1 | **FTS5 conversation search** — sidebar search box over messages | Small | Hermes's best session feature; biggest parity gap for least code |
| 2 | **Audit log** (Phase 1 ships it) | Tiny | Trust + debuggability |
| 3 | **Desktop notification on stream end** (Web Notification API) | Tiny (frontend) | Jobs feel attended without polling |
| 4 | **Slash commands** (`/new`, `/skills`, `/jobs`, `/resume`) | Small (frontend) | Agent feel |
| 5 | **Mic input button** — STT backend already exists | ✅ already present | Confirmed: mic button exists in the chat input (`main_chat.html`), STT wired via `/api/stt/transcribe`. Nothing to do. |
| 6 | **"Save this conversation as a skill"** — pick message range → draft | Small | Manual path into the learning loop; user-controlled |
| 7 | **Export conversation → markdown** | Tiny | Share job outputs / feed skills / hand off |
| 8 | **Skill registry import UI** (Phase 4.5) | Med | One-click install from the open ecosystem |

Skipped deliberately: screenshot/vision tool (local llama.cpp stack is
text-only), image generation, provider routing.

## 13. Data model additions

| Table | Columns | Phase |
|---|---|---|
| `memory_entries` | id, scope, scope_id, content, tags, source, importance, created_at, updated_at | 2 |
| `skill_runs` | id, skill_name, conversation_id, success, user_correction, created_at | 4 |
| `job_runs` | id, job_name, params, status, started_at, finished_at, output_path, conversation_id, error | 5 |
| (no table) | `terminal_audit.jsonl` — file log | 1 |
| (no table) | `skills/`, `skills/_drafts/`, `outputs/` — filesystem | 3–5 |

Agent table additions: `enable_terminal`, `enable_memory`, `enabled_skills`
(JSON), `auto_learn` (Integer).

Settings additions: `terminal_*` (Phase 1), `memory_auto_extract_interval`
(Phase 2), `skills_dir` (Phase 3), `outputs_dir` (Phase 5).

## 14. Sequencing & effort

| Phase | Effort | Depends on |
|---|---|---|
| 1 Terminal + safety | Med (backend + approval UI) | — |
| 2 Memory | Med | 1 (auto-extraction scripts) — can start parallel |
| 3 Skills | Med | — (can start parallel with 1) |
| 6 Context panel | Small | — (anytime) |
| 4 Self-improvement | Med–Large (model-quality dependent) | 2 + 3 |
| 5 Jobs | Small once 3 exists | 3 |

Recommended order: **1 → 2/3 (parallel) → 6 → 4 → 5**.

## 15. Risks

- **Approval pause** needs an async gate in the SSE generator (asyncio.Event
  keyed by request_id) plus a resume endpoint; the disconnect watcher must
  cancel the pending subprocess. Moderate, well-understood pattern.
- **Reflection quality** is model-bound; weak local models produce weak
  skills — mitigate with QUERY_MODEL/stronger endpoint for reflection and
  approval-first drafts.
- **Context growth**: memory injection + skill index + tool definitions
  compete for tokens. Keep skill index compact, memory injected as top-k
  only, and enforce the output truncation in `run_command`.
- **Security boundary** is accident-prevention, not adversarial (stated in
  Phase 1). Do not add a terminal MCP server as a bypass.

## 16. Verification approach (per phase)

- Phase 1: ✅ done 2026-08-15 — see §6 notes. Actual results: `ls`/`echo`
  execute; `rm -rf /`, `sudo`, `curl|sh`, `mkfs` blocked with reason;
  `ffmpeg`/`openssl` → approval pause → approve executes / deny aborts with
  "Command denied by user"; audit log has every verdict; browser UI renders
  command + Approve/Deny and completes the run.
- Phase 2: ✅ done 2026-08-15 — see §7 notes. Verified: memory injected into
  system prompt (model recalled "Rust preference" from a new conversation),
  auto-extraction fired after 3 turns, memory tools callable by the LLM, and
  the Settings → Memory tab adds/lists/edits/deletes entries.
- Phase 3: ✅ done 2026-08-15 — see §8 notes. Verified: skill index
  injected into the system prompt, `load_skill` tool returned the skill body,
  Skills modal lists/creates/edits skills.
- Phase 4: ✅ done 2026-08-15 — see §9 notes. Verified: reflection proposed a
  draft after a multi-tool task, Skills modal Accept moved it live, and
  `load_skill` usage was logged in `skill_runs`.
- Phase 5: ✅ done 2026-08-15 — see §10 notes. Verified: external
  `POST /api/jobs/run` completed with output file; in-chat `run_job` tool
  flow completed and was logged; Jobs modal lists jobs + history.
- Phase 6: ✅ done 2026-08-15 — see §11 notes. Verified: `context_info`
  event captured in the browser with the exact system prompt + 10 tools;
  panel renders it.

## 17. Post-plan additions (2026-08-15, same session)

### Multi-provider LLM support — ✅ DONE

- `llm_providers` table: name, base_url, api_key, cached `models` JSON,
  enabled, is_default. `backend/database/provider_crud.py` +
  `backend/tools/provider_service.py` (auto-fetch `/v1/models`).
- **Auto-fetch on connect**: `POST /api/providers` saves the provider then
  immediately fetches its models and returns the count. Refresh endpoint
  `POST /api/providers/{id}/refresh`, `PUT /api/providers/{id}` re-fetches
  on URL change, `POST /api/providers/{id}/default`, DELETE promotes the
  next provider if the default was deleted.
- **Bootstrap**: on startup, if no providers exist, a default provider is
  seeded from the legacy `llama_cpp_base_url` setting (existing installs
  keep working unchanged).
- `/api/models` now aggregates cached models across enabled providers and
  returns a `providers` list; each model is tagged `provider_id` /
  `provider_name`.
- `llm_client.stream_chat` accepts per-request `base_url`/`api_key`
  (Authorization: Bearer header); `_core_stream_handler` resolves the
  provider (agent's provider > requested provider_id > default) and threads
  it through the main stream, title generation, memory extraction and skill
  reflection. `send_message` / SSE stream / regenerate endpoints accept
  `provider_id`.
- `agents` table gained `provider_id` (column + migration); the stream
  handler honors it. ⚠️ Agents modal UI for picking a provider is a
  follow-up; agents without one use the default provider.
- Frontend: model selector groups models by provider (default badge);
  selection persists provider + model; Settings → **Providers** tab with
  add (auto-fetch) / refresh / set-default / delete and a model badge list.
- ⚠️ RAG embeddings/rerank still use the legacy `llama_cpp_base_url`
  settings (local embeddings); noted limitation.
- Verified: bootstrap created the default provider with 9 fetched models;
  adding a second provider fetched 9 models; chat via explicit provider_id
  streamed correctly; browser UI send through the provider returned a reply.
  Test provider/conversations cleaned up.

### Settings dialog redesign — ✅ DONE

- Modal is now a **fixed size** (1100×min(86vh, 900px), body scrolls) — no
  more resizing per tab.
- New **Terminal tab**: require-approval toggle, default timeout, audit log
  path, allowed directories / commands / extra blocked patterns editors
  (one per line, converted to arrays on save), plus a read-only
  **hard-coded blocked patterns** list (new `GET /api/terminal/blocked-patterns`).
- **Providers tab** (above), STT settings added to the TTS tab (engine /
  model / language / OpenAI key), `jobs_model` added to the LLM tab,
  General tab gained max upload size / upload dir / CORS / SQLAlchemy echo.
- Verified: allowlist edited in the UI and saved; `ffmpeg` added via UI ran
  without approval after a clean restart (see operational note below).

### ⚠️ Operational discovery: stale processes sharing the port

`run.py` uses `uvicorn.run(..., reload=True)`. The reload parent/worker pair
plus every stdio MCP subprocess (they inherit the listening socket fd) can
leave multiple processes holding the :8002 listener; kernel load-balancing
then delivers requests to a stale process with old in-memory settings. This
made a UI allowlist change appear broken until the server was restarted
cleanly. Fixes: restart the server after significant changes (hub restart
kills the tree), and consider `reload=False` for production. Root fix (close
fds in MCP child processes) is a follow-up.

### Settings consolidation (2026-08-15, same session) — ✅ DONE

- **Sidebar reduced to feature entry points**: Knowledge Base, Notes,
  Skills, Jobs, Settings. **AI Agents moved out of the sidebar** into the
  Settings dialog — it is persona/capability configuration, not a workspace.
- **Settings tabs now**: General, **LLM (merged with Providers)**, Agents,
  TTS, Terminal, MCP Servers, Backup, Memory.
- **LLM + Providers merged**: the separate Providers tab is gone; the LLM
  tab hosts the provider list (Add / refresh / set-default / delete +
  model badges) above the legacy default-model config (LLM server URL,
  default/query/embedding/reranking models, temperature, max tokens, jobs
  model). The providers panel shows whenever the LLM tab is active.
- **Agents as a settings tab**: agents logic extracted from the old modal
  into `components/agents_panel.js` + `partials/agents_panel.html` (nested
  Alpine component inside the Settings modal). The agent form gained a
  **Provider** select with models grouped by provider (default provider
  preselected); agent cards show `model · provider`. `agents_modal.html`,
  the `modalAgents` registration and the unused `openAgents/closeAgents`
  store methods were removed (clean cutover).
- ⚠️ Judgment call: Knowledge Base / Notes / Skills / Jobs stay in the
  sidebar — they are content/execution workspaces, not configuration.
  Moving them into Settings would bury primary features.
- Verified in browser: sidebar shows 5 entry points; tabs render correctly;
  LLM tab lists 3 providers (llama.cpp default, OpenCode, OpenCode Zen)
  with Add Provider; Agents tab loads agents and the create form shows
  provider + per-provider model selects.

### Settings consolidation round 2 (2026-08-15, same session) — ✅ DONE

- **Skills moved into Settings** (Skills tab) using the same panel pattern as
  Agents: `components/skills.js` refactored from a modal into `skillsPanel`
  (shell methods removed), `partials/skills_panel.html` nested inside the
  Settings modal. Sidebar Skills button, `modalSkills` registration,
  `skills_modal.html` and unused `openSkills/closeSkills` store methods
  removed. Sidebar now: Knowledge Base, Notes, Jobs, Settings.
- **LLM tab rework**: removed the **LLM Server URL** and **Max Tokens**
  fields. The remaining model fields — Default / Query / Embedding /
  Reranking / Jobs model — are now **selects populated from every
  configured provider's cached models**, labelled `model (provider)` and
  deduplicated by provider+id (61 options across the 3 configured
  providers). Selecting the new "LLM Providers" section stays on the same
  tab.
- ⚠️ Notes: `llama_cpp_base_url` and `default_max_tokens` remain in
  settings.json (fallbacks for RAG embeddings/rerank and request caps) but
  are no longer exposed in the UI; per-model context limits from providers
  are a future refinement (the app still sends the settings default max
  tokens).
- Verified in browser: sidebar shows 4 feature buttons; 9 settings tabs;
  LLM tab has provider-populated selects and no URL/max-tokens fields;
  Skills tab creates and lists a skill (panel save path tested, test skill
  deleted).

### Configure llm-ui through llm-ui + skill registry installer (2026-08-15) — ✅ DONE

**Skill registry browser/installer (Phase 4.5)** — `backend/tools/skill_registry.py`:
- Listing: `GET /api/skills/registry?query=&limit=` → skills.sh public search
  API (`www.skills.sh/api/search`; the `/api/v1/skills` catalog views need
  auth, so search is the listing path; query min 2 chars).
- Install: `POST /api/skills/install {id}` resolves `owner/repo[/path]`,
  locates SKILL.md at the common layouts (skills/<name>/, <path>/, <name>/,
  repo root) on GitHub, downloads the skill directory (contents API,
  capped at 60 files / 4 MB, dotfiles skipped), writes it into
  `skills/<name>/` verbatim (frontmatter preserved) so the existing
  loader/index/load_skill pipeline picks it up. Names slugified; fetches
  pinned to raw.githubusercontent.com + api.github.com.
- UI: Skills tab now has **Installed / Browse Registry** views — search
  box, results with install counts + source, Install button (marks
  installed, refreshes the list, error toasts).
- Verified in-process: search "news" returned 3 skills; installed brave
  `news-search` into a temp dir (SKILL.md + frontmatter intact); test
  cleaned up.

**Admin toolset (agent configures the app through chat)** —
`backend/tools/admin_tool.py`, 10 tools wired into ToolExecutor and gated
by per-agent `enabled_tools` (exclusion list in main.py extended to cover
all 19 custom tools):
- `list_agents` / `create_agent` / `delete_agent`
- `list_mcp_servers` / `add_mcp_server` / `remove_mcp_server`
- `list_providers` / `add_provider` (models auto-fetched)
- `search_skills` / `install_skill` (registry)
- Together with the existing `create_skill`, `run_command`, `memory_*`,
  `load_skill` and `run_job` tools, the agent can now perform every
  configuration task through chat: create agents, add MCP servers, add
  providers, install skills.
- Verified in-process: create_agent → list (3 agents) → delete_agent →
  no residue; search + install paths exercised.
- ⚠️ Trust note: MCP server commands spawn processes (npx etc.) and are
  NOT covered by run_command's approval gate — this toolset should be
  limited to trusted agents via `enabled_tools` if that matters.
- ⚠️ **Activation**: the app server must be restarted to load the new
  endpoints/tools (the running instance predates them; the UI surfaces
  "API Error" for the registry search until then).

### Registry results enrichment + MCP server registry (2026-08-15) — ✅ DONE

**Skills registry cards** (user request):
- GitHub enrichment in `skill_registry.enrich_registry`: each unique
  `owner/repo` source gets stars, repo description, and a GitHub link via
  the GitHub API. In-process cache (1h TTL) + per-call cap (15 repos) +
  stop-on-rate-limit (403/429 skips remaining, uncached so they retry) —
  unauthenticated GitHub quota is 60 req/hr/IP.
- Frontend: registry results now render as a **3-column card grid**
  (name, source, installs, ★ stars, 1-2 line description, GitHub link,
  Install button / installed badge).

**MCP server registry** (answer to "is something similar present?": no —
now it is):
- `backend/tools/mcp_registry.py`: browse Smithery
  (`registry.smithery.ai/servers?q=&perPage=`; blank = sorted by use
  count), GitHub enrichment reusing the shared cache, detail fetch
  (`/servers/{ns}/{slug}`) for install config.
- Install: auto-picks the first usable connection — stdio (command/args/
  env) or http (deployment URL → streamable-http) — and adds the server
  through `mcp_manager.add_server` (same path as the Settings form).
- Endpoints: `GET /api/mcp/registry`, `POST /api/mcp/registry/install`.
- UI: MCP Servers tab now has **Configured / Browse Registry** views —
  cards with name, description, use count, ★ stars, GitHub link, verified
  badge, Install (switches back to Configured + refreshes on success).
- ⚠️ Observed: Smithery remote servers often need an API key (their
  configSchema) — the install adds them but the connection may fail; the
  toast says "added, but connection failed" and the user can fill the key
  in the edit form. Surfacing configSchema fields in the install flow is a
  follow-up.
- Verified: skills cards render with stars/links (7/10 enriched in one
  call); MCP most-used list loads; install of a remote server added it as
  streamable-http (then removed — test data cleaned).

# LLM UI - Agent Guide

A real-time chat interface for LLMs with MCP (Model Context Protocol), RAG, and TTS support. Built with FastAPI, Alpine.js, and Jinja2 templates.

## Quick Reference

| Area | Key File(s) | Purpose |
|------|-------------|---------|
| **Entry Point** | `run.py` | Launches uvicorn with `backend/app/main:app` |
| **API Router** | `backend/app/main.py` | All FastAPI endpoints, SSE streaming, MCP/Agent CRUD |
| **Settings** | `backend/settings.py` | `SettingsManager` singleton; `settings.json` on disk |
| **DB Models** | `backend/database/models.py` | SQLAlchemy models (Conversation, Message, Agent, MCPServer, Document) |
| **DB CRUD** | `backend/database/crud.py` | Async CRUD helpers for Conversation, Message, MCPServer, Document |
| **Agent CRUD** | `backend/database/agent_crud.py` | Agent-specific CRUD (create, update, soft-delete, default) |
| **LLM Client** | `backend/llm_client/client.py` | llama.cpp OpenAI-compatible streaming client |
| **MCP Client** | `backend/mcp_client/client.py` | `MCPClientManager` — connects/disconnects MCP servers via FastMCP |
| **Tool Executor** | `backend/tools/tool_executor.py` | Dispatches custom+MCP tools, yields `tool_progress` events |
| **RAG Service** | `backend/tools/rag_service.py` | Document processing → chunking → embedding → retrieval |
| **TTS Service** | `backend/tools/tts_service.py` | Edge TTS / pyttsx3 / Kokoro text-to-speech |
| **Shared Utils** | `backend/tools/base.py` | Embedding + reranking helpers (cosine similarity) |
| **Progress Helpers** | `backend/tools/progress.py` | `ToolProgress` for standardized progress events |
| **Frontend Entrypoint** | `frontend/static/js/main.js` | Imports Alpine ESM, registers stores+components, calls Alpine.start() |
| **State Store** | `frontend/static/js/store.js` | Alpine stores: `chat`, `ui` + `createModal()` factory |
| **Chat Component** | `frontend/static/js/components/chat.js` | Message send, SSE processing, tool call display, regeneration |
| **Sidebar Component** | `frontend/static/js/components/sidebar.js` | Conversation list, load/delete, resize |
| **Settings Component** | `frontend/static/js/components/settings.js` | App settings, MCP server management |
| **SSE Service** | `frontend/static/js/services/sse.js` | `SSEService` — fetch-based SSE streaming |
| **TTS Service** | `frontend/static/js/services/tts.js` | `TTSService` — audio playback control |
| **Utilities** | `frontend/static/js/utils.js` | API client, markdown rendering, formatters, helpers |
| **Main Template** | `frontend/templates/index.html` | Sidebar + chat + all modals (settings, agents, mcp, documents) |
| **Chat Partial** | `frontend/templates/partials/chat.html` | Message rendering, input box, tool call UI |
| **Config** | `settings.json` | Runtime settings (loaded by `SettingsManager`) |

## Architecture Overview

### High-Level Data Flow

```
Browser (Alpine.js)
    │
    ├─ POST /api/conversations/{id}/messages  →  request_id
    │       │
    │       ▼
    │  FastAPI (main.py)
    │       │
    │       ├─ DB (SQLite via SQLAlchemy) — persist messages
    │       ├─ LLM Client (llama.cpp) — stream chat completions
    │       ├─ Tool Executor — run MCP/custom tools with progress
    │       ├─ MCP Manager — manage server connections
    │       └─ SSE Stream — yield content/thinking/tool_progress events
    │       │
    │       ▼
    │  GET /api/stream/{request_id}  →  SSE text/event-stream
    │       │
    ▼       ▼
Browser  SSEService  →  processStreamEvent()  →  update store
```

### Backend Modules

```
backend/
├── app/main.py              # FastAPI app, lifespan, all endpoints
├── settings.py              # SettingsManager + Settings Pydantic model
├── database/
│   ├── models.py            # SQLAlchemy ORM: Conversation, Message, Agent, MCPServer, Document
│   ├── crud.py              # Async CRUD for Conversation, Message, MCPServer, Document
│   └── agent_crud.py        # Async CRUD for Agent (soft-delete, default)
├── llm_client/client.py     # LLMClient — streaming chat with llama.cpp
├── mcp_client/client.py     # MCPClientManager — server lifecycle, tool discovery/calls
└── tools/
    ├── base.py              # SharedLLMUtils — embeddings, reranking, cosine similarity
    ├── tool_executor.py     # ToolExecutor — custom tool dispatch + MCP tool wrapping
    ├── rag_service.py       # RAGService — document processing + semantic retrieval
    ├── tts_service.py       # TTSService — Edge TTS / pyttsx3 / Kokoro
    └── progress.py          # ToolProgress — standardized progress event helper
```

### Frontend Modules

```
frontend/
├── static/js/
│   ├── main.js              # Entry point — imports Alpine ESM, registers stores+components, calls Alpine.start()
│   ├── store.js             # Alpine stores: chat, ui + createModal() factory
│   ├── utils.js             # API client, markdown, formatters, helpers
│   ├── services/
│   │   ├── sse.js           # SSEService — fetch-based SSE streaming
│   │   └── tts.js           # TTSService — audio playback
│   └── components/
│       ├── chat.js          # Chat: send, stream, tool calls, regeneration, TTS (exports chatComponent)
│       ├── sidebar.js       # Sidebar: conversations, resize, load/delete (exports sidebar)
│       ├── settings.js      # Settings: app config, MCP server management (exports settings)
│          All component files export named factories — registration happens in main.js
├── templates/
│   ├── base.html            # Tailwind + marked.js CDN; defines store data on window; loads main.js as ES module
│   ├── index.html           # Main layout (sidebar + chat + all modals: settings, agents, mcp, documents)
│   └── partials/
│       ├── sidebar.html     # Conversation list
│       ├── chat.html        # Messages, input, tool call display
└── static/css/theme.css     # Custom theme + modal styles
```

## Core Concepts

### 1. Streaming Architecture (SSE)

The app uses Server-Sent Events for real-time LLM streaming:

1. **Client** sends `POST /api/conversations/{id}/messages` → gets `request_id`
2. **Client** opens SSE connection: `GET /api/stream/{request_id}?conversation_id={id}`
3. **Server** runs `_core_stream_handler()` which:
   - Loads conversation messages from DB
   - Prepends system prompt (with agent config if applicable)
   - Streams LLM response chunks via `llm_client.stream_chat()`
   - On tool calls: executes tools via `tool_executor.execute_tool()` (yields progress)
   - Yields events: `content`, `thinking`, `tool_call_start`, `tool_progress`, `tool_error`, `done`, `title_update`
4. **Client** `SSEService` parses SSE events → calls `processStreamEvent()` → updates Alpine store

**Key event types:**
| Event | SSE Data | UI Effect |
|-------|----------|-----------|
| `content` | `{ type: "content", content: "..." }` | Appends to last assistant message |
| `thinking` | `{ type: "thinking", content: "..." }` | Collapsible thinking block |
| `tool_call_start` | `{ type: "tool_call_start", tool: "name", args: {...} }` | Shows tool call in progress |
| `tool_progress` | `{ type: "tool_progress", tool, status, progress, result }` | Updates tool call status |
| `tool_error` | `{ type: "tool_error", tool, error }` | Shows error in tool call |
| `done` | `{ type: "done" }` | Closes SSE stream, marks complete |
| `title_update` | `{ type: "title_update", title: "..." }` | Updates conversation title |

### 2. Message Block System

Messages are stored with a `blocks` array in `metadata.blocks` for sequential rendering:

```python
# Each block has a type:
blocks = [
    {"type": "content", "content": "Hello..."},
    {"type": "thinking", "content": "Let me think..."},
    {"type": "tool_call", "name": "query_documents", "arguments": {...}, "result": {...}, "sources": [...]},
    {"type": "content", "content": "Here are the results..."},
]
```

The frontend renders blocks in order, preserving the interleaving of content, thinking, and tool calls.

### 3. Modal System (Consistent Design)

All secondary features (Settings, Agents, MCP Servers, Knowledge Base) are implemented as **centered modals** using a consistent pattern:

**Opening a modal:**
```html
<!-- Sidebar button dispatches event -->
<button @click="$dispatch('open-settings'); $store.ui.openSettings()">
    <i class="ph ph-gear"></i> Settings
</button>
```

**Modal structure:**
```html
<div x-data="modalSettings"
     x-show="open"
     x-cloak
     @open-settings.window="open = true"
     @keydown.escape.window="closeModal()"
     class="modal-backdrop"
     @click.self="closeModal()">
    <div class="modal-content">
        <!-- Header -->
        <div class="px-6 py-4 border-b flex items-center justify-between">
            <h2 class="text-lg font-semibold">Settings</h2>
            <button @click="closeModal()" class="btn btn-icon btn-ghost">
                <i class="ph ph-x text-lg"></i>
            </button>
        </div>
        <!-- Body -->
        <div class="p-6 overflow-y-auto">...</div>
        <!-- Footer (optional) -->
        <div class="px-6 py-4 border-t flex justify-end">...</div>
    </div>
</div>
```

**Modal component factory** (`store.js`):
```javascript
export function createModal(storeKey, openMethod, closeMethod) {
  return function () {
    return {
      open: false,
      openModal() {
        this.open = true
        if (typeof openMethod === 'function') openMethod()
      },
      closeModal() {
        if (closeMethod) {
          if (typeof closeMethod === 'function') closeMethod()
          else if (typeof $store !== 'undefined') $store.ui[closeMethod]()
        }
        this.open = false
      }
    }
  }
}
```

**Registered modals:**
| Modal | Component | Store Methods | Z-Index |
|-------|-----------|---------------|---------|
| MCP Servers | `modalMcp` | `openMcpPanel`/`closeMcpPanel` | 101 |
| AI Agents | `modalAgents` | `openAgents`/`closeAgents` | 102 |
| Knowledge Base | `modalDocuments` | `openDocuments`/`closeDocuments` | 103 |
| Create/Edit forms | inline | — | 104 |

**Key patterns:**
- All modals use `x-data` with local `open` state (not direct store binding)
- Opening: `$dispatch('open-xxx')` + store method (cross-scope communication)
- Closing: `closeModal()` sets `open = false` and calls store close method
- Escape key: `@keydown.escape.window="closeModal()"`
- Backdrop click: `@click.self="closeModal()"`
- `x-cloak` prevents flash of unstyled content (requires `[x-cloak] { display: none !important; }` in CSS)

### 4. Agent System

Agents are pre-configured AI personalities stored in the `agents` table:

- Each agent has: `name`, `system_prompt`, `model`, `temperature`, `top_k`, `max_tokens`, `enabled_tools`, `enabled_mcp_servers`, `enable_rag`, `rag_similarity_threshold`, `conversation_starters`
- Conversations link to agents via `conversation.agent_id`
- When a conversation has an agent, the agent's config overrides chat defaults
- Agents support soft-delete (`is_active = 0`)
- CRUD via `/api/agents` endpoints, managed through the AI Agents modal

### 5. MCP Server Management

- Servers stored in `mcp_servers` table with `transport_type` (stdio/sse/streamable-http)
- `MCPClientManager` connects to servers on startup (enabled ones only)
- Supports stdio (command + args + env) and HTTP/SSE (url + timeout) transports
- Tool discovery: `client.list_tools()` → parsed to OpenAI function format
- Tools are prefixed with server name: `server_name:tool_name`
- Web search and other capabilities are provided dynamically by connected MCP servers
- Server lifecycle: add, remove, reconnect, refresh tools, toggle enabled/disabled
- Managed through the MCP Servers modal

### 6. Custom Tools (Progress Tracking)

Custom tools in `ToolExecutor.custom_tools` use `ToolProgress` to bridge callback-based progress with async generators:

```python
# Custom tools that yield progress events:
"query_documents": self._query_documents_with_progress,
"generate_speech": self._generate_speech_with_progress,
```

Each yields: `tool_progress` (with status/progress/result) or `tool_error`.

### 7. RAG Pipeline

```
Upload → Extract text → Chunk → Embed → Store (SQLite BLOB)
                                              ↓
Query → Embed → Search (cosine similarity) → Rerank → Format with citations
```

- `DocumentProcessor`: extracts text from PDF/DOCX/TXT/MD/JSON/YAML
- `Chunker`: word-based chunking with configurable overlap
- `EmbeddingStore`: SQLite tables (`document_chunks`, `document_embeddings`)
- `RAGService`: coordinates processing and querying
- Embeddings stored as raw `float32` blobs in SQLite
- Managed through the Knowledge Base modal

### 8. TTS Service

Three backends with automatic fallback:
1. **Edge TTS**: High quality, requires internet
2. **pyttsx3**: Offline, lower quality
3. **Kokoro**: High quality, local, requires model download

Features: caching by content hash, volume/speed adjustment (Kokoro), CUDA OOM fallback to CPU.

## Frontend State Architecture

### Alpine Stores

Two stores manage global state:

**`chat` store** — Chat state:
```javascript
{
  conversations: [],
  currentConversationId: null,
  currentConversationTitle: '',
  messages: [],
  inputMessage: '',
  isLoading: false,
  activeStreaming: { isStreaming, requestId, conversationId, msgIndex, conversationTitle, messages },
  toolStatus: { active, tool, status, progress, data },
  selectedModel: '',
  availableModels: [],
  selectedAgentId: null,
  availableAgents: [],
  currentAgentConfig: null,
  enableRAG: false,
  editingMessageId: null,
  editContent: '',
  // Methods: addConversation, updateConversation, removeConversation,
  // addMessage, updateMessage, removeMessage, setModel, loadSavedModel,
  // setAgent, loadSavedAgent, startStreaming, stopStreaming, applyAgentConfig
}
```

**`ui` store** — UI state (theme, panels, settings data):
```javascript
{
  // Theme
  darkMode: boolean,
  sidebarCollapsed: boolean,
  sidebarWidth: number,

  // Panel visibility (controls modals)
  showMcpPanel: false,
  showAgents: false,
  showSettings: false,
  showDocuments: false,

  // Toast
  toast: { show, message, type },

  // Settings data
  settingsData: {},
  mcpServers: [],
  mcpTools: [],
  documents: [],

  // MCP form state
  newServer: { name, transport_type, command, args, url, env },
  editingServer: false,
  editServer: { name, transport_type, command, args, url, env, enabled, originalName },

  // Settings tab
  settingsTab: 'general',

  // Panel methods
  toggleDarkMode(), initTheme(), toggleSidebar(), setSidebarWidth(w),
  showToast(message, type),
  openMcpPanel(), closeMcpPanel(),
  openAgents(), closeAgents(),
  openSettings(), closeSettings(),
  openDocuments(), closeDocuments()
}
```

### Alpine.js Initialization (ESM Module Build)

Alpine.js is loaded via the **ESM module build** (`alpinejs@3.15.12/+esm`), which does NOT auto-start. This gives full control over initialization.

**Initialization flow** (deterministic, driven by ES module import graph):
1. `base.html` defines store data as plain objects on `window` (`__chatStoreData__`, `__uiStoreData__`) via inline `<script>` — runs synchronously during HTML parse
2. `main.js` loads as `<script type="module">` — ES module is deferred by default
3. `main.js` imports Alpine ESM module → sets `window.Alpine` (no auto-start)
4. `main.js` imports SSE/TTS services and utils (no Alpine dependency)
5. `main.js` registers stores: `Alpine.store('chat', __chatStoreData__)`, `Alpine.store('ui', __uiStoreData__)`
6. `main.js` imports component factories, then calls `Alpine.data('name', factory)` for each
7. `main.js` calls `Alpine.start()` — Alpine walks DOM, all stores and components are ready

**Key insight**: ES module import order is guaranteed. No race conditions. The module graph ensures everything registers before `Alpine.start()`.

**Component registration**: Component files export named factories. `main.js` registers them with `Alpine.data('name', factory)`. Templates use `x-data="name"` (no parentheses).

**Modal components**: Registered via `createModal()` factory in `main.js`. Each modal has its own `x-data` scope with local `open` state, synced with the global store via `$dispatch` events.

## Key Implementation Details

### Streaming Loop (main.py)

The `_core_stream_handler()` implements a conversation loop:
1. Load messages, prepend system prompt (agent config if applicable)
2. Get MCP tools + custom tool definitions
3. **Loop** (max 35 iterations):
   a. Stream LLM response (content/thinking/tool_call chunks)
   b. If tool call: execute tool, add result to messages, loop back
   c. If no tool call: exit loop
4. Save assistant message with consolidated blocks
5. Yield `done` event
6. Generate title (if first exchange, uses `QUERY_MODEL`)

### Tool Execution Flow

```
LLM returns tool_call → backend receives chunk
    → pending_tool_call created
    → yield tool_call_start event
    → tool_executor.execute_tool() called
    → Custom tools: yield progress events from ToolProgress
    → MCP tools: call via MCPClientManager, yield start/progress/complete/error
    → Tool result added to llm_messages as role="tool"
    → Loop continues (LLM sees tool result)
```

### Message Saving

Assistant messages are saved with:
- `content`: Full concatenated text
- `thinking`: Concatenated thinking content (from blocks)
- `metadata.blocks`: Array of {type, content} blocks for sequential rendering
- `metadata.model`: Model used for generation
- `metadata.tool_calls`: Extracted from tool_call blocks (backward compat)

### Settings System

`SettingsManager` is a singleton that:
1. Loads `settings.json` on startup
2. Falls back to environment variables
3. Provides `get_settings()` and `update_settings()` methods
4. Updates environment variables for runtime changes
5. Has a `set_tts_service()` hook for TTS config updates
6. Module-level constants (`DATABASE_URL`, `LLAMA_CPP_BASE_URL`, etc.) are synced via `_update_module_constants()`

### MCP Client Lifecycle

1. **Startup**: `lifespan()` calls `mcp_manager.initialize()` → loads enabled servers from DB → connects each
2. **Add server**: Saves to DB, connects via FastMCP, discovers tools
3. **Remove server**: Disconnects, removes from DB
4. **Reconnect**: Closes existing connection, reconnects
5. **Refresh tools**: Re-discover tools for a server
6. **Tool call**: Uses `async with instance.client:` context manager, calls `client.call_tool()`, parses `CallToolResult`

### RAG Storage

RAG uses raw SQLite (not SQLAlchemy) for embeddings:
- `document_chunks` table: chunk text + metadata
- `document_embeddings` table: embedding as `float32` blob
- Queries: cosine similarity in Python (numpy)
- Index on `document_chunks.document_id` for fast document queries

## Common Tasks

### Add a New Tool

1. **Backend**: Add to `ToolExecutor.custom_tools` dict in `backend/tools/tool_executor.py`
2. Add tool definition to `get_tool_definitions()` return list
3. Implement `_tool_name_with_progress()` async generator yielding `tool_progress` events
4. **Frontend**: Tool calls are auto-displayed via the block system — no frontend changes needed

### Add Web Search via MCP

Web search is provided by MCP servers instead of a built-in adapter:

1. Create or connect an MCP server that provides a search tool
2. The tool will be auto-discovered and available for the LLM to call
3. No UI toggle is needed — the LLM decides whether to use the search tool based on the user's query

### Add a New API Endpoint

1. Add route decorator to `backend/app/main.py`
2. Use `async with get_db() as db:` for DB access
3. Import CRUD functions from `backend/database/crud.py` or `agent_crud.py`
4. Return JSON response

### Add a New Database Field

1. Add column to model in `backend/database/models.py`
2. Add CRUD accessor in `backend/database/crud.py`
3. Run migration (or delete `llm_ui.db` to recreate schema)

### Add a New Modal

1. Add panel state to `ui` store: `showXxx: false`
2. Add methods to `ui` store: `openXxx()`, `closeXxx()`
3. Add methods to `__uiStoreData__` in `base.html`
4. Register modal in `main.js`: `Alpine.data('modalXxx', createModal('xxx', 'openXxx', 'closeXxx'))`
5. Add modal HTML to `index.html` using the consistent modal pattern
6. Add sidebar button: `<button @click="$dispatch('open-xxx'); $store.ui.openXxx()">...</button>`
7. Add z-index class in `theme.css`: `.modal-backdrop-xxx { z-index: N; }`

### Change Default Settings

Edit `settings.json` at project root, or use environment variables. The `SettingsManager` loads from file first, then falls back to env vars.

### Add a New TTS Engine

1. Add import check in `backend/tools/tts_service.py`
2. Add to `TTSConfig` dataclass
3. Implement `_generate_with_<engine>()` method
4. Add to the engine selection logic in `generate_speech()`
5. Add to `get_available_engines()` and `list_available_voices()`

## Debugging Tips

- **SSE streaming issues**: Check browser Network tab for SSE connection, look for SSEService errors in console
- **Tool call not appearing**: Check `[TOOLS]` debug logs in server output for tool discovery
- **RAG search returning nothing**: Verify embeddings were generated (check document status in DB), check similarity threshold
- **MCP server not connecting**: Check server logs, verify command/args/env, check `MCPClientManager` connection logs
- **Settings not persisting**: Check `settings.json` file permissions, verify `SettingsManager.save_settings_to_file()` is called
- **Message blocks not rendering**: Check `metadata.blocks` in DB, verify frontend `processStreamEvent()` is receiving correct events
- **Web search not available**: Connect an MCP server that provides a search tool. Web search is no longer built-in.
- **Modal not opening**: Check browser console for `$dispatch` errors, verify `@open-xxx.window` listener is on the modal element
- **Modal close button not working**: Ensure `closeModal()` is defined as a method (not arrow function) in `x-data`, use `closeModal() { this.open = false }` not `closeModal: () => { this.open = false }`

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.14, FastAPI, Uvicorn, SQLAlchemy (async), aiosqlite |
| **LLM** | llama.cpp (OpenAI-compatible API), aiohttp for streaming |
| **MCP** | FastMCP library, MCP protocol (stdio + SSE + StreamableHTTP) |
| **Embeddings** | llama.cpp `/v1/embeddings` endpoint, numpy |
| **Reranking** | llama.cpp `/v1/rerank` endpoint |
| **RAG** | Document processing + semantic retrieval via embeddings |
| **TTS** | edge-tts, pyttsx3, Kokoro (optional) |
| **Frontend** | Alpine.js 3.15 (ESM module), Tailwind CSS, Marked.js |
| **Database** | SQLite (async via aiosqlite) |
| **Templates** | Jinja2 |

## Page Routes

| Path | Behavior |
|------|----------|
| `/` | Main chat interface with sidebar + all modals |
| `/settings` | Redirects to `/` (settings in modal) |
| `/knowledge` | Redirects to `/` (knowledge base in modal) |
| `/agents` | Redirects to `/` (agents in modal) |

## File Sizes (for reference)

| File | Lines | Size |
|------|-------|------|
| `backend/app/main.py` | ~1155 | 45KB |
| `backend/mcp_client/client.py` | ~400 | Large |
| `frontend/static/js/components/chat.js` | ~500 | Large |
| `frontend/templates/index.html` | ~1100 | Large (all modals inline) |
| `frontend/static/js/store.js` | ~220 | Medium (stores + modal factory) |
| `frontend/templates/partials/chat.html` | ~565 | Large |
| `frontend/static/js/components/settings.js` | ~200 | Medium |
| `frontend/static/js/components/sidebar.js` | ~150 | Small |

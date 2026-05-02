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
| **State Store** | `frontend/static/js/store.js` | Alpine stores: `chat`, `settings`, `documents`, `tts` |
| **Chat Component** | `frontend/static/js/components/chat.js` | Message send, SSE processing, tool call display, regeneration |
| **Sidebar Component** | `frontend/static/js/components/sidebar.js` | Conversation list, load/delete, resize |
| **Settings Component** | `frontend/static/js/components/settings.js` | App settings, MCP server management |
| **Documents Component** | `frontend/static/js/components/documents.js` | Upload/delete knowledge base docs |
| **SSE Service** | `frontend/static/js/services/sse.js` | `SSEService` — fetch-based SSE streaming |
| **TTS Service** | `frontend/static/js/services/tts.js` | `TTSService` — audio playback control |
| **Utilities** | `frontend/static/js/utils.js` | API client, markdown rendering, formatters, helpers |
| **Main Template** | `frontend/templates/index.html` | Extends `base.html`, includes sidebar/chat/MCP partials |
| **Chat Partial** | `frontend/templates/partials/chat.html` | Message rendering, input box, tool call UI |
| **Settings Partial** | `frontend/templates/partials/settings.html` | Settings modal |
| **Knowledge Page** | `frontend/templates/knowledge.html` | Document management page |
| **Agents Page** | `frontend/templates/agents.html` | Agent management page |
| **Config** | `settings.json` | Runtime settings (loaded by `SettingsManager`) |

## Architecture Overview

### High-Level Data Flow

```
Browser (Alpine.js + HTMX)
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
│   ├── store.js             # Alpine stores: chat, ui (merged from chat, settings, documents, tts)
│   ├── utils.js             # API client, markdown, formatters, helpers
│   ├── services/
│   │   ├── sse.js           # SSEService — fetch-based SSE streaming
│   │   └── tts.js           # TTSService — audio playback
│   └── components/
│       ├── chat.js          # Chat: send, stream, tool calls, regeneration, TTS (exports chatComponent)
│       ├── sidebar.js       # Sidebar: conversations, resize, load/delete (exports sidebar)
│       ├── settings.js      # Settings: app config, MCP server management (exports settings)
│       └── documents.js     # Documents: upload, list, delete (exports documents)
│          All component files export named factories — registration happens in main.js
├── templates/
│   ├── base.html            # Tailwind + marked.js CDN; defines store data on window; loads main.js as ES module
│   ├── index.html           # Main layout (sidebar + chat + MCP panel)
│   ├── settings.html        # Settings page (standalone)
│   ├── knowledge.html       # Knowledge base page (standalone)
│   ├── agents.html          # Agents page (standalone)
│   └── partials/
│       ├── sidebar.html     # Conversation list
│       ├── chat.html        # Messages, input, tool call display
│       ├── settings.html    # Settings modal
│       ├── documents.html   # Documents modal
│       └── mcp_panel.html   # MCP server panel
└── static/css/styles.css    # Custom Tailwind overrides
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

### 3. Agent System

Agents are pre-configured AI personalities stored in the `agents` table:

- Each agent has: `name`, `system_prompt`, `model`, `temperature`, `top_k`, `max_tokens`, `enabled_tools`, `enabled_mcp_servers`, `enable_rag`, `rag_similarity_threshold`, `conversation_starters`
- Conversations link to agents via `conversation.agent_id`
- When a conversation has an agent, the agent's config overrides chat defaults
- Agents support soft-delete (`is_active = 0`)
- CRUD endpoints: `/api/agents` (list, create), `/api/agents/{id}` (get, update, delete)

### 4. MCP Server Management

- Servers stored in `mcp_servers` table with `transport_type` (stdio/sse/streamable-http)
- `MCPClientManager` connects to servers on startup (enabled ones only)
- Supports stdio (command + args + env) and HTTP/SSE (url + timeout) transports
- Tool discovery: `client.list_tools()` → parsed to OpenAI function format
- Tools are prefixed with server name: `server_name:tool_name`
- Web search and other capabilities are provided dynamically by connected MCP servers
- Server lifecycle: add, remove, reconnect, refresh tools, toggle enabled/disabled

### 5. Custom Tools (Progress Tracking)

Custom tools in `ToolExecutor.custom_tools` use `ToolProgress` to bridge callback-based progress with async generators:

```python
# Custom tools that yield progress events:
"query_documents": self._query_documents_with_progress,
"generate_speech": self._generate_speech_with_progress,
```

Each yields: `tool_progress` (with status/progress/result) or `tool_error`.

### 6. RAG Pipeline

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

### 7. TTS Service

Three backends with automatic fallback:
1. **Edge TTS**: High quality, requires internet
2. **pyttsx3**: Offline, lower quality
3. **Kokoro**: High quality, local, requires model download

Features: caching by content hash, volume/speed adjustment (Kokoro), CUDA OOM fallback to CPU.

## Database Schema

### Tables

```sql
conversations (id PK, title, agent_id FK→agents.id, created_at, updated_at)
messages      (id PK, conversation_id FK→conversations.id, role, content,
               thinking, tool_calls JSON, metadata JSON, created_at)
mcp_servers   (id PK, name UNIQUE, transport_type, command, args JSON,
               env JSON, url, enabled, created_at)
documents     (id PK, filename, filepath, file_type, size_bytes, status,
               metadata JSON, uploaded_at, processed_at)
agents        (id PK, name UNIQUE, description, model, temperature, top_k,
               max_tokens, system_prompt, enabled_tools JSON,
               enabled_mcp_servers JSON, enable_rag, rag_similarity_threshold,
               conversation_starters JSON,
               created_at, updated_at, is_active)

-- RAG tables (managed by EmbeddingStore, not SQLAlchemy):
document_chunks (id PK, document_id FK, chunk_index, content, start_char, end_char, created_at)
document_embeddings (chunk_id PK, embedding BLOB, FK→document_chunks.id)
```

### Key Relationships

- `Conversation` → `Message` (1:N, cascade delete)
- `Conversation` → `Agent` (N:1, nullable)
- `Agent` → `Conversation` (1:N, cascade delete)
- `Document` → `document_chunks` (1:N)
- `document_chunks` → `document_embeddings` (1:1)

## API Endpoints

### Conversations
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations` | Create new conversation |
| GET | `/api/conversations/{id}` | Get conversation with messages |
| PUT | `/api/conversations/{id}` | Update conversation title |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| POST | `/api/conversations/{id}/messages` | Send message (returns `request_id`) |
| POST | `/api/conversations/{id}/regenerate` | Regenerate last assistant response |
| GET | `/api/stream/{request_id}` | SSE stream (query params: `conversation_id`, `enable_rag`, `model`) |
| GET | `/api/stream/regenerate/{request_id}` | SSE stream for regeneration |

### Messages
| Method | Path | Description |
|--------|------|-------------|
| PUT | `/api/messages/{id}` | Edit message content |
| DELETE | `/api/messages/{id}` | Delete message |

### MCP Servers
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/mcp/servers` | List all MCP servers with status |
| POST | `/api/mcp/servers` | Add MCP server |
| DELETE | `/api/mcp/servers/{name}` | Remove MCP server |
| PUT | `/api/mcp/servers/{name}` | Update MCP server |
| POST | `/api/mcp/servers/{name}/refresh` | Refresh tool list |
| POST | `/api/mcp/servers/{name}/reconnect` | Reconnect server |
| POST | `/api/mcp/servers/{name}/toggle` | Enable/disable server |
| GET | `/api/mcp/tools` | List all available tools |

### Agents
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/agents` | List all agents |
| GET | `/api/agents/{id}` | Get agent details |
| POST | `/api/agents` | Create agent |
| PUT | `/api/agents/{id}` | Update agent |
| DELETE | `/api/agents/{id}` | Soft-delete agent |

### Documents & RAG
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/documents/upload` | Upload document (multipart/form-data) |
| GET | `/api/documents` | List documents |
| GET | `/api/documents/{id}` | Get document details |
| DELETE | `/api/documents/{id}` | Delete document |
| POST | `/api/rag/query` | Direct RAG query |

### TTS
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/tts/generate` | Generate speech |
| GET | `/api/tts/voices` | List available voices |
| GET | `/api/tts/status` | Check TTS availability |
| GET | `/api/audio/{filename}` | Serve audio file |

### Settings & Models
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/settings` | Get settings |
| PUT | `/api/settings` | Update settings |
| GET | `/api/models` | List available LLM models |

### Pages
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Main chat interface |
| GET | `/settings` | Settings page |
| GET | `/knowledge` | Knowledge base page |
| GET | `/agents` | Agents management page |

## Frontend State Architecture

### Alpine Stores

Four stores manage global state:

**`chat` store** — Chat state:
```javascript
{
  conversations: [],           // All conversations
  currentConversationId: null, // Active conversation
  currentConversationTitle: '',
  messages: [],                // Messages for active conversation
  inputMessage: '',
  isLoading: false,
  selectedModel: '',
  availableModels: [],
  enableRAG: false,
  toolStatus: { active, tool, status, progress, data },
  activeStreaming: { isStreaming, requestId, conversationId, msgIndex, conversationTitle, messages },
  availableAgents: [],
  selectedAgentId: null,
  currentAgentConfig: null,    // Full agent config when selected
  toast: { show, message, type },
  // Methods: addConversation, updateConversation, removeConversation,
  // addMessage, updateMessage, removeMessage, setModel, loadSavedModel,
  // setAgent, loadSavedAgent, showToast, startStreaming, stopStreaming
}
```

**`settings` store** — App settings:
```javascript
{
  show: false,
  data: {},
  mcpServers: [],
  mcpTools: [],
  activeTab: 'general',
  newServer: { name, transport_type, command, args, url, env }
}
```

**`documents` store** — Knowledge base:
```javascript
{
  show: false,
  list: []
}
```

**`tts` store** — TTS state:
```javascript
{
  available: false,
  currentAudio: null,
  currentAudioMessageId: null,
  isPlaying: false,
  loading: {}
}
```

### Component Registration

Components are **exported as named factories** from their files and registered centrally in `main.js`:
- `sidebar` — exported from `sidebar.js`, registered as `Alpine.data('sidebar', sidebar)`
- `chatComponent` — exported from `chat.js`, registered as `Alpine.data('chat', chatComponent)`
- `settings` — exported from `settings.js`, registered as `Alpine.data('settings', settings)`

Component files do NOT call `Alpine.data()` directly — they only `export` their factory functions. Registration happens in `main.js` after Alpine is fully initialized.

### Alpine.js Initialization (ESM Module Build)

Alpine.js is loaded via the **ESM module build** (`@alpinejs/[email protected]/dist/module.esm.js`), which does NOT auto-start. This gives full control over initialization.

**Initialization flow** (deterministic, driven by ES module import graph):
1. `base.html` defines store data as plain objects on `window` (`__chatStoreData__`, `__uiStoreData__`) via inline `<script>` — runs synchronously during HTML parse
2. `main.js` loads as `<script type="module">` — ES module is deferred by default
3. `main.js` imports Alpine ESM module → sets `window.Alpine` (no auto-start)
4. `main.js` imports SSE/TTS services and utils (no Alpine dependency)
5. `main.js` registers stores: `Alpine.store('chat', __chatStoreData__)`, `Alpine.store('ui', __uiStoreData__)`
6. `main.js` imports component factories from their files, then calls `Alpine.data('name', factory)` for each
7. `main.js` calls `Alpine.start()` — Alpine walks DOM, all stores and components are ready

**Key insight**: ES module import order is guaranteed. No race conditions, no `Object.defineProperty` hacks, no `deferLoadingAlpine` workarounds. The module graph ensures everything registers before `Alpine.start()`.

**Component registration**: Component files export named factories (`export function sidebar() { ... }`). `main.js` registers them with `Alpine.data('name', factory)`. Templates use `x-data="sidebar"` (no parentheses).

**x-cloak**: Added to Settings modal, Documents modal, and MCP panel to prevent flash of unstyled content before Alpine processes `x-show` directives.

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

### Add a New Frontend Component

1. Create `frontend/static/js/components/new_component.js`
2. Register with `Alpine.data('name', fn)` (see sidebar.js, chat.js, settings.js)
3. Import in `frontend/static/js/main.js`
4. Use in templates with `x-data="newComponent()"`

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
| **Frontend** | Alpine.js (synchronous load + manual start), HTMX, Tailwind CSS, Marked.js |
| **Database** | SQLite (async via aiosqlite) |
| **Templates** | Jinja2 |

## File Sizes (for reference)

| File | Lines | Size |
|------|-------|------|
| `backend/app/main.py` | ~1155 | 45KB |
| `backend/mcp_client/client.py` | ~400 | Large |
| `frontend/static/js/components/chat.js` | ~500 | Large |
| `frontend/templates/partials/chat.html` | 565 | Large |
| `frontend/templates/settings.html` | 380 | Medium |
| `frontend/templates/partials/settings.html` | 367 | Medium |
| `frontend/templates/agents.html` | 274 | Medium |
| `frontend/templates/knowledge.html` | 177 | Medium |
| `frontend/static/js/store.js` | ~130 | Small |

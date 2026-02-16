# LLM UI - Complete Project Structure

## 📁 Project Layout

```
llm-ui-app/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # 5-minute setup guide
├── DEVELOPMENT.md                     # Customization guide
├── SEARXNG_INTEGRATION.md             # Guide for your search tool
├── LLM_MODELS_REFERENCE.md            # LLM configuration guide
├── .env.example                       # Environment variables template
├── run.py                             # Application launcher
│
├── backend/                           # Python FastAPI backend
│   ├── requirements.txt               # Python dependencies
│   ├── config.py                      # Configuration settings
│   ├── settings.py                    # Settings management
│   │
│   ├── app/
│   │   └── main.py                    # FastAPI application & SSE streaming
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py                  # SQLAlchemy models (Conversation, Message, etc.)
│   │   └── crud.py                    # Database operations
│   │
│   ├── mcp_client/
│   │   ├── __init__.py
│   │   └── client.py                  # MCP protocol client & server management
│   │
│   ├── llm_client/
│   │   ├── __init__.py
│   │   └── client.py                  # llama.cpp API client
│   │
│   └── tools/
│       ├── __init__.py
│       ├── base.py                    # Shared utilities (embeddings, reranking)
│       ├── rag_service.py             # RAG (Retrieval-Augmented Generation)
│       ├── searxng_tool.py            # SearXNG web search integration
│       ├── tool_executor.py           # Tool execution with progress tracking
│       └── tts_service.py             # Text-to-Speech service
│
└── frontend/                          # HTML/CSS/JS frontend
    ├── templates/
    │   └── index.html                 # Main UI (HTMX + Alpine.js)
    │
    └── static/
        ├── css/
        │   └── styles.css             # Custom styles
        │
        └── js/
            └── app.js                 # Alpine.js application logic
```

## 📄 Key Files Explained

| File | Purpose | Key Features |
|------|---------|--------------|
| `app/main.py` | Main FastAPI app | • SSE streaming endpoints<br>• Conversation management<br>• MCP server configuration<br>• Real-time updates<br>• RAG & web search integration<br>• Model selection<br>• Settings management |
| `database/models.py` | Database schema | • Conversations table<br>• Messages table<br>• MCP servers table<br>• Documents table<br>• Document chunks table<br>• Document embeddings table |
| `database/crud.py` | Database operations | • Async SQLAlchemy queries<br>• CRUD functions<br>• Relationship handling |
| `mcp_client/client.py` | MCP integration | • Multi-server management<br>• Tool discovery<br>• Stdio communication<br>• Dynamic tool loading |
| `llm_client/client.py` | llama.cpp client | • Streaming chat completion<br>• OpenAI-compatible API<br>• Tool integration<br>• Title generation<br>• Model switching |
| `tools/tool_executor.py` | Tool execution | • Progress tracking<br>• Custom tools support<br>• MCP tool wrapping<br>• Error handling<br>• TTS service integration |
| `tools/searxng_tool.py` | Web search | • SearXNG integration<br>• Multi-query generation<br>• Content extraction<br>• Semantic reranking<br>• Citation support |
| `tools/rag_service.py` | RAG service | • Document processing<br>• Chunking and indexing<br>• Semantic search<br>• Embedding generation<br>• Re-ranking |
| `tools/tts_service.py` | TTS service | • Multiple TTS engines<br>• Edge TTS support<br>• Offline TTS support<br>• Audio generation |
| `tools/base.py` | Shared utilities | • Embedding utilities<br>• Reranking utilities<br>• Cosine similarity |
| `config.py` | Configuration | • Environment variables<br>• Default settings<br>• URL configurations |
| `settings.py` | Settings management | • Runtime settings<br>• TTS configuration<br>• Model settings<br>• UI settings |

### Frontend Files

| File | Purpose | Key Features |
|------|---------|--------------|
| `templates/index.html` | Main UI | • Chat interface<br>• Conversation sidebar<br>• MCP server management<br>• Real-time updates<br>• Knowledge base<br>• Settings modal<br>• Model selection<br>• Tool toggles (web search, RAG)<br>• TTS controls |
| `static/js/app.js` | Application logic | • Alpine.js reactive state<br>• SSE event handling<br>• Message management<br>• Tool progress display<br>• TTS integration<br>• Document management<br>• Settings management |
| `static/css/styles.css` | Styling | • Custom scrollbars<br>• Animations<br>• Markdown rendering<br>• Dark mode support<br>• Responsive design |

## 🔧 Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM with async support
- **aiosqlite** - Async SQLite driver
- **aiohttp** - Async HTTP client
- **Uvicorn** - ASGI server
- **NumPy** - Numerical computations for embeddings
- **BeautifulSoup4** - HTML parsing for web search
- **Requests** - HTTP requests
- **PyPDF2** - PDF processing
- **python-docx** - DOCX processing

### Frontend
- **HTMX** - Dynamic HTML updates
- **Alpine.js** - Reactive UI framework
- **Tailwind CSS** - Utility-first CSS
- **Marked.js** - Markdown rendering

### External Services
- **llama.cpp** - LLM inference engine
- **MCP Servers** - Tool providers via MCP protocol
- **SearXNG** - Privacy-focused web search
- **Edge TTS** - High-quality text-to-speech

## 🎯 Core Features Implementation

### 1. Real-time Chat Streaming
```
User types message
    ↓
POST /api/conversations/{id}/messages
    ↓
Backend creates request_id
    ↓
Frontend connects to SSE: /api/stream/{request_id}
    ↓
Backend streams events (content, thinking, tool_progress, etc.)
    ↓
Frontend updates UI in real-time
```

### 2. MCP Server Integration
```
User adds MCP server via UI
    ↓
Saved to database
    ↓
MCP client starts server process
    ↓
Discovers available tools via MCP protocol
    ↓
Tools available for LLM to use
```

### 3. Tool Execution with Progress
```
LLM decides to use tool
    ↓
Tool executor checks if custom or MCP tool
    ↓
Executes tool with progress callbacks
    ↓
Yields progress events (0-100%)
    ↓
SSE streams progress to frontend
    ↓
UI shows live status updates
```

### 4. Web Search with SearXNG
```
User enables web search
    ↓
Multi-query generation
    ↓
SearXNG search execution
    ↓
Content extraction from results
    ↓
Semantic chunking and embedding
    ↓
Similarity filtering
    ↓
Re-ranking for relevance
    ↓
Citation formatting
    ↓
Context injection for LLM
```

### 5. RAG (Retrieval-Augmented Generation)
```
User uploads document
    ↓
Document processing pipeline
    ↓
Text extraction and chunking
    ↓
Embedding generation for chunks
    ↓
Storage in SQLite with embeddings
    ↓
Semantic search when querying
    ↓
Re-ranking for relevance
    ↓
Context injection for LLM
```

### 6. Persistent Conversations
```
SQLite Database Schema:
    conversations (id, title, created_at, updated_at)
        ↓
    messages (id, conversation_id, role, content, tool_calls, thinking, created_at)
        ↓
    documents (id, filename, filepath, file_type, size_bytes, status, metadata, created_at)
        ↓
    document_chunks (id, document_id, chunk_index, content, start_char, end_char, created_at)
        ↓
    document_embeddings (chunk_id, embedding)
```

## 📊 Data Flow Diagrams

### Message Flow
```
┌─────────┐      ┌──────────┐      ┌───────────┐      ┌──────────┐
│ Browser │─────▶│ FastAPI  │─────▶│ llama.cpp │─────▶│   MCP    │
│ (HTMX)  │◀─────│   (SSE)  │◀─────│  (Stream) │◀─────│  Server  │
└─────────┘      └──────────┘      └───────────┘      └──────────┘
     │                 │                                      │
     │                 ▼                                      │
     │           ┌──────────┐                                │
     │           │ Database │                                │
     │           │ (SQLite) │                                │
     └──────────▶└──────────┘◀───────────────────────────────┘
```

### Web Search Flow
```
┌─────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Browser │─────▶│ FastAPI  │─────▶│ SearXNG  │─────▶│   Web    │
│ (Query) │      │ (Search) │      │ (Query)  │      │ (Pages)  │
└─────────┘      └──────────┘      └──────────┘      └──────────┘
     │                 │                 │                 │
     │                 ▼                 ▼                 ▼
     │           ┌──────────┐      ┌──────────┐      ┌──────────┐
     │           │  Query   │─────▶│  Pages   │─────▶│  Content │
     │           │ Gen      │      │ Extract  │      │ Process  │
     │           └──────────┘      └──────────┘      └──────────┘
     │                 │                 │                 │
     │                 ▼                 ▼                 ▼
     │           ┌──────────┐      ┌──────────┐      ┌──────────┐
     │           │ Embedding│─────▶│ Rerank   │─────▶│  Format  │
     │           │ Gen      │      │ Results  │      │ Results  │
     └──────────▶└──────────┘      └──────────┘      └──────────┘
```

### RAG Flow
```
┌─────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Browser │─────▶│ FastAPI  │─────▶│  RAG     │─────▶│  SQLite  │
│ (Upload)│      │(Process) │      │ (Index)  │      │ (Chunks) │
└─────────┘      └──────────┘      └──────────┘      └──────────┘
     │                 │                 │                 │
     │                 ▼                 ▼                 ▼
     │           ┌──────────┐      ┌──────────┐      ┌──────────┐
     │           │  Extract │─────▶│  Chunk   │─────▶│  Store   │
     │           │  Text    │      │  & Emb   │      │  & Link  │
     │           └──────────┘      └──────────┘      └──────────┘
     │                 │                 │                 │
     │                 ▼                 ▼                 ▼
     │           ┌──────────┐      ┌──────────┐      ┌──────────┐
     │           │  Query   │─────▶│  Search  │─────▶│  Retrieve│
     │           │  Embed   │      │  & Rank  │      │  & Format│
     └──────────▶└──────────┘      └──────────┘      └──────────┘
```

## 🚀 Getting Started Paths

### Path 1: Quick Start (Basic Chat)
1. Start llama.cpp
2. Install Python deps
3. Run application
4. Chat immediately

Time: **5 minutes**
Complexity: **Easy**

### Path 2: With MCP Servers
1. Complete Quick Start
2. Install Node.js
3. Add MCP server via UI
4. Use tools in chat

Time: **10 minutes**
Complexity: **Medium**

### Path 3: Full Integration (Enhanced Features)
1. Complete Path 2
2. Set up SearXNG for web search
3. Configure embedding/reranking models
4. Enable document processing
5. Configure TTS services

Time: **1-2 hours**
Complexity: **Advanced**

## 📚 Documentation Index

| Document | When to Use |
|----------|-------------|
| `README.md` | Overview, features, installation |
| `QUICKSTART.md` | First-time setup in 5 minutes |
| `SEARXNG_INTEGRATION.md` | Integrating your search tool |
| `DEVELOPMENT.md` | Customization and extension |
| `LLM_MODELS_REFERENCE.md` | Model configuration and optimization |

## 🔍 Common Use Cases

### Use Case 1: Basic Chat Assistant
**What you get:** Clean UI for chatting with local LLM
**Setup needed:** llama.cpp + this app
**Time to value:** 5 minutes

### Use Case 2: Multi-Tool Assistant
**What you get:** LLM with access to filesystem, time, etc.
**Setup needed:** llama.cpp + this app + MCP servers
**Time to value:** 15 minutes

### Use Case 3: Research Assistant (Enhanced)
**What you get:** LLM + web search with citations + document analysis + TTS
**Setup needed:** Full integration with SearXNG + embedding + reranking + TTS
**Time to value:** 2-3 hours

## 🎨 Customization Points

### Easy Customizations
- ✅ Change UI colors/theme
- ✅ Modify conversation title length
- ✅ Add custom CSS animations
- ✅ Change default settings
- ✅ Adjust model parameters (temp, max_tokens)

### Medium Customizations
- ⚙️ Add document upload
- ⚙️ Implement user authentication
- ⚙️ Add conversation export
- ⚙️ Custom system prompts
- ⚙️ Configure TTS voices and settings

### Advanced Customizations
- 🔧 Multi-modal support
- 🔧 Custom embedding pipeline
- 🔧 Advanced tool orchestration
- 🔧 Custom search algorithms
- 🔧 Vector database integration

## 🐛 Troubleshooting Quick Reference

| Issue | Solution File | Section |
|-------|---------------|---------|
| Can't connect to llama.cpp | `QUICKSTART.md` | Troubleshooting |
| MCP server won't start | `README.md` | MCP Server Setup |
| No progress updates | `DEVELOPMENT.md` | Streaming with Tool Results |
| Search tool integration | `SEARXNG_INTEGRATION.md` | Integration Steps |
| Customizing UI | `DEVELOPMENT.md` | Change UI Theme/Colors |
| TTS not working | `QUICKSTART.md` | TTS troubleshooting |
| Document processing fails | `DEVELOPMENT.md` | Document Processing |

## 📈 Performance Characteristics

### Expected Latency
- Initial page load: **< 1s**
- Message send: **< 100ms**
- LLM first token: **200-500ms** (depends on model)
- Tool execution: **1-10s** (depends on tool)
- Web search: **5-15s** (depends on results)
- Document processing: **10-60s** (depends on size)
- Database query: **< 50ms**

### Scalability
- **Current:** Single-user, local deployment
- **Potential:** Multi-user with PostgreSQL + Redis
- **Bottleneck:** llama.cpp inference speed

## 🔐 Security Considerations

### Current State (Local Use)
- ✅ No authentication (single user)
- ✅ Local database
- ✅ No external API keys exposed by default

### Production Recommendations
- 🔒 Add JWT authentication
- 🔒 Use HTTPS
- 🔒 Validate MCP server inputs
- 🔒 Rate limiting
- 🔒 Input sanitization
- 🔒 Secure file uploads
- 🔒 Environment variable management

## 🎯 Next Steps After Setup

1. ✅ **Complete QUICKSTART.md** - Get basic chat working
2. ✅ **Add one MCP server** - Test tool integration
3. ✅ **Explore enhanced features** - Web search, RAG, TTS
4. ✅ **Customize UI** - Make it yours
5. ✅ **Configure models** - Optimize for your use case

## 💡 Tips for Success

### For First-Time Setup
- Start with the simplest config
- Test llama.cpp separately first
- Use small model for testing
- Check logs frequently

### For Development
- Enable DEBUG mode
- Use browser DevTools
- Test tools independently
- Read the source code

### For Integration
- Start with one custom tool
- Test progress updates thoroughly
- Handle errors gracefully
- Document your changes

## 🤝 Community & Support

While this is a standalone project, here are resources:

- **MCP Protocol:** https://modelcontextprotocol.io/
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **FastAPI:** https://fastapi.tiangolo.com/
- **Alpine.js:** https://alpinejs.dev/
- **SearXNG:** https://searxng.org/
- **Edge TTS:** https://github.com/rany2/edge-tts

## 📝 License

MIT License - Use freely for personal or commercial projects!

---

**Ready to start?** → Open `QUICKSTART.md`

**Need help?** → Check the appropriate guide above

**Want to customize?** → See `DEVELOPMENT.md`

**Model configuration?** → See `LLM_MODELS_REFERENCE.md`
# LLM UI with MCP Support

A real-time chat interface for LLMs with MCP (Model Context Protocol), RAG, and TTS support. Built with FastAPI, Alpine.js, and Jinja2 templates.

## Features

✨ **Core Features:**
- 💬 Real-time streaming chat interface
- 🔌 Full MCP (Model Context Protocol) support — tools discovered dynamically from connected servers
- 📊 Live tool execution progress with granular status updates
- 💾 Persistent conversation history with SQLite
- 🎨 Modern, responsive UI with Tailwind CSS
- 📝 Markdown rendering support
- 🔍 Source citation display for MCP tool results

✨ **Advanced Capabilities:**
- Real-time Server-Sent Events (SSE) for streaming responses
- Custom tools with progress tracking (document analysis, speech generation)
- Multiple MCP server support with dynamic tool discovery
- Automatic conversation title generation
- Tool call tracking and visualization
- **RAG (Retrieval-Augmented Generation)** with document indexing and querying
- **Document Processing** with support for PDF, DOCX, TXT, MD, JSON, YAML
- **Text-to-Speech (TTS)** with multiple engine support (edge-tts, pyttsx3, kokoro)
- **Knowledge Base** with document upload and management
- **Model Selection** with dynamic model switching
- **Thinking Models Support** with collapsible reasoning display
- **Advanced Settings** with configurable parameters

## Architecture

```
├── backend/
│   ├── app/main.py              # FastAPI app, all endpoints, SSE streaming
│   ├── settings.py              # SettingsManager singleton
│   ├── database/
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── crud.py              # Async CRUD helpers
│   │   └── backup.py            # Database backup scheduler
│   ├── llm_client/client.py     # llama.cpp OpenAI-compatible streaming client
│   ├── mcp_client/client.py     # MCPClientManager — FastMCP-based
│   └── tools/
│       ├── tool_executor.py     # Tool dispatch + progress events
│       ├── rag_service.py       # RAG pipeline (chunk → embed → search)
│       ├── tts_service.py       # Edge TTS / pyttsx3 / Kokoro
│       └── base.py              # Embedding + reranking helpers
├── frontend/
│   ├── static/
│   │   ├── css/theme.css        # Custom theme + modal styles
│   │   └── js/
│   │       ├── main.js          # ES module entry point
│   │       ├── utils.js         # API client, markdown, formatters
│   │       ├── services/
│   │       │   ├── sse.js       # SSEService
│   │       │   └── tts.js       # TTSService
│   │       └── components/
│   │           ├── chat.js      # Message send, SSE, tool calls
│   │           ├── sidebar.js   # Conversations, resize, filters
│   │           └── settings.js  # Settings + MCP management
│   └── templates/
│       ├── base.html            # Tailwind + Alpine CDN, store data
│       ├── index.html           # Main layout + all modals
│       └── partials/
│           ├── sidebar.html
│           ├── main_chat.html
│           ├── settings_modal.html
│           ├── agents_modal.html
│           ├── mcp_modal.html
│           ├── documents_modal.html
│           └── notes_modal.html
├── run.py                       # Launches uvicorn
└── settings.json                # Runtime settings
```

## Prerequisites

- Python 3.10+
- llama.cpp running with OpenAI-compatible API (default: http://localhost:8080)
- Node.js (for MCP servers)
- Additional dependencies for enhanced features:
  - **edge-tts** for high-quality text-to-speech (optional: `pip install edge-tts`)
  - **pyttsx3** for offline TTS (optional)
  - **Kokoro** for local high-quality TTS (optional)
  - **PyPDF2** for PDF processing (included in requirements)
  - **python-docx** for DOCX processing (included in requirements)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llm-ui-app
```

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Install Optional Dependencies

For enhanced features:

```bash
# For high-quality text-to-speech
pip install edge-tts

# For offline text-to-speech (alternative)
pip install pyttsx3

# For Kokoro TTS (high-quality local TTS)
pip install kokoro
```

### 4. Configure llama.cpp

Make sure llama.cpp is running with the server endpoint:

```bash
# Example llama.cpp server command
./llama-server -m /path/to/model.gguf --port 8080 --host 0.0.0.0 --embeddings
```

**Note:** For embedding and reranking features, ensure your llama.cpp server supports the `/v1/embeddings` and `/v1/rerank` endpoints.

If your llama.cpp is running on a different host/port, update the configuration in `settings.json`:

```json
{
  "llama_cpp_base_url": "http://localhost:8080"
}
```

### 5. Run the Application

```bash
cd ..
python run.py
```

The application will be available at: **http://localhost:8000**

## MCP Server Setup

### Adding MCP Servers via UI

1. Click the "MCP Servers" button in the sidebar
2. Fill in the server details:
   - **Name**: Unique identifier (e.g., "filesystem")
   - **Command**: Executable command (e.g., "npx")
   - **Arguments**: JSON array of arguments (e.g., `["-y", "@modelcontextprotocol/server-filesystem"]`)
3. Click "Add Server"

### Example MCP Servers

**Filesystem Server:**
```json
{
  "name": "filesystem",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
}
```

**GitHub Server:**
```json
{
  "name": "github",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_TOKEN": "your_github_token_here"
  }
}
```

**Google Drive Server:**
```json
{
  "name": "gdrive",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-gdrive"]
}
```

## Advanced Features

### RAG (Retrieval-Augmented Generation)

Upload documents to create a knowledge base that the LLM can reference:

1. Navigate to the "Knowledge Base" section in the sidebar
2. Upload documents (PDF, DOCX, TXT, MD, JSON, YAML)
3. The system automatically processes and indexes the documents
4. Enable "Search Documents" when chatting to query your knowledge base

### Text-to-Speech (TTS)

Convert text responses to speech using multiple engines:

- **Edge TTS**: High-quality online service (requires internet)
- **Pyttsx3**: Offline service (lower quality but works without internet)
- **Kokoro**: High-quality local TTS (~300MB model download)

Configure TTS in the Settings modal or via environment variables:
```bash
TTS_ENGINE=edge-tts  # or pyttsx3, kokoro
TTS_VOICE=en-US-JennyNeural  # Voice ID
```

### Model Selection

Switch between different models dynamically:
- Select from available models in the dropdown menu
- Models are automatically detected from your LLM server
- Settings are preserved between sessions

### Thinking Models Support

The application supports reasoning models that separate thinking from responses:
- Thinking content is displayed in a collapsible section
- Helps understand the model's reasoning process



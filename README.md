# LLM UI with MCP Support

A modern, real-time chat interface for Large Language Models with Model Context Protocol (MCP) support. Built with FastAPI, HTMX, Alpine.js, and designed to work with llama.cpp. Enhanced with advanced features including RAG, web search, document processing, and text-to-speech.

*This is a completely vibe coded project*

## Features

✨ **Core Features:**
- 💬 Real-time streaming chat interface
- 🔌 Full MCP (Model Context Protocol) support
- 📊 Live tool execution progress with granular status updates
- 💾 Persistent conversation history with SQLite
- 🎨 Modern, responsive UI with Tailwind CSS
- 📝 Markdown rendering support
- 🔍 Source citation display for web searches

✨ **Advanced Capabilities:**
- Real-time Server-Sent Events (SSE) for streaming responses
- Custom tools with progress tracking (search, document analysis)
- Multiple MCP server support with dynamic tool discovery
- Automatic conversation title generation
- Tool call tracking and visualization
- **Enhanced Web Search** with SearXNG integration and semantic reranking
- **RAG (Retrieval-Augmented Generation)** with document indexing and querying
- **Document Processing** with support for PDF, DOCX, TXT, and other formats
- **Text-to-Speech (TTS)** with multiple engine support (edge-tts, pyttsx3)
- **Knowledge Base** with document upload and management
- **Model Selection** with dynamic model switching
- **Thinking Models Support** with collapsible reasoning display
- **Advanced Settings** with configurable parameters

## Architecture

```
llm-ui-app/
├── backend/
│   ├── app/
│   │   └── main.py              # FastAPI application with SSE streaming
│   ├── database/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── crud.py              # Database operations
│   ├── mcp_client/
│   │   └── client.py            # MCP client manager
│   ├── llm_client/
│   │   └── client.py            # llama.cpp client
│   └── tools/
│       ├── tool_executor.py     # Tool execution with progress tracking
│       ├── searxng_tool.py      # SearXNG web search integration
│       ├── rag_service.py       # RAG (Retrieval-Augmented Generation)
│       ├── tts_service.py       # Text-to-Speech service
│       └── base.py              # Shared utilities (embeddings, reranking)
└── frontend/
    ├── static/
    │   ├── css/
    │   │   └── styles.css       # Custom styles
    │   └── js/
    │       └── app.js           # Alpine.js app logic
    └── templates/
        └── index.html           # Main HTML template with HTMX/Alpine.js
```

## Prerequisites

- Python 3.10+
- llama.cpp running with OpenAI-compatible API (default: http://localhost:8080)
- Node.js (for MCP servers)
- Additional dependencies for enhanced features:
  - **SearXNG** for web search (optional)
  - **edge-tts** for high-quality text-to-speech (optional: `pip install edge-tts`)
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
```

### 4. Configure llama.cpp

Make sure llama.cpp is running with the server endpoint:

```bash
# Example llama.cpp server command
./llama-server -m /path/to/model.gguf --port 8080 --host 0.0.0.0 --embeddings
```

**Note:** For embedding and reranking features, ensure your llama.cpp server supports the `/v1/embeddings` and `/v1/rerank` endpoints.

If your llama.cpp is running on a different host/port, update the configuration in `backend/config.py`:

```python
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
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

## Enhanced Features

### Web Search with SearXNG

The application integrates with SearXNG for privacy-focused web search. To use this feature:

1. Set up a SearXNG instance (local or remote)
2. Update the configuration in `backend/config.py`:
   ```python
   SEARXNG_URL = "http://localhost:8888/search"  # Your SearXNG instance
   ```

The search tool performs semantic search with:
- Multi-query generation for better coverage
- Adaptive query generation based on initial results
- Content extraction and chunking
- Semantic similarity filtering
- Re-ranking for relevance

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

Configure TTS in the Settings modal or via environment variables:
```bash
TTS_ENGINE=edge-tts  # or pyttsx3
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

## Real-time Progress Updates

The app implements a hybrid approach for tool execution:

### Standard MCP Tools
- Execute via MCP protocol
- Show start/complete status
- No intermediate progress (MCP limitation)

### Custom Tools with Progress
Custom tools (like `search_web`, `query_documents`, and `generate_speech`) provide granular progress updates:

```python
async for progress in tool_executor.execute_tool(tool_name, args, request_id):
    # Progress updates streamed to UI via SSE
    # {
    #   "type": "tool_progress",
    #   "tool": "search_web",
    #   "status": "Parsing website 3/10...",
    #   "progress": 45,
    #   "data": {"parsing": 3, "total": 10}
    # }
```

## Customizing Tools

### Adding a Custom Tool with Progress

Edit `backend/tools/tool_executor.py`:

```python
self.custom_tools = {
    "search_web": self._search_web_with_progress,
    "query_documents": self._query_documents_with_progress,
    "generate_speech": self._generate_speech_with_progress,
    "your_tool": self._your_tool_with_progress,  # Add your tool
}

async def _your_tool_with_progress(
    self,
    arguments: Dict[str, Any],
    request_id: str
) -> AsyncGenerator[Dict, None]:
    """Your custom tool implementation"""

    # Yield progress updates
    yield {
        "type": "tool_progress",
        "tool": "your_tool",
        "status": "Starting...",
        "progress": 0
    }

    # ... your logic ...

    yield {
        "type": "tool_progress",
        "tool": "your_tool",
        "status": "Complete",
        "progress": 100,
        "result": {"data": "your result"}
    }
```

## Database Schema

The app uses SQLite with the following tables:

- **conversations**: Chat conversations
- **messages**: Individual messages
- **mcp_servers**: MCP server configurations
- **documents**: Uploaded document metadata
- **document_chunks**: Indexed document chunks for RAG
- **document_embeddings**: Embeddings for document chunks

Database file: `llm_ui.db` (created automatically on first run)

## API Endpoints

### Conversations
- `GET /api/conversations` - List all conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation details

### Messages
- `POST /api/conversations/{id}/messages` - Send message
- `GET /api/stream/{request_id}` - Stream LLM response (SSE)
- `PUT /api/messages/{message_id}` - Edit message content
- `DELETE /api/messages/{message_id}` - Delete message

### MCP Servers
- `GET /api/mcp/servers` - List MCP servers
- `POST /api/mcp/servers` - Add MCP server
- `DELETE /api/mcp/servers/{name}` - Remove MCP server
- `GET /api/mcp/tools` - List all available tools

### Documents & RAG
- `POST /api/documents/upload` - Upload document for RAG
- `GET /api/documents` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/rag/query` - Direct RAG query endpoint

### Web Search
- `POST /api/search/web` - Direct web search endpoint

### Text-to-Speech
- `POST /api/tts/generate` - Generate speech from text
- `GET /api/tts/voices` - List available TTS voices
- `GET /api/tts/status` - Check TTS availability
- `GET /api/audio/{filename}` - Serve generated audio files

### Settings
- `GET /api/settings` - Get application settings
- `PUT /api/settings` - Update application settings

### Models
- `GET /api/models` - List available models from LLM server

## Development

### Project Structure

**Backend:**
- `main.py`: FastAPI app with SSE streaming
- `models.py`: Database models
- `crud.py`: Database operations
- `client.py` (mcp_client): MCP protocol implementation
- `client.py` (llm_client): llama.cpp integration
- `tool_executor.py`: Tool execution with progress
- `searxng_tool.py`: Web search integration
- `rag_service.py`: RAG (Retrieval-Augmented Generation)
- `tts_service.py`: Text-to-Speech service
- `base.py`: Shared utilities (embeddings, reranking)

**Frontend:**
- `index.html`: Main UI with HTMX and Alpine.js
- `app.js`: Alpine.js reactive state management
- `styles.css`: Custom styling

### Adding New Features

1. **Add a new database table**: Edit `models.py`
2. **Add CRUD operations**: Edit `crud.py`
3. **Add API endpoint**: Edit `main.py`
4. **Update UI**: Edit `index.html` and `app.js`

## Integrating Your SearXNG Tool

The application already includes a sophisticated SearXNG integration with:
- Multi-query generation for better coverage
- Adaptive query generation based on initial results
- Content extraction and semantic chunking
- Embedding-based similarity filtering
- Re-ranking for relevance
- Citation support with source tracking

The integration is located in `backend/tools/searxng_tool.py`.

## Troubleshooting

**Issue:** llama.cpp connection refused
- **Solution:** Ensure llama.cpp server is running on port 8080

**Issue:** MCP server fails to start
- **Solution:** Check that Node.js is installed and the MCP package exists

**Issue:** Database errors
- **Solution:** Delete `llm_ui.db` and restart (will recreate schema)

**Issue:** SSE connection drops
- **Solution:** Check firewall settings and proxy configurations

**Issue:** Web search not working
- **Solution:** Verify SearXNG instance is accessible and URL is correct in config

**Issue:** TTS not working
- **Solution:** Install edge-tts (`pip install edge-tts`) or pyttsx3 (`pip install pyttsx3`)

**Issue:** Document processing fails
- **Solution:** Check if PyPDF2 and python-docx are installed (they're in requirements.txt)

## Future Enhancements

- [ ] Multi-modal support (images)
- [ ] Export conversations
- [ ] User authentication
- [ ] Tool usage analytics
- [ ] Custom system prompts
- [ ] Conversation branching
- [ ] Advanced RAG with vector databases
- [ ] Plugin system for custom tools

## License

MIT License - feel free to modify and use for your projects!

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI powered by [HTMX](https://htmx.org/) and [Alpine.js](https://alpinejs.dev/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- LLM backend: [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Protocol: [Model Context Protocol](https://modelcontextprotocol.io/)
- Web search: [SearXNG](https://searxng.org/)
- TTS: [Edge TTS](https://github.com/rany2/edge-tts)

# LLM UI with MCP Support

A modern, real-time chat interface for Large Language Models with Model Context Protocol (MCP) support. Built with FastAPI, HTMX, Alpine.js, and designed to work with llama.cpp.

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

## Architecture

```
llm-ui-app/
├── backend/
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── database/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── crud.py              # Database operations
│   ├── mcp_client/
│   │   └── client.py            # MCP client manager
│   ├── llm_client/
│   │   └── client.py            # llama.cpp client
│   └── tools/
│       └── tool_executor.py     # Tool execution with progress
└── frontend/
    ├── static/
    │   ├── css/
    │   │   └── styles.css       # Custom styles
    │   └── js/
    │       └── app.js           # Alpine.js app logic
    └── templates/
        └── index.html           # Main HTML template
```

## Prerequisites

- Python 3.10+
- llama.cpp running with OpenAI-compatible API (default: http://localhost:8080)
- Node.js (for MCP servers)

## Installation

### 1. Clone the Repository

```bash
cd llm-ui-app/backend
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure llama.cpp

Make sure llama.cpp is running with the server endpoint:

```bash
# Example llama.cpp server command
./llama-server -m /path/to/model.gguf --port 8080 --host 0.0.0.0
```

If your llama.cpp is running on a different host/port, update the `LLMClient` configuration in `backend/llm_client/client.py`:

```python
def __init__(self, base_url: str = "http://localhost:8080"):
```

### 4. Run the Application

```bash
cd backend
python -m app.main
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

## Real-time Progress Updates

The app implements a hybrid approach for tool execution:

### Standard MCP Tools
- Execute via MCP protocol
- Show start/complete status
- No intermediate progress (MCP limitation)

### Custom Tools with Progress
Custom tools (like `search_web` and `analyze_document`) provide granular progress updates:

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
    "analyze_document": self._analyze_document_with_progress,
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

Database file: `llm_ui.db` (created automatically on first run)

## API Endpoints

### Conversations
- `GET /api/conversations` - List all conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation details

### Messages
- `POST /api/conversations/{id}/messages` - Send message
- `GET /api/stream/{request_id}` - Stream LLM response (SSE)

### MCP Servers
- `GET /api/mcp/servers` - List MCP servers
- `POST /api/mcp/servers` - Add MCP server
- `DELETE /api/mcp/servers/{name}` - Remove MCP server
- `GET /api/mcp/tools` - List all available tools

## Development

### Project Structure

**Backend:**
- `main.py`: FastAPI app with SSE streaming
- `models.py`: Database models
- `crud.py`: Database operations
- `client.py` (mcp_client): MCP protocol implementation
- `client.py` (llm_client): llama.cpp integration
- `tool_executor.py`: Tool execution with progress

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

To integrate your existing SearXNG search tool:

1. Create a new file `backend/tools/searxng_tool.py`
2. Implement the search pipeline with progress updates
3. Register it in `tool_executor.py`:

```python
from tools.searxng_tool import search_with_progress

self.custom_tools = {
    "search_web": search_with_progress,
}
```

## Troubleshooting

**Issue:** llama.cpp connection refused
- **Solution:** Ensure llama.cpp server is running on port 8080

**Issue:** MCP server fails to start
- **Solution:** Check that Node.js is installed and the MCP package exists

**Issue:** Database errors
- **Solution:** Delete `llm_ui.db` and restart (will recreate schema)

**Issue:** SSE connection drops
- **Solution:** Check firewall settings and proxy configurations

## Future Enhancements

- [ ] Document upload and processing
- [ ] Multi-modal support (images)
- [ ] Export conversations
- [ ] User authentication
- [ ] Tool usage analytics
- [ ] Custom system prompts
- [ ] Conversation branching
- [ ] RAG integration with vector databases

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

# Development & Customization Guide

This guide covers how to customize and extend the LLM UI for your specific needs.

## Architecture Overview

### Backend Flow

```
User Request
    ↓
FastAPI Endpoint (main.py)
    ↓
Database Layer (models.py, crud.py)
    ↓
LLM Client (llm_client/client.py) ←→ llama.cpp
    ↓
Tool Executor (tools/tool_executor.py)
    ↓
MCP Client Manager (mcp_client/client.py) ←→ MCP Servers
    ↓
Server-Sent Events (SSE) Stream
    ↓
Frontend (Alpine.js)
```

### Real-time Communication

The app uses **Server-Sent Events (SSE)** for real-time updates:

1. User sends message → POST to `/api/conversations/{id}/messages`
2. Backend returns `request_id`
3. Frontend connects to SSE stream: `/api/stream/{request_id}`
4. Backend streams events:
   - `content`: Text chunks from LLM
   - `thinking`: Reasoning content from thinking models
   - `tool_call_start`: Tool execution begins
   - `tool_progress`: Progress updates (0-100%)
   - `tool_error`: Tool execution failed
   - `title_update`: Conversation title generated
   - `done`: Stream complete

## Common Customizations

### 1. Change UI Theme/Colors

Edit `frontend/static/css/styles.css`:

```css
/* Change primary color from blue to purple */
.bg-blue-600 { background-color: #9333ea !important; }
.bg-blue-700 { background-color: #7e22ce !important; }
.text-blue-600 { color: #9333ea !important; }
/* ... etc */
```

Or use Tailwind's configuration in `frontend/templates/index.html`:

```html
<script>
    tailwind.config = {
        theme: {
            extend: {
                colors: {
                    primary: '#9333ea',
                }
            }
        }
    }
</script>
```

### 2. Enhanced Document Processing (RAG)

The application already includes comprehensive document processing with RAG (Retrieval-Augmented Generation):

**Backend Components:**
- `tools/rag_service.py` - RAG service with document processing
- `tools/base.py` - Shared utilities for embeddings and reranking
- `database/models.py` - Document and chunk models

**Document Processing Flow:**
1. User uploads document via `/api/documents/upload`
2. Document is processed by `RAGService.process_document()`
3. Text is extracted and chunked
4. Embeddings are generated for each chunk
5. Chunks and embeddings are stored in SQLite
6. Documents can be queried via `/api/rag/query`

**Customize Document Processing:**
Edit `tools/rag_service.py` to modify:
- Chunk size and overlap
- Embedding model
- Similarity threshold
- Supported file formats

### 3. Text-to-Speech (TTS) Integration

The application supports multiple TTS engines:

**Backend Components:**
- `tools/tts_service.py` - TTS service with multiple backends

**Supported Engines:**
- **Edge TTS**: High-quality online service (requires internet)
- **Pyttsx3**: Offline service (lower quality but works without internet)
- **Kokoro**: High-quality local TTS (~300MB model download)

**Customize TTS:**
Edit `tools/tts_service.py` to modify:
- Default voice
- Speech rate
- Volume level
- Output format

### 4. Add System Prompts

**Database Model (`backend/database/models.py`):**

```python
class Conversation(Base):
    # ... existing fields ...
    system_prompt = Column(Text, nullable=True)
    model_settings = Column(JSON, nullable=True)  # temp, max_tokens, etc.
```

**API Endpoint:**

```python
@app.put("/api/conversations/{conversation_id}/settings")
async def update_conversation_settings(conversation_id: str, request: Request):
    data = await request.json()

    async with get_db() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = result.scalar_one_or_none()

        if conv:
            conv.system_prompt = data.get("system_prompt")
            conv.model_settings = data.get("model_settings")
```

**UI Addition:**

Add settings panel in the chat header area.

### 5. Add Conversation Export

**Backend:**

```python
@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str, format: str = "markdown"):
    """Export conversation as markdown or JSON"""

    async with get_db() as db:
        conversation = await get_conversation(db, conversation_id)
        messages = await get_conversation_messages(db, conversation_id)

    if format == "markdown":
        content = f"# {conversation['title']}\n\n"
        for msg in messages:
            content += f"**{msg['role'].title()}**: {msg['content']}\n\n"

        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={conversation['title']}.md"}
        )

    elif format == "json":
        return JSONResponse({
            "conversation": conversation,
            "messages": messages
        })
```

### 6. Add Conversation Search

**Backend:**

```python
@app.get("/api/conversations/search")
async def search_conversations(q: str):
    """Search conversations by content"""

    async with get_db() as db:
        # Search in messages
        result = await db.execute(
            select(Message)
            .where(Message.content.like(f"%{q}%"))
            .limit(20)
        )
        messages = result.scalars().all()

        # Get unique conversation IDs
        conv_ids = list(set(msg.conversation_id for msg in messages))

        # Fetch conversations
        conversations = []
        for conv_id in conv_ids:
            conv = await get_conversation(db, conv_id)
            conversations.append(conv)

        return {"conversations": conversations, "query": q}
```

### 7. Add User Authentication

Use FastAPI's security utilities:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# Add to protected endpoints
@app.get("/api/conversations")
async def list_conversations(user=Depends(verify_token)):
    # Only return conversations for this user
    pass
```

### 8. Streaming with Tool Results

The application uses Server-Sent Events (SSE) for streaming LLM responses with tool execution:

**In `backend/app/main.py`:**

```python
async def _core_stream_handler(
    request_id: str,
    conversation_id: str,
    enable_rag: bool = False,
    model: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Universal SSE handler for streaming LLM responses and tool execution."""
    try:
        async with get_db() as db:
            messages = await get_conversation_messages(db, conversation_id)
            llm_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            tool_calls_history = []

            # Stream LLM response (LLM decides when to call tools)
            assistant_message, thinking_content = "", ""

            async for chunk in llm_client.stream_chat(llm_messages, model=model):
                chunk_type = chunk.get("type")
                if chunk_type == "content":
                    content = chunk.get("content", "")
                    assistant_message += content
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                elif chunk_type == "thinking":
                    thinking = chunk.get("content", "")
                    thinking_content += thinking
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thinking})}\n\n"
                elif chunk_type == "tool_call":
                    tc_data = chunk.get("tool_call")
                    # Send tool call start event
                    yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': tc_data['name'], 'args': tc_data['arguments']})}\n\n"
                    # Create tool call record
                    tc_record = {"name": tc_data["name"], "arguments": tc_data["arguments"], "status": "pending", "result": None, "progress_history": []}
                    tool_calls_history.append(tc_record)

                    # Execute the tool
                    async for progress_event in tool_executor.execute_tool(tc_data["name"], tc_data["arguments"], request_id):
                        # Forward the progress event
                        yield f"data: {json.dumps(progress_event)}\n\n"
                        if progress_event.get("type") == "tool_progress":
                            tc_record["status"] = progress_event.get("status", "running")
                            tc_record["progress"] = progress_event.get("progress", 0)
                            # Add progress event to history
                            tc_record["progress_history"].append(progress_event)
                            if progress_event.get("result"):
                                tc_record["result"] = progress_event["result"]
                                tc_record["status"] = "completed"
                        elif progress_event.get("type") == "tool_error":
                            tc_record["status"] = "error"
                            tc_record["result"] = {"error": progress_event.get("error")}
                await asyncio.sleep(0)

            # Save assistant message
            if assistant_message.strip():
                await add_message(db, conversation_id, "assistant", assistant_message, tool_calls_history or None, thinking_content or None)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
    except Exception as e:
        print(f"Error in event generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
```

## Model Configuration

Models are configured in `backend/config.py`:

```python
# Main LLM model for chat completion
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "glm4.7-30ba3b")

# Model for query processing and title generation (should be non-thinking)
QUERY_MODEL = os.getenv("QUERY_MODEL", "qwen3-30ba3b")

# Base URL for llama.cpp server
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
```

### Supported Models

Any model compatible with llama.cpp's OpenAI-compatible API works:
- **Chat**: Llama 2/3, Mistral/Mixtral, Phi-2/Phi-3, Qwen series, Gemma, Zephyr
- **Embedding**: Any model with embedding support (e.g., `nomic-embed-text`)
- **Reranking**: Any model with reranking support
- **Thinking**: Models that output reasoning (e.g., DeepSeek-R1, QwQ)

### Optimal Settings by Model Size

| Model Size | Temperature | Max Tokens | Top-K |
|-----------|-------------|------------|-------|
| 7B-13B    | 0.7         | 4096       | 40    |
| 30B-34B   | 0.7         | 8192       | 40    |
| 70B+      | 0.7         | 16384      | 40    |

### Embedding & Reranking Models

- **Embedding**: `nomic-embed-text`, `BGE-m3`, or any embedding-compatible model
- **Reranking**: `bge-reranker-v2-m3`, `bge-reranker-large`, or any reranking-compatible model

## MCP Server Transports

The application supports three MCP transport types:

| Transport | Description | Config |
|-----------|-------------|--------|
| **stdio** | Local process via command line | `command`, `args[]`, `env{}` |
| **SSE** | HTTP SSE endpoint | `url` |
| **StreamableHTTP** | HTTP streaming | `url` |

### stdio Transport (Most Common)

```json
{
  "name": "filesystem",
  "transport_type": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
  "env": {}
}
```

### SSE Transport

```json
{
  "name": "remote-server",
  "transport_type": "sse",
  "url": "http://remote-host:3001/mcp"
}
```

### StreamableHTTP Transport

```json
{
  "name": "streamable-server",
  "transport_type": "streamable-http",
  "url": "http://remote-host:3001/mcp"
}
```

## Performance Optimization

### 1. Database Connection Pooling

For PostgreSQL (production):

```python
from sqlalchemy.pool import NullPool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool,  # or configure pool size
    echo=False  # Disable SQL logging in production
)
```

### 2. Caching

Add Redis for caching:

```python
import redis.asyncio as redis

redis_client = redis.from_url("redis://localhost:6379")

# Cache conversation messages
@app.get("/api/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    # Try cache first
    cached = await redis_client.get(f"conv:{conversation_id}")
    if cached:
        return json.loads(cached)

    # Fetch from DB
    # ... existing code ...

    # Cache for 5 minutes
    await redis_client.setex(
        f"conv:{conversation_id}",
        300,
        json.dumps(result)
    )
```

### 3. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/conversations/{conversation_id}/messages")
@limiter.limit("10/minute")
async def send_message(request: Request, ...):
    # ... existing code ...
```

## Testing

### Unit Tests

Create `backend/tests/test_tools.py`:

```python
import pytest
from tools.tool_executor import ToolExecutor

@pytest.mark.asyncio
async def test_query_documents_tool():
    executor = ToolExecutor(None)

    results = []
    async for progress in executor.execute_tool(
        "query_documents",
        {"query": "test query"},
        "test-request-id"
    ):
        results.append(progress)

    assert any(r["type"] == "tool_progress" for r in results)
    assert results[-1]["progress"] == 100
```

### Integration Tests

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_conversation():
    response = client.post(
        "/api/conversations",
        json={"title": "Test Chat"}
    )
    assert response.status_code == 200
    assert "conversation" in response.json()
```

## Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies
RUN pip install edge-tts

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY run.py .

EXPOSE 8000

CMD ["python", "run.py"]
```

### Using systemd (Linux)

Create `/etc/systemd/system/llm-ui.service`:

```ini
[Unit]
Description=LLM UI Application
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/llm-ui-app
ExecStart=/usr/bin/python3 run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Debugging

### Enable Debug Mode

Set in `.env`:
```
DEBUG=true
```

### Check Logs

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Browser DevTools

Press F12 in browser:
- **Console**: JavaScript errors
- **Network**: API requests/responses
- **Application**: LocalStorage, cookies

## Contributing

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Alpine.js Documentation](https://alpinejs.dev/)
- [HTMX Documentation](https://htmx.org/)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Edge TTS Documentation](https://github.com/rany2/edge-tts)
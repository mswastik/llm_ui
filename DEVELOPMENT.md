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

### 2. Add Document Upload Support

**Backend (`backend/app/main.py`):**

```python
from fastapi import UploadFile, File
from database.crud import create_document

@app.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create database record
    async with get_db() as db:
        doc = await create_document(
            db,
            filename=file.filename,
            filepath=file_path,
            file_type=file.content_type,
            size_bytes=len(content)
        )
    
    # Trigger async processing
    asyncio.create_task(process_document(doc["id"]))
    
    return {"document_id": doc["id"], "status": "processing"}


async def process_document(document_id: str):
    """Process uploaded document asynchronously"""
    # Your document processing logic
    # Extract text, generate embeddings, etc.
    pass
```

**Frontend (`frontend/templates/index.html`):**

Add file input in the chat input area:

```html
<form @submit.prevent="sendMessage()" class="flex gap-3">
    <label class="cursor-pointer">
        <input type="file" class="hidden" @change="handleFileUpload($event)">
        <svg class="w-6 h-6 text-gray-500 hover:text-gray-700">
            <!-- Paperclip icon -->
        </svg>
    </label>
    <textarea ...></textarea>
    <button type="submit">...</button>
</form>
```

**Alpine.js (`frontend/static/js/app.js`):**

```javascript
async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/upload/document', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    this.inputMessage = `Analyze the document I just uploaded: ${file.name}`;
}
```

### 3. Add System Prompts

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

### 4. Add Voice Input

Use browser's Web Speech API:

```html
<button @click="startVoiceInput()" class="...">
    <svg><!-- Microphone icon --></svg>
</button>
```

```javascript
startVoiceInput() {
    if (!('webkitSpeechRecognition' in window)) {
        alert('Voice input not supported in this browser');
        return;
    }
    
    const recognition = new webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        this.inputMessage = transcript;
    };
    
    recognition.start();
}
```

### 5. Export Conversations

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

Currently, tool results aren't fed back to the LLM. To enable this:

**In `backend/app/main.py`:**

```python
async def stream_response(request_id: str, conversation_id: str):
    async def event_generator() -> AsyncGenerator[str, None]:
        # ... existing code ...
        
        tool_results = []
        
        async for chunk in llm_client.stream_chat(llm_messages):
            if chunk_type == "tool_call":
                tool_call = chunk.get("tool_call")
                
                # Execute tool
                async for progress in tool_executor.execute_tool(...):
                    yield f"data: {json.dumps(progress)}\n\n"
                    
                    if progress.get("type") == "tool_progress" and progress.get("result"):
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": json.dumps(progress["result"])
                        })
                
                # Continue LLM generation with tool results
                llm_messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
                llm_messages.extend(tool_results)
                
                # Stream continuation
                async for continuation_chunk in llm_client.stream_chat(llm_messages):
                    # Process continuation...
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
async def test_search_tool():
    executor = ToolExecutor(None)
    
    results = []
    async for progress in executor.execute_tool(
        "search_web",
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

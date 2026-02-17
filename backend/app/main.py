from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import asyncio
import json
import os
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from config import APP_HOST, APP_PORT, DEBUG, MAX_UPLOAD_SIZE, UPLOAD_DIR
from database.models import init_db, get_db
from database.crud import (
    create_conversation, get_conversation, get_all_conversations,
    add_message, get_conversation_messages, update_conversation_title,
    delete_conversation as db_delete_conversation,
    update_message, get_message, create_document, update_document_status, get_documents,
    delete_message as db_delete_message, delete_document as db_delete_document, get_document
)
from mcp_client.client import MCPClientManager
from tools.tool_executor import ToolExecutor
from llm_client.client import LLMClient
from backend.settings import settings_manager

# Initialize MCP client manager
mcp_manager = MCPClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await mcp_manager.initialize()
    yield
    # Shutdown
    await mcp_manager.cleanup()

app = FastAPI(title="LLM UI with MCP Support", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Initialize components
llm_client = LLMClient()
tool_executor = ToolExecutor(mcp_manager)

# Set TTS service in settings manager
settings_manager.set_tts_service(tool_executor.tts_service)

# Active SSE connections for real-time status updates
active_connections: Dict[str, asyncio.Queue] = {}


@app.get("/")
async def index(request: Request):
    """Render main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/conversations")
async def list_conversations():
    """Get all conversations"""
    async with get_db() as db:
        conversations = await get_all_conversations(db)
        return {"conversations": conversations}


@app.post("/api/conversations")
async def new_conversation(request: Request):
    """Create a new conversation"""
    data = await request.json()
    title = data.get("title", "New Chat")
    
    async with get_db() as db:
        conversation = await create_conversation(db, title)
        return {"conversation": conversation}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """Get conversation with messages"""
    async with get_db() as db:
        conversation = await get_conversation(db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = await get_conversation_messages(db, conversation_id)
        return {
            "conversation": conversation,
            "messages": messages
        }


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, request: Request):
    """Send a message and get LLM response"""
    data = await request.json()
    user_message = data.get("message", "")
    enable_web_search = data.get("enable_web_search", False)
    enable_rag = data.get("enable_rag", False)
    
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    async with get_db() as db:
        # Save user message
        await add_message(db, conversation_id, "user", user_message)
        
        # Create a unique request ID for this interaction
        request_id = str(uuid.uuid4())
        
        return {
            "request_id": request_id,
            "status": "processing",
            "enable_web_search": enable_web_search,
            "enable_rag": enable_rag
        }


async def _core_stream_handler(
    request_id: str,
    conversation_id: str,
    enable_web_search: bool = False,
    enable_rag: bool = False,
    model: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Universal SSE handler for streaming LLM responses and tool execution."""
    try:
        async with get_db() as db:
            messages = await get_conversation_messages(db, conversation_id)
            llm_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            context_additions = []
            tool_calls_history = []

            # Execute pre-processing tools (web search, RAG) if enabled
            if enable_web_search:
                query_text = llm_messages[-1]["content"] if llm_messages else ""
                # Send tool call start event
                yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': 'search_web', 'args': {'query': query_text}})}\n\n"
                # Create tool call record
                web_search_tool_call = {"name": "search_web", "arguments": {"query": query_text}, "status": "starting", "progress": 0, "result": None, "progress_history": []}
                tool_calls_history.append(web_search_tool_call)

                async for progress_event in tool_executor.execute_tool("search_web", {"query": query_text}, request_id):
                    # Forward the progress event
                    yield f"data: {json.dumps(progress_event)}\n\n"

                    # Update tool call status
                    if progress_event.get("type") == "tool_progress":
                        web_search_tool_call["status"] = progress_event.get("status", "running")
                        web_search_tool_call["progress"] = progress_event.get("progress", 0)
                        # Add progress event to history
                        web_search_tool_call["progress_history"].append(progress_event)
                        if progress_event.get("result"):
                            web_search_tool_call["result"] = progress_event["result"]
                            web_search_tool_call["status"] = "completed"
                            if progress_event["result"].get("content"):
                                context_additions.append(f"\n\n**Web Search Results:**\n{progress_event['result']['content']}")
                    elif progress_event.get("type") == "tool_error":
                        web_search_tool_call["status"] = "error"
                        web_search_tool_call["result"] = {"error": progress_event.get("error")}
                    await asyncio.sleep(0)

            if enable_rag:
                query_text = llm_messages[-1]["content"] if llm_messages else ""
                # Send tool call start event
                yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': 'query_documents', 'args': {'query': query_text}})}\n\n"
                # Create tool call record
                rag_tool_call = {"name": "query_documents", "arguments": {"query": query_text}, "status": "starting", "progress": 0, "result": None, "progress_history": []}
                tool_calls_history.append(rag_tool_call)

                async for progress_event in tool_executor.execute_tool("query_documents", {"query": query_text}, request_id):
                    # Forward the progress event
                    yield f"data: {json.dumps(progress_event)}\n\n"

                    # Update tool call status
                    if progress_event.get("type") == "tool_progress":
                        rag_tool_call["status"] = progress_event.get("status", "running")
                        rag_tool_call["progress"] = progress_event.get("progress", 0)
                        # Add progress event to history
                        rag_tool_call["progress_history"].append(progress_event)
                        if progress_event.get("result"):
                            rag_tool_call["result"] = progress_event["result"]
                            rag_tool_call["status"] = "completed"
                            if progress_event["result"].get("context"):
                                context_additions.append(f"\n\n**Document Search Results:**\n{progress_event['result']['context']}")
                    elif progress_event.get("type") == "tool_error":
                        rag_tool_call["status"] = "error"
                        rag_tool_call["result"] = {"error": progress_event.get("error")}
                    await asyncio.sleep(0)

            # Add context to the last user message
            if context_additions:
                for msg in reversed(llm_messages):
                    if msg["role"] == "user":
                        msg["content"] += "".join(context_additions) + "\n\n**Important**: Please cite sources using [1], [2], etc."
                        break

            # Build list of tools to exclude (already executed as pre-processing)
            exclude_tools = []
            if enable_web_search:
                exclude_tools.append("search_web")
            if enable_rag:
                exclude_tools.append("query_documents")

            # Stream LLM response
            assistant_message, thinking_content = "", ""
            
            # Get MCP tools for LLM function calling
            mcp_tools = []
            if mcp_manager:
                mcp_tools = await mcp_manager.list_all_tools()
            
            async for chunk in llm_client.stream_chat(llm_messages, model=model, tools=tool_executor.get_tool_definitions(exclude_tools=exclude_tools, mcp_tools=mcp_tools)):
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
                # Add model info to message metadata
                message_extra_metadata = {"model": model} if model else {}
                await add_message(db, conversation_id, "assistant", assistant_message, tool_calls_history or None, thinking_content or None, extra_metadata=message_extra_metadata)

            # Title Generation Logic (only for first exchange)
            messages_after_save = await get_conversation_messages(db, conversation_id)
            user_count = len([m for m in messages_after_save if m["role"] == "user"])
            assistant_count = len([m for m in messages_after_save if m["role"] == "assistant"])

            if user_count == 1 and assistant_count == 1:
                first_user_message = next((m for m in messages_after_save if m["role"] == "user"), None)
                if first_user_message:
                    try:
                        # Use QUERY_MODEL for title generation to avoid issues with thinking models
                        from config import QUERY_MODEL
                        title = await asyncio.wait_for(llm_client.generate_title(first_user_message["content"], model=QUERY_MODEL), timeout=40.0)
                        await update_conversation_title(db, conversation_id, title)
                        yield f"data: {json.dumps({'type': 'title_update', 'title': title})}\n\n"
                    except Exception as e:
                        print(f"Error generating or updating title: {e}")

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except asyncio.CancelledError:
        print(f"Event generator cancelled for request {request_id}")
    except Exception as e:
        print(f"Error in event generator: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    finally:
        if request_id in active_connections:
            del active_connections[request_id]


@app.get("/api/stream/{request_id}")
async def stream_response(
    request_id: str,
    conversation_id: str,
    enable_web_search: bool = False,
    enable_rag: bool = False,
    model: str = None
):
    """Stream LLM response with real-time tool execution updates."""
    return StreamingResponse(
        _core_stream_handler(request_id, conversation_id, enable_web_search, enable_rag, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/stream/regenerate/{request_id}")
async def stream_regenerate_response(request_id: str, conversation_id: str, model: str = None):
    """Stream regenerated LLM response using unified handler."""
    return StreamingResponse(
        _core_stream_handler(request_id, conversation_id, False, False, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# MCP Server Management
@app.get("/api/mcp/servers")
async def list_mcp_servers():
    """List all available MCP servers"""
    servers = await mcp_manager.list_servers()
    return {"servers": servers}


@app.post("/api/mcp/servers")
async def add_mcp_server(request: Request):
    """Add a new MCP server"""
    data = await request.json()
    name = data.get("name")
    command = data.get("command")
    args = data.get("args", [])
    env = data.get("env", {})
    transport_type = data.get("transport_type", "stdio")
    url = data.get("url")
    
    # Validate based on transport type
    if transport_type in ("sse", "streamable-http"):
        if not url:
            raise HTTPException(status_code=400, detail="URL is required for SSE/StreamableHTTP transport")
    elif transport_type == "stdio":
        if not command:
            raise HTTPException(status_code=400, detail="Command is required for stdio transport")

    success = await mcp_manager.add_server(name, command, args, env, transport_type, url)

    if success:
        return {"status": "success", "message": f"Server '{name}' added successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add server")


@app.delete("/api/mcp/servers/{server_name}")
async def remove_mcp_server(server_name: str):
    """Remove an MCP server"""
    success = await mcp_manager.remove_server(server_name)
    
    if success:
        return {"status": "success", "message": f"Server '{server_name}' removed"}
    else:
        raise HTTPException(status_code=404, detail="Server not found")


# Conversation Management
@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    async with get_db() as db:
        await db_delete_conversation(db, conversation_id)
        return {"status": "success", "message": "Conversation deleted"}


@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: Request):
    """Update a conversation title"""
    data = await request.json()
    title = data.get("title", "")
    
    if not title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    async with get_db() as db:
        await update_conversation_title(db, conversation_id, title)
        return {"status": "success", "message": "Conversation updated"}


@app.put("/api/messages/{message_id}")
async def edit_message(message_id: str, request: Request):
    """Edit a message's content"""
    data = await request.json()
    content = data.get("content", "")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    async with get_db() as db:
        message = await update_message(db, message_id, content)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return {"message": message}


@app.delete("/api/messages/{message_id}")
async def delete_message_endpoint(message_id: str):
    """Delete a message"""
    async with get_db() as db:
        success = await db_delete_message(db, message_id)
        if not success:
            raise HTTPException(status_code=404, detail="Message not found")
        return {"status": "success", "message": "Message deleted"}


@app.post("/api/conversations/{conversation_id}/regenerate")
async def regenerate_last_response(conversation_id: str, request: Request):
    """Regenerate the last assistant response"""
    data = await request.json()
    message_id = data.get("message_id")
    
    async with get_db() as db:
        # Get the message to regenerate
        if message_id:
            message = await get_message(db, message_id)
            if not message:
                raise HTTPException(status_code=404, detail="Message not found")
            if message.get("role") != "assistant":
                raise HTTPException(status_code=400, detail="Can only regenerate assistant messages")
        else:
            # Get last assistant message if no message_id provided
            messages = await get_conversation_messages(db, conversation_id)
            # Find last assistant message
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            if not assistant_messages:
                raise HTTPException(status_code=400, detail="No assistant message to regenerate")
            message = assistant_messages[-1]
        
        # Find the user message that preceded this assistant message
        messages = await get_conversation_messages(db, conversation_id)
        msg_index = next((i for i, m in enumerate(messages) if m.get("id") == message.get("id") or m.get("id") == message_id), -1)
        
        if msg_index > 0:
            user_message = messages[msg_index - 1]
            if user_message.get("role") == "user":
                # Delete the old assistant message
                await db_delete_message(db, message.get("id"))
                
                # Get all messages up to the user message (excluding the old assistant response)
                messages_to_keep = messages[:msg_index]
                
                # Create new request ID
                request_id = str(uuid.uuid4())
                
                return {"request_id": request_id, "status": "processing", "conversation_id": conversation_id}
        
        raise HTTPException(status_code=400, detail="Could not find preceding user message")


# Document Management
@app.post("/api/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a document to the knowledgebase and process it for RAG"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to start
    
    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_UPLOAD_SIZE} bytes")
    
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Determine file type
    file_type = "unknown"
    if file_ext in [".txt", ".md"]:
        file_type = "text"
    elif file_ext in [".pdf"]:
        file_type = "pdf"
    elif file_ext in [".doc", ".docx"]:
        file_type = "document"
    elif file_ext in [".json", ".yaml", ".yml"]:
        file_type = "data"
    
    # Create document record
    async with get_db() as db:
        document = await create_document(
            db,
            filename=file.filename,
            filepath=file_path,
            file_type=file_type,
            size_bytes=file_size,
            metadata={"original_filename": file.filename}
        )
        
        # Mark as processing
        await update_document_status(db, document["id"], "processing", {})
        
        document_id = document["id"]
    
    # Process document for RAG in background
    background_tasks.add_task(
        process_document_background,
        document_id,
        file_path,
        file_type
    )
    
    return {
        "status": "processing",
        "document": {
            "id": document_id,
            "filename": file.filename,
            "file_type": file_type,
            "size_bytes": file_size,
            "status": "processing"
        }
    }


async def process_document_background(document_id: str, file_path: str, file_type: str):
    """Background task to process document for RAG"""
    async with get_db() as db:
        try:
            # Process document for RAG
            result = await tool_executor.process_document_for_rag(
                document_id=document_id,
                filepath=file_path,
                file_type=file_type
            )
            
            if result.get("success"):
                await update_document_status(
                    db,
                    document_id,
                    "completed",
                    {"chunks": result.get("chunk_count", 0)}
                )
            else:
                await update_document_status(
                    db,
                    document_id,
                    "failed",
                    {"error": result.get("error", "Unknown error")}
                )
        except Exception as e:
            await update_document_status(
                db,
                document_id,
                "failed",
                {"error": str(e)}
            )


@app.get("/api/documents")
async def list_documents():
    """List all documents in knowledgebase"""
    async with get_db() as db:
        documents = await get_documents(db)
        return {"documents": documents}


@app.get("/api/documents/{document_id}")
async def get_document_detail(document_id: str):
    """Get document details"""
    async with get_db() as db:
        document = await get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"document": document}


@app.delete("/api/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document from knowledgebase"""
    async with get_db() as db:
        # Get document to find filepath
        document = await get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from RAG index
        tool_executor.delete_document_from_rag(document_id)
        
        # Delete file from disk
        try:
            if os.path.exists(document["filepath"]):
                os.remove(document["filepath"])
        except Exception as e:
            print(f"Error deleting file: {e}")
        
        # Delete from database
        success = await db_delete_document(db, document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "success", "message": "Document deleted"}


@app.get("/api/mcp/tools")
async def list_available_tools():
    """List all tools from all MCP servers and custom tools"""
    # Get MCP tools
    mcp_tools = []
    if mcp_manager:
        mcp_tools = await mcp_manager.list_all_tools()
    
    # Get custom tool definitions
    custom_tools = tool_executor.get_tool_definitions()
    
    return {
        "tools": mcp_tools,
        "custom_tools": custom_tools
    }


@app.get("/api/models")
async def list_available_models():
    """List all available models from the LLM server"""
    models = await llm_client.list_models()
    return {"models": models}


@app.post("/api/rag/query")
async def rag_query_endpoint(request: Request):
    """
    Direct RAG query endpoint for searching documents.
    
    This can be used for explicit document queries without LLM tool calling.
    """
    data = await request.json()
    query = data.get("query", "")
    document_ids = data.get("document_ids")
    top_k = data.get("top_k", 10)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    result = await tool_executor.rag_service.query(
        query=query,
        document_ids=document_ids,
        top_k=top_k
    )
    
    return result


@app.post("/api/search/web")
async def web_search_endpoint(request: Request):
    """
    Direct web search endpoint using SearXNG.
    
    This can be used for explicit web searches without LLM tool calling.
    """
    data = await request.json()
    query = data.get("query", "")
    max_results = data.get("max_results", 15)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    result = await tool_executor.search_tool.search(
        query=query,
        max_results=max_results,
        top_k=max_results
    )
    
    return result


# TTS Endpoints
@app.post("/api/tts/generate")
async def generate_tts(request: Request):
    """
    Generate speech audio from text using TTS.
    
    Returns audio file URL that can be played in the browser.
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        voice = data.get("voice")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        result = await tool_executor.tts_service.generate_speech(
            text=text,
            voice=voice
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "TTS generation failed"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.get("/api/tts/voices")
async def list_tts_voices():
    """List available TTS voices"""
    return tool_executor.tts_service.list_available_voices()


@app.get("/api/tts/status")
async def get_tts_status():
    """Check if TTS is available"""
    from tools.tts_service import HAS_EDGE_TTS, HAS_PYTTSX3, _check_kokoro_available
    kokoro_available = _check_kokoro_available()
    return {
        "available": HAS_EDGE_TTS or HAS_PYTTSX3 or kokoro_available,
        "edge_tts": HAS_EDGE_TTS,
        "pyttsx3": HAS_PYTTSX3,
        "kokoro": kokoro_available,
        "engine": tool_executor.tts_service.config.engine
    }


# Settings Management
@app.get("/api/settings")
async def get_settings():
    """Get current application settings"""
    return settings_manager.get_settings()


@app.put("/api/settings")
async def update_settings(request: Request):
    """Update application settings"""
    data = await request.json()
    updated_settings = settings_manager.update_settings(data)
    return updated_settings


@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated TTS audio files"""
    audio_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, debug=DEBUG)

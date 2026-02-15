from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
import json
import os
import uuid
from typing import AsyncGenerator, Dict, List, Optional
from datetime import datetime

from config import APP_HOST, APP_PORT, DEBUG, MAX_UPLOAD_SIZE, UPLOAD_DIR #, SYSTEM_PROMPT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
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
        
        # Get conversation history
        messages = await get_conversation_messages(db, conversation_id)
        
        # Create a unique request ID for this interaction
        request_id = str(uuid.uuid4())
        
        # Store tool options in a way that stream_response can access
        # We'll use query parameters for the stream endpoint
        return {
            "request_id": request_id,
            "status": "processing",
            "enable_web_search": enable_web_search,
            "enable_rag": enable_rag
        }


@app.get("/api/stream/{request_id}")
async def stream_response(
    request_id: str,
    conversation_id: str,
    enable_web_search: bool = False,
    enable_rag: bool = False,
    model: str = None
):
    """Stream LLM response with real-time tool execution updates"""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async with get_db() as db:
                # Get conversation history
                messages = await get_conversation_messages(db, conversation_id)
                
                # Format messages for LLM
                llm_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ]
                
                # Pre-process with tools if enabled
                context_additions = []
                
                # Track tool calls with their results for storage
                tool_calls_data = []
                
                # Web search if enabled
                if enable_web_search:
                    query_text = messages[-1]["content"] if messages else ""
                    
                    # Initialize tool call data with progress history
                    web_search_tool_call = {
                        "name": "search_web",
                        "arguments": {"query": query_text},
                        "status": "starting",
                        "progress": 0,
                        "progress_history": [],  # Track all progress events
                        "result": None
                    }
                    tool_calls_data.append(web_search_tool_call)
                    
                    yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': 'search_web', 'args': {'query': query_text}})}\n\n"
                    await asyncio.sleep(0)
                    
                    # Send initial progress
                    web_search_tool_call["status"] = "Starting web search..."
                    web_search_tool_call["progress"] = 5
                    web_search_tool_call["progress_history"].append({"status": "Starting web search...", "progress": 5, "timestamp": datetime.utcnow().isoformat()})
                    yield f"data: {json.dumps({'type': 'tool_progress', 'tool': 'search_web', 'status': 'Starting web search...', 'progress': 5})}\n\n"
                    await asyncio.sleep(0)
                    
                    try:
                        # Create a queue for progress updates
                        progress_queue = asyncio.Queue()
                        search_complete = asyncio.Event()
                        search_error = [None]  # Use list to share state
                        search_result = [None]
                        
                        async def run_search():
                            """Run search in background and send progress updates"""
                            try:
                                async def progress_callback(status: str, progress: int):
                                    # Put progress in queue for main coroutine to send
                                    try:
                                        await progress_queue.put((status, progress))
                                    except:
                                        pass
                                
                                result = await tool_executor.search_tool.search(
                                    query=query_text,
                                    max_results=10,
                                    progress_callback=progress_callback
                                )
                                search_result[0] = result
                            except Exception as e:
                                search_error[0] = e
                            finally:
                                search_complete.set()
                        
                        # Start search task
                        search_task = asyncio.create_task(run_search())
                        
                        # Send progress updates while search is running
                        while not search_complete.is_set():
                            try:
                                # Check for progress updates with timeout
                                status, progress = await asyncio.wait_for(
                                    progress_queue.get(), 
                                    timeout=0.5
                                )
                                web_search_tool_call["status"] = status
                                web_search_tool_call["progress"] = progress
                                web_search_tool_call["progress_history"].append({"status": status, "progress": progress, "timestamp": datetime.utcnow().isoformat()})
                                yield f"data: {json.dumps({'type': 'tool_progress', 'tool': 'search_web', 'status': status, 'progress': progress})}\n\n"
                                await asyncio.sleep(0)
                            except asyncio.TimeoutError:
                                # Send keepalive comment to prevent timeout
                                yield ": keepalive\n\n"
                                await asyncio.sleep(0)
                        
                        # Wait for search to complete
                        await search_task
                        
                        if search_error[0]:
                            raise search_error[0]
                        
                        result = search_result[0]
                        if result and result.get("content"):
                            context_additions.append(f"\n\n**Web Search Results:**\n{result['content']}")
                            web_search_tool_call["status"] = "completed"
                            web_search_tool_call["progress"] = 100
                            web_search_tool_call["result"] = {"sources": result.get("sources", [])}
                            web_search_tool_call["progress_history"].append({"status": "Web search complete", "progress": 100, "timestamp": datetime.utcnow().isoformat()})
                            yield f"data: {json.dumps({'type': 'tool_progress', 'tool': 'search_web', 'status': 'Web search complete', 'progress': 100, 'result': {'sources': result.get('sources', [])}})}\n\n"
                            await asyncio.sleep(0)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        web_search_tool_call["status"] = "error"
                        web_search_tool_call["result"] = {"error": str(e)}
                        yield f"data: {json.dumps({'type': 'tool_error', 'tool': 'search_web', 'error': str(e)})}\n\n"
                        await asyncio.sleep(0)
                
                # RAG query if enabled
                if enable_rag:
                    query_text = messages[-1]["content"] if messages else ""
                    
                    # Initialize tool call data with progress history
                    rag_tool_call = {
                        "name": "query_documents",
                        "arguments": {"query": query_text},
                        "status": "starting",
                        "progress": 0,
                        "progress_history": [],  # Track all progress events
                        "result": None
                    }
                    tool_calls_data.append(rag_tool_call)
                    
                    yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': 'query_documents', 'args': {'query': query_text}})}\n\n"
                    await asyncio.sleep(0)
                    
                    rag_tool_call["status"] = "Searching documents..."
                    rag_tool_call["progress"] = 10
                    rag_tool_call["progress_history"].append({"status": "Searching documents...", "progress": 10, "timestamp": datetime.utcnow().isoformat()})
                    yield f"data: {json.dumps({'type': 'tool_progress', 'tool': 'query_documents', 'status': 'Searching documents...', 'progress': 10})}\n\n"
                    await asyncio.sleep(0)
                    
                    try:
                        rag_result = await tool_executor.rag_service.query(
                            query=query_text,
                            top_k=10
                        )
                        if rag_result.get("context"):
                            context_additions.append(f"\n\n**Document Search Results:**\n{rag_result['context']}")
                            rag_tool_call["status"] = "completed"
                            rag_tool_call["progress"] = 100
                            rag_tool_call["result"] = {"result_count": len(rag_result.get("results", []))}
                            rag_tool_call["progress_history"].append({"status": "Document search complete", "progress": 100, "timestamp": datetime.utcnow().isoformat()})
                            yield f"data: {json.dumps({'type': 'tool_progress', 'tool': 'query_documents', 'status': 'Document search complete', 'progress': 100, 'result': {'result_count': len(rag_result.get('results', []))}})}\n\n"
                            await asyncio.sleep(0)
                    except Exception as e:
                        rag_tool_call["status"] = "error"
                        rag_tool_call["result"] = {"error": str(e)}
                        yield f"data: {json.dumps({'type': 'tool_error', 'tool': 'query_documents', 'error': str(e)})}\n\n"
                        await asyncio.sleep(0)
                
                # Add context to the last user message if we have additions
                if context_additions and llm_messages:
                    # Find the last user message and append context
                    for i in range(len(llm_messages) - 1, -1, -1):
                        if llm_messages[i]["role"] == "user":
                            # Add citation instruction to help LLM cite sources
                            citation_instruction = "\n\n**Important**: When using information from the search results above, please cite your sources using the citation numbers [1], [2], etc. that correspond to the sources. For example: 'According to [1], the information states...'"
                            llm_messages[i]["content"] += "".join(context_additions) + citation_instruction
                            break
                
                assistant_message = ""
                thinking_content = ""  # Track thinking content separately
                tool_calls = tool_calls_data  # Start with the pre-tool calls from web search and RAG
                
                # Stream LLM response
                try:
                    async for chunk in llm_client.stream_chat(llm_messages, model=model):
                        chunk_type = chunk.get("type")
                        
                        if chunk_type == "content":
                            # Text content from LLM
                            content = chunk.get("content", "")
                            assistant_message += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                            # Small yield to allow streaming to work properly
                            await asyncio.sleep(0)
                        
                        elif chunk_type == "thinking":
                            # Thinking content from reasoning models (e.g., DeepSeek)
                            thinking = chunk.get("content", "")
                            thinking_content += thinking
                            yield f"data: {json.dumps({'type': 'thinking', 'content': thinking})}\n\n"
                            await asyncio.sleep(0)
                        
                        elif chunk_type == "tool_call":
                            # LLM wants to call a tool
                            tool_call = chunk.get("tool_call")
                            tool_calls.append({
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                                "status": "pending",
                                "result": None
                            })
                            
                            yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': tool_call['name'], 'args': tool_call['arguments']})}\n\n"
                            await asyncio.sleep(0)
                            
                            # Execute tool with progress updates
                            async for progress in tool_executor.execute_tool(
                                tool_call['name'],
                                tool_call['arguments'],
                                request_id
                            ):
                                # Update tool call status in our tracking
                                if progress.get("type") == "tool_progress":
                                    for tc in tool_calls:
                                        if tc["name"] == progress.get("tool") and tc["status"] != "completed":
                                            tc["status"] = progress.get("status", "running")
                                            tc["progress"] = progress.get("progress", 0)
                                            if progress.get("result"):
                                                tc["result"] = progress["result"]
                                                tc["status"] = "completed"
                                            break
                                yield f"data: {json.dumps(progress)}\n\n"
                                await asyncio.sleep(0)
                
                except asyncio.CancelledError:
                    # Client disconnected, clean up gracefully
                    print(f"Stream cancelled for request {request_id}")
                    raise
                except Exception as e:
                    # Handle other errors
                    error_msg = f"Error during streaming: {str(e)}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                # Determine if this is the first exchange before saving the assistant message
                messages_before_save = await get_conversation_messages(db, conversation_id)
                user_count_before = len([m for m in messages_before_save if m["role"] == "user"])
                assistant_count_before = len([m for m in messages_before_save if m["role"] == "assistant"])

                is_first_exchange = (user_count_before == 1 and assistant_count_before == 0)

                # Save assistant message (only if we have content)
                if assistant_message.strip():
                    await add_message(db, conversation_id, "assistant", assistant_message, tool_calls if tool_calls else None, thinking_content if thinking_content else None)

                # Auto-generate conversation title if this is the first exchange
                # This check happens regardless of whether the assistant message had content
                if is_first_exchange:
                    # Find the first user message to generate title from
                    first_user_message = next((m for m in messages_before_save if m["role"] == "user"), None)
                    if first_user_message:
                        title = None
                        try:
                            # Shield title generation from cancellation and add timeout
                            title = await asyncio.wait_for(
                                asyncio.shield(llm_client.generate_title(first_user_message["content"], model=model)),
                                timeout=30.0
                            )
                            # If title is empty or "New Chat", use fallback
                            if not title or title == "New Chat":
                                title = first_user_message["content"][:50].strip()
                        except asyncio.TimeoutError:
                            print(f"Title generation timed out for request {request_id}")
                            title = first_user_message["content"][:50].strip()
                        except asyncio.CancelledError:
                            # Client disconnected during title generation - try to save anyway
                            print(f"Title generation cancelled for request {request_id}, attempting fallback")
                            title = first_user_message["content"][:50].strip()
                        except Exception as e:
                            print(f"Error generating title: {e}")
                            title = first_user_message["content"][:50].strip()
                        
                        # Update title if we have one
                        if title:
                            try:
                                await update_conversation_title(db, conversation_id, title)
                                yield f"data: {json.dumps({'type': 'title_update', 'title': title})}\n\n"
                            except Exception as e:
                                print(f"Error updating title: {e}")

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except asyncio.CancelledError:
            # Client disconnected, this is normal
            print(f"Event generator cancelled for request {request_id}")
        except Exception as e:
            print(f"Error in event generator: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Clean up connection
            if request_id in active_connections:
                del active_connections[request_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


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
    
    success = await mcp_manager.add_server(name, command, args, env)
    
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
                
                # Format messages for LLM
                llm_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages_to_keep
                ]
                
                # Create new request ID
                request_id = str(uuid.uuid4())
                
                return {"request_id": request_id, "status": "processing", "conversation_id": conversation_id}
        
        raise HTTPException(status_code=400, detail="Could not find preceding user message")


@app.get("/api/stream/regenerate/{request_id}")
async def stream_regenerate_response(request_id: str, conversation_id: str, model: str = None):
    """Stream regenerated LLM response"""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async with get_db() as db:
                # Get conversation history
                messages = await get_conversation_messages(db, conversation_id)
                
                # Format messages for LLM
                llm_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ]
                
                assistant_message = ""
                tool_calls = []
                
                # Stream LLM response
                try:
                    async for chunk in llm_client.stream_chat(llm_messages, model=model):
                        chunk_type = chunk.get("type")
                        
                        if chunk_type == "content":
                            content = chunk.get("content", "")
                            assistant_message += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                            await asyncio.sleep(0)  # Allow streaming to work properly
                        
                        elif chunk_type == "tool_call":
                            tool_call = chunk.get("tool_call")
                            tool_calls.append(tool_call)
                            
                            yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': tool_call['name'], 'args': tool_call['arguments']})}\n\n"
                            await asyncio.sleep(0)
                            
                            async for progress in tool_executor.execute_tool(
                                tool_call['name'],
                                tool_call['arguments'],
                                request_id
                            ):
                                yield f"data: {json.dumps(progress)}\n\n"
                                await asyncio.sleep(0)
                
                except asyncio.CancelledError:
                    print(f"Stream cancelled for request {request_id}")
                    raise
                except Exception as e:
                    error_msg = f"Error during streaming: {str(e)}"
                    print(error_msg)
                    yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                
                # Save assistant message
                if assistant_message.strip():
                    await add_message(db, conversation_id, "assistant", assistant_message, tool_calls if tool_calls else None)
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in event generator: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if request_id in active_connections:
                del active_connections[request_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# Document upload endpoints
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
    import aiohttp
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{llm_client.base_url}/v1/models",
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    return {
                        "models": [
                            {
                                "id": m.get("id", ""),
                                "name": m.get("id", m.get("id", "Unknown")),
                                "owned_by": m.get("owned_by", "unknown")
                            }
                            for m in models
                        ]
                    }
                else:
                    return {"models": [], "error": f"Failed to fetch models: {response.status}"}
    except Exception as e:
        return {"models": [], "error": str(e)}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, debug=DEBUG)

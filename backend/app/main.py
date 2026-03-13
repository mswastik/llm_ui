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

from settings import APP_HOST, APP_PORT, DEBUG, MAX_UPLOAD_SIZE, UPLOAD_DIR
from database.models import init_db, get_db
from database.crud import (
    create_conversation, get_conversation, get_all_conversations,
    add_message, get_conversation_messages, update_conversation_title,
    delete_conversation as db_delete_conversation,
    update_message, get_message, create_document, update_document_status, get_documents,
    delete_message as db_delete_message, delete_document as db_delete_document, get_document
)
from database.agent_crud import (
    get_all_agents, get_agent, get_agent_by_name, create_agent,
    update_agent, delete_agent, get_default_agent
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


@app.get("/settings")
async def settings_page(request: Request):
    """Render settings page"""
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/knowledge")
async def knowledge_page(request: Request):
    """Render knowledge base page"""
    return templates.TemplateResponse("knowledge.html", {"request": request})


@app.get("/agents")
async def agents_page(request: Request):
    """Render agents management page"""
    return templates.TemplateResponse("agents.html", {"request": request})


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
            # Get conversation to retrieve agent configuration
            from sqlalchemy import select
            from database.models import Conversation
            result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
            conversation = result.scalar_one_or_none()

            # Get agent configuration if conversation has an agent
            agent_config = None
            if conversation:
                agent_id = conversation.agent_id
                if agent_id is not None:
                    agent = await get_agent(db, agent_id)
                    if agent:
                        agent_config = {
                            "system_prompt": agent.system_prompt,
                            "model": agent.model,
                            "temperature": agent.temperature,
                            "top_k": agent.top_k,
                            "max_tokens": agent.max_tokens,
                            "enable_web_search": bool(agent.enable_web_search),
                            "enable_rag": bool(agent.enable_rag)
                        }
            
            # Get current date for system prompt
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Build system prompt with current date
            system_prompt_content = agent_config["system_prompt"] if agent_config and agent_config.get("system_prompt") else ""
            if system_prompt_content:
                # Add current date to system prompt
                system_prompt_content = f"{system_prompt_content}\n\nCurrent date: {current_date}"
            else:
                # Default minimal system prompt with current date
                system_prompt_content = f"You are a helpful AI assistant. Current date: {current_date}"
            
            messages = await get_conversation_messages(db, conversation_id)
            llm_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            # Prepend system prompt to messages
            if system_prompt_content:
                llm_messages.insert(0, {"role": "system", "content": system_prompt_content})

            tool_calls_history = []

            # Track message blocks for sequential display (content, thinking, tool calls)
            message_blocks = []
            current_content_block = ""
            current_thinking_block = ""

            # Get MCP tools for LLM function calling
            mcp_tools = []
            if mcp_manager:
                mcp_tools = await mcp_manager.list_all_tools()

            # Get all tool definitions - include web_search and query_documents if enabled
            # The LLM will decide which tools to call based on the user's request
            all_tools = tool_executor.get_tool_definitions(
                exclude_tools=[],  # Don't exclude any tools - let LLM choose
                mcp_tools=mcp_tools,
                enable_web_search=enable_web_search,
                enable_rag=enable_rag
            )

            # Debug logging - show exactly what tools are being sent
            print(f"[TOOLS] MCP tools discovered: {len(mcp_tools)}")
            print(f"[TOOLS] Total tools sent to LLM: {len(all_tools)}")
            print(f"[TOOLS] enable_web_search={enable_web_search}, enable_rag={enable_rag}")
            for tool in all_tools:
                tool_name = tool.get("function", {}).get("name", "unknown")
                tool_server = tool.get("server", "builtin")
                print(f"[TOOLS]   - {tool_name} ({'MCP: ' + tool_server if tool_server != 'builtin' else 'builtin'})")
            if mcp_tools:
                for tool in mcp_tools:
                    print(f"  - MCP: {tool['name']} from {tool['server']}")
            
            # Main conversation loop - handles multiple tool calls with content in between
            max_tool_iterations = 15  # Prevent infinite loops
            tool_iteration = 0

            while tool_iteration < max_tool_iterations:
                tool_iteration += 1
                print(f"[DEBUG] Conversation loop iteration {tool_iteration}")

                # Stream LLM response
                assistant_message, thinking_content = "", ""
                pending_tool_call = None
                had_content = False

                async for chunk in llm_client.stream_chat(llm_messages, model=model, tools=all_tools):
                    try:
                        chunk_type = chunk.get("type")
                        #print(f"[DEBUG] Chunk received: type={chunk_type}")

                        if chunk_type == "content":
                            content = chunk.get("content", "")
                            assistant_message += content
                            # Save content immediately as a block for proper sequencing
                            # Preserve all content including whitespace/newlines for proper formatting
                            if content:  # Save even if it's just whitespace/newlines
                                message_blocks.append({
                                    "type": "content",
                                    "content": content
                                })
                            had_content = True
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                        elif chunk_type == "thinking":
                            thinking = chunk.get("content", "")
                            thinking_content += thinking
                            # Save thinking immediately as a block for proper sequencing
                            # Preserve all thinking content including whitespace/newlines
                            if thinking:  # Save even if it's just whitespace/newlines
                                # Check if last block is also thinking - if so, append to it
                                if message_blocks and message_blocks[-1].get('type') == 'thinking':
                                    message_blocks[-1]['content'] += thinking
                                else:
                                    message_blocks.append({
                                        "type": "thinking",
                                        "content": thinking
                                    })
                            yield f"data: {json.dumps({'type': 'thinking', 'content': thinking})}\n\n"
                        elif chunk_type == "tool_call":
                            print(f"[DEBUG] Tool call chunk received")
                            tc_data = chunk.get("tool_call")
                            print(f"[DEBUG] tc_data: {tc_data}")

                            if tc_data:
                                tool_name = tc_data.get('name')
                                tool_args = tc_data.get('arguments')

                                # Store the pending tool call
                                if tool_name:
                                    pending_tool_call = {
                                        "name": tool_name,
                                        "arguments": tool_args if isinstance(tool_args, dict) else {},
                                        "status": "pending",
                                        "result": None,
                                        "progress_history": []
                                    }
                                    print(f"[DEBUG] Pending tool call: {tool_name}, args: {pending_tool_call['arguments']}")

                    except Exception as e:
                        print(f"[DEBUG] Error processing chunk: {e}")
                        print(f"[DEBUG] Chunk data: {chunk}")
                        import traceback
                        traceback.print_exc()
                    await asyncio.sleep(0)

                # If we have a pending tool call, execute it and continue the loop
                if pending_tool_call:
                    print(f"Executing tool: {pending_tool_call['name']}")

                    # Send tool call start event
                    yield f"data: {json.dumps({'type': 'tool_call_start', 'tool': pending_tool_call['name'], 'args': pending_tool_call['arguments']})}\n\n"
                    tool_calls_history.append(pending_tool_call)

                    # Execute the tool
                    tool_result = None
                    async for progress_event in tool_executor.execute_tool(
                        pending_tool_call['name'],
                        pending_tool_call['arguments'],
                        request_id
                    ):
                        # Forward the progress event
                        yield f"data: {json.dumps(progress_event)}\n\n"
                        if progress_event.get("type") == "tool_progress":
                            pending_tool_call["status"] = progress_event.get("status", "running")
                            pending_tool_call["progress"] = progress_event.get("progress", 0)
                            pending_tool_call["progress_history"].append(progress_event)
                            if progress_event.get("result"):
                                pending_tool_call["result"] = progress_event["result"]
                                pending_tool_call["status"] = "completed"
                                tool_result = progress_event["result"]
                        elif progress_event.get("type") == "tool_error":
                            pending_tool_call["status"] = "error"
                            pending_tool_call["result"] = {"error": progress_event.get("error")}

                    # Add tool call block to message blocks for sequential display
                    message_blocks.append({
                        "type": "tool_call",
                        "name": pending_tool_call['name'],
                        "arguments": pending_tool_call['arguments'],
                        "status": pending_tool_call['status'],
                        "result": pending_tool_call['result'],
                        "progress_history": pending_tool_call['progress_history']
                    })

                    # Add tool result to conversation for LLM to continue
                    # Format for llama.cpp: role=tool with content as string
                    tool_result_str = json.dumps(tool_result, default=str) if tool_result else "No result"
                    llm_messages.append({
                        "role": "tool",
                        "content": tool_result_str,
                        "tool_call_id": pending_tool_call['name']
                    })

                    print(f"[DEBUG] Tool executed, continuing conversation with result")
                    # Continue the while loop to get LLM's response to the tool result
                    # (LLM may respond with content, thinking, or another tool call)

                else:
                    # No tool call, conversation is complete
                    print(f"[DEBUG] No pending tool call, conversation complete")
                    break
            
            # Save any remaining content/thinking blocks after the loop ends
            # Note: These variables are not used in the current streaming logic
            # but kept for backward compatibility
            if current_content_block: #.strip():
                message_blocks.append({
                    "type": "content",
                    "content": current_content_block  # Preserve original formatting
                })
            if current_thinking_block: #.strip():
                message_blocks.append({
                    "type": "thinking",
                    "content": current_thinking_block  # Preserve original formatting
                })

            # Save assistant message with message blocks for sequential display
            # Always save if we have any content, thinking, or message blocks
            if assistant_message.strip() or thinking_content.strip() or message_blocks:
                # Consolidate consecutive content blocks to avoid fragmentation
                # But preserve newlines and formatting within each block
                consolidated_blocks = []
                for block in message_blocks:
                    if block.get('type') == 'content':
                        # Check if last block is also content - if so, merge
                        if consolidated_blocks and consolidated_blocks[-1].get('type') == 'content':
                            # Preserve newlines - concatenate exactly as received
                            prev_content = consolidated_blocks[-1].get('content', '')
                            new_content = block.get('content', '')
                            # Don't strip or modify - preserve exact formatting
                            consolidated_blocks[-1]['content'] = prev_content + new_content
                        else:
                            consolidated_blocks.append(block)
                    elif block.get('type') == 'thinking':
                        # Check if last block is also thinking - if so, merge
                        if consolidated_blocks and consolidated_blocks[-1].get('type') == 'thinking':
                            prev_content = consolidated_blocks[-1].get('content', '')
                            new_content = block.get('content', '')
                            consolidated_blocks[-1]['content'] = prev_content + new_content
                        else:
                            consolidated_blocks.append(block)
                    else:
                        # Tool calls are kept as-is
                        consolidated_blocks.append(block)
                
                # Add model info to message metadata
                message_extra_metadata = {"model": model} if model else {}
                # Store consolidated_blocks in metadata['blocks'] for sequential rendering
                print(f"[DEBUG] Saving message with {len(consolidated_blocks)} consolidated blocks")
                for i, block in enumerate(consolidated_blocks):
                    content_preview = block.get('content', '')[:100].replace('\n', '\\n') if block.get('content') else ''
                    print(f"[DEBUG] Block {i}: type={block.get('type')}, content_preview='{content_preview}...'")
                await add_message(db, conversation_id, "assistant", assistant_message, blocks=consolidated_blocks or None, extra_metadata=message_extra_metadata)

            # Title Generation Logic (only for first exchange)
            messages_after_save = await get_conversation_messages(db, conversation_id)
            user_count = len([m for m in messages_after_save if m["role"] == "user"])
            assistant_count = len([m for m in messages_after_save if m["role"] == "assistant"])

            if user_count == 1 and assistant_count == 1:
                first_user_message = next((m for m in messages_after_save if m["role"] == "user"), None)
                if first_user_message:
                    try:
                        # Use QUERY_MODEL for title generation to avoid issues with thinking models
                        from settings import QUERY_MODEL
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
    except asyncio.CancelledError:
        # Request was cancelled by client - this is normal, don't log as error
        print(f"Request {request_id} cancelled by client")
    finally:
        if request_id in active_connections:
            try:
                del active_connections[request_id]
            except KeyError:
                pass


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
    from database.crud import get_all_mcp_servers
    from database.models import get_db

    # Get runtime server status from MCP manager
    runtime_servers = await mcp_manager.list_servers()

    # Get all servers from database (both enabled and disabled)
    async with get_db() as db:
        db_servers = await get_all_mcp_servers(db)
        db_enabled = {s["name"]: s["enabled"] for s in db_servers}

    # Merge runtime info with enabled status
    servers_with_status = []
    for server in runtime_servers:
        servers_with_status.append({
            **server,
            "enabled": db_enabled.get(server["name"], True)
        })

    # Also include servers from DB that might not be connected
    for db_server in db_servers:
        if not any(s["name"] == db_server["name"] for s in runtime_servers):
            servers_with_status.append({
                "name": db_server["name"],
                "transport_type": db_server.get("transport_type", "stdio"),
                "command": db_server.get("command"),
                "url": db_server.get("url"),
                "tool_count": 0,
                "is_connected": False,
                "is_initialized": False,
                "error": None,
                "enabled": db_server.get("enabled", True)
            })

    return {"servers": servers_with_status}


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


@app.put("/api/mcp/servers/{server_name}")
async def update_mcp_server(server_name: str, request: Request):
    """
    Update an existing MCP server configuration.
    
    This updates the server config and reconnects with new settings.
    """
    data = await request.json()
    
    # Get new configuration
    new_name = data.get("name", server_name)
    command = data.get("command")
    args = data.get("args", [])
    env = data.get("env", {})
    transport_type = data.get("transport_type", "stdio")
    url = data.get("url")
    timeout = data.get("timeout", 30.0)
    
    # Validate based on transport type
    if transport_type in ("sse", "http"):
        if not url:
            raise HTTPException(status_code=400, detail="URL is required for SSE/HTTP transport")
    elif transport_type == "stdio":
        if not command:
            raise HTTPException(status_code=400, detail="Command is required for stdio transport")
    
    # First remove the old server
    await mcp_manager.remove_server(server_name)
    
    # Add with new configuration (using new name if provided)
    success = await mcp_manager.add_server(
        name=new_name,
        command=command,
        args=args,
        env=env,
        transport_type=transport_type,
        url=url,
        timeout=timeout
    )
    
    if success:
        return {"status": "success", "message": f"Server '{new_name}' updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to update server")


@app.post("/api/mcp/servers/{server_name}/refresh")
async def refresh_mcp_server_tools(server_name: str):
    """
    Refresh the tool list for a specific MCP server.
    
    Useful when server tools have changed without restarting.
    """
    if server_name not in [s["name"] for s in await mcp_manager.list_servers()]:
        raise HTTPException(status_code=404, detail="Server not found")
    
    success = await mcp_manager.refresh_tools(server_name)
    
    if success:
        return {"status": "success", "message": f"Tools refreshed for server '{server_name}'"}
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh tools")


@app.post("/api/mcp/servers/{server_name}/reconnect")
async def reconnect_mcp_server(server_name: str):
    """
    Reconnect to an MCP server.

    Useful when connection was lost or server was restarted.
    """
    if server_name not in [s["name"] for s in await mcp_manager.list_servers()]:
        raise HTTPException(status_code=404, detail="Server not found")

    success = await mcp_manager.reconnect_server(server_name)

    if success:
        return {"status": "success", "message": f"Reconnected to server '{server_name}'"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reconnect to server")


@app.post("/api/mcp/servers/{server_name}/toggle")
async def toggle_mcp_server_endpoint(server_name: str, request: Request):
    """
    Enable or disable an MCP server.
    """
    from database.crud import toggle_mcp_server as db_toggle_mcp_server
    
    data = await request.json()
    enabled = data.get("enabled", True)

    async with get_db() as db:
        await db_toggle_mcp_server(db, server_name, enabled)

    return {"status": "success", "message": f"Server '{server_name}' {'enabled' if enabled else 'disabled'}"}


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
    """Regenerate the last assistant response by re-sending the previous user message"""
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
                # Delete all messages from this point (the user message and all following messages)
                # This ensures we regenerate from the correct point
                messages_to_delete = messages[msg_index:]
                for msg_to_delete in messages_to_delete:
                    await db_delete_message(db, msg_to_delete.get("id"))
                
                # Re-add the user message (it was deleted but we need it in the conversation)
                await add_message(db, conversation_id, "user", user_message["content"])
                
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


# Agent Management
@app.get("/agents")
async def agents_page(request: Request):
    """Render agents management page"""
    return templates.TemplateResponse("agents.html", {"request": request})


@app.get("/api/agents")
async def list_agents():
    """List all agents"""
    async with get_db() as db:
        agents = await get_all_agents(db)
        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description,
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "top_k": agent.top_k,
                    "max_tokens": agent.max_tokens,
                    "system_prompt": agent.system_prompt,
                    "enabled_tools": agent.enabled_tools,
                    "enabled_mcp_servers": agent.enabled_mcp_servers,
                    "enable_rag": bool(agent.enable_rag),
                    "rag_similarity_threshold": agent.rag_similarity_threshold,
                    "enable_web_search": bool(agent.enable_web_search),
                    "conversation_starters": agent.conversation_starters,
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat(),
                    "is_active": bool(agent.is_active)
                }
                for agent in agents
            ]
        }


@app.get("/api/agents/{agent_id}")
async def get_agent_detail(agent_id: int):
    """Get agent details"""
    async with get_db() as db:
        agent = await get_agent(db, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "temperature": agent.temperature,
                "top_k": agent.top_k,
                "max_tokens": agent.max_tokens,
                "system_prompt": agent.system_prompt,
                "enabled_tools": agent.enabled_tools,
                "enabled_mcp_servers": agent.enabled_mcp_servers,
                "enable_rag": bool(agent.enable_rag),
                "rag_similarity_threshold": agent.rag_similarity_threshold,
                "enable_web_search": bool(agent.enable_web_search),
                "conversation_starters": agent.conversation_starters,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "is_active": bool(agent.is_active)
            }
        }


@app.post("/api/agents")
async def create_agent_endpoint(request: Request):
    """Create a new agent"""
    data = await request.json()
    
    agent_data = {
        "name": data.get("name"),
        "description": data.get("description", ""),
        "model": data.get("model", "qwen3-4b"),
        "temperature": data.get("temperature", 0.7),
        "top_k": data.get("top_k", 40),
        "max_tokens": data.get("max_tokens", 16048),
        "system_prompt": data.get("system_prompt", ""),
        "enabled_tools": data.get("enabled_tools", []),
        "enabled_mcp_servers": data.get("enabled_mcp_servers", []),
        "enable_rag": 1 if data.get("enable_rag", False) else 0,
        "rag_similarity_threshold": data.get("rag_similarity_threshold", 0.4),
        "enable_web_search": 1 if data.get("enable_web_search", False) else 0,
        "conversation_starters": data.get("conversation_starters", [])
    }
    
    # Validate required fields
    if not agent_data["name"]:
        raise HTTPException(status_code=400, detail="Agent name is required")
    
    async with get_db() as db:
        # Check if name already exists
        existing = await get_agent_by_name(db, agent_data["name"])
        if existing:
            raise HTTPException(status_code=400, detail="Agent with this name already exists")
        
        agent = await create_agent(db, agent_data)
        return {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "temperature": agent.temperature,
                "top_k": agent.top_k,
                "max_tokens": agent.max_tokens,
                "system_prompt": agent.system_prompt,
                "enabled_tools": agent.enabled_tools,
                "enabled_mcp_servers": agent.enabled_mcp_servers,
                "enable_rag": bool(agent.enable_rag),
                "rag_similarity_threshold": agent.rag_similarity_threshold,
                "enable_web_search": bool(agent.enable_web_search),
                "conversation_starters": agent.conversation_starters,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "is_active": bool(agent.is_active)
            }
        }


@app.put("/api/agents/{agent_id}")
async def update_agent_endpoint(agent_id: int, request: Request):
    """Update an agent"""
    data = await request.json()
    
    update_data = {}
    if "name" in data:
        update_data["name"] = data["name"]
    if "description" in data:
        update_data["description"] = data["description"]
    if "model" in data:
        update_data["model"] = data["model"]
    if "temperature" in data:
        update_data["temperature"] = data["temperature"]
    if "top_k" in data:
        update_data["top_k"] = data["top_k"]
    if "max_tokens" in data:
        update_data["max_tokens"] = data["max_tokens"]
    if "system_prompt" in data:
        update_data["system_prompt"] = data["system_prompt"]
    if "enabled_tools" in data:
        update_data["enabled_tools"] = data["enabled_tools"]
    if "enabled_mcp_servers" in data:
        update_data["enabled_mcp_servers"] = data["enabled_mcp_servers"]
    if "enable_rag" in data:
        update_data["enable_rag"] = 1 if data["enable_rag"] else 0
    if "rag_similarity_threshold" in data:
        update_data["rag_similarity_threshold"] = data["rag_similarity_threshold"]
    if "enable_web_search" in data:
        update_data["enable_web_search"] = 1 if data["enable_web_search"] else 0
    if "conversation_starters" in data:
        update_data["conversation_starters"] = data["conversation_starters"]
    
    async with get_db() as db:
        agent = await update_agent(db, agent_id, update_data)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "temperature": agent.temperature,
                "top_k": agent.top_k,
                "max_tokens": agent.max_tokens,
                "system_prompt": agent.system_prompt,
                "enabled_tools": agent.enabled_tools,
                "enabled_mcp_servers": agent.enabled_mcp_servers,
                "enable_rag": bool(agent.enable_rag),
                "rag_similarity_threshold": agent.rag_similarity_threshold,
                "enable_web_search": bool(agent.enable_web_search),
                "conversation_starters": agent.conversation_starters,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "is_active": bool(agent.is_active)
            }
        }


@app.delete("/api/agents/{agent_id}")
async def delete_agent_endpoint(agent_id: int):
    """Delete an agent (soft delete)"""
    async with get_db() as db:
        success = await delete_agent(db, agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"status": "success", "message": "Agent deleted"}


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
    
    # Update LLM client with new settings if URL or model changed
    if 'llama_cpp_base_url' in data or 'llama_cpp_model' in data:
        settings = settings_manager.get_settings()
        llm_client.base_url = settings.get('llama_cpp_base_url', llm_client.base_url)
        llm_client.model = settings.get('llama_cpp_model', llm_client.model)
        print(f"Updated LLM client: base_url={llm_client.base_url}, model={llm_client.model}")
    
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

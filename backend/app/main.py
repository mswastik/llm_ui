from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import asyncio
import json
import os
import uuid
import contextvars
from typing import AsyncGenerator, Dict, List, Optional

from tools.base import current_document_ids

from settings import APP_HOST, APP_PORT, DEBUG, MAX_UPLOAD_SIZE, UPLOAD_DIR
from database.models import init_db, get_db
from database.crud import (
    create_conversation, get_conversation, get_all_conversations,
    add_message, get_conversation_messages, update_conversation_title,
    delete_conversation as db_delete_conversation,
    update_message, get_message, create_document, update_document_status, get_documents,
    delete_message as db_delete_message, delete_document as db_delete_document, get_document,
    get_all_agents, get_agent, get_agent_by_name, create_agent,
    update_agent, delete_agent,
    add_mcp_server, get_all_mcp_servers, get_enabled_mcp_servers,
    toggle_mcp_server, remove_mcp_server, update_mcp_server_disabled_tools,
    update_conversation_tags, update_conversation_agent,
    create_note, get_all_notes, delete_note, get_notes_for_conversation,
    get_message_versions
)
from mcp_client.client import MCPClientManager, MCPServerConfig
from tools.tool_executor import ToolExecutor
from llm_client.client import LLMClient
from backend.settings import settings_manager
from backend.database.backup import backup_scheduler

# Initialize MCP client manager
mcp_manager = MCPClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await mcp_manager.initialize()
    backup_scheduler.start()
    yield
    # Shutdown
    await backup_scheduler.stop()
    await mcp_manager.cleanup()

app = FastAPI(title="LLM UI with MCP Support", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
# Mount uploads directory so uploaded files are publicly accessible
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory="frontend/templates")

# Initialize components
llm_client = LLMClient()
tool_executor = ToolExecutor(mcp_manager)

# Set TTS service in settings manager
settings_manager.set_tts_service(tool_executor.tts_service)




@app.get("/")
async def index(request: Request):
    """Render main chat interface"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/settings")
async def settings_page(request: Request):
    """Redirect to main page (settings are now a modal)"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/knowledge")
async def knowledge_page(request: Request):
    """Redirect to main page (knowledge base is now a modal)"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/agents")
async def agents_page(request: Request):
    """Redirect to main page (agents are now a modal)"""
    return templates.TemplateResponse(request, "index.html")


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
    agent_id = data.get("agent_id")
    
    async with get_db() as db:
        conversation = await create_conversation(db, title, agent_id)
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
    enable_rag = data.get("enable_rag", False)
    document_ids = data.get("document_ids")
    files = data.get("files", [])  # [{url, filename, type, size}]

    if not user_message.strip() and not files:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build message content with file references included
    message_content = user_message
    
    async with get_db() as db:
        await add_message(db, conversation_id, "user", message_content, extra_metadata={"files": files} if files else None)
        request_id = str(uuid.uuid4())

        return {
            "request_id": request_id,
            "status": "processing",
            "enable_rag": enable_rag,
            "document_ids": document_ids,
            "files": files
        }


def _strip_thinking(text: str) -> str:
    """Remove thinking tags and HTML tags from text."""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text).strip()
    return text


async def _generate_title_with_model(
    llm_messages: list,
    assistant_message: str,
    llm_client,
    tools: list = None,
    model: str = None
) -> str:
    """
    Generate a title by appending to the existing conversation messages.
    KV cache reuses the prefix from the main response.
    The appended messages are not saved to DB — only the title is returned.
    The title generation request is built from a copy of llm_messages, so the
    original is untouched and future requests can still reuse the KV cache.
    """
    title_messages = list(llm_messages)
    title_messages.append({"role": "assistant", "content": assistant_message})
    title_messages.append({"role": "user", "content": "Generate a title (3-6 words) for this conversation. No reasoning. Just output the title."})

    title = ""
    thinking = ""
    try:
        async for chunk in llm_client.stream_chat(
            title_messages,
            model=model,
            temperature=0.0,
            max_tokens=1024,
            tools=tools,
            tool_choice="none"
        ):
            chunk_type = chunk.get("type")
            if chunk_type == "content":
                title += chunk.get("content", "")
            elif chunk_type == "thinking":
                thinking += chunk.get("content", "")
            else:
                print(f"[TITLE GEN] unexpected chunk type: {chunk_type}")
    except Exception as e:
        print(f"Title generation error: {e}")
        return ""

    print(f"[TITLE GEN] raw title: '{title[:200]}'")
    print(f"[TITLE GEN] thinking length: {len(thinking)} chars")

    # If content is empty but thinking has content, the title might be inside thinking
    if not title.strip() and thinking.strip():
        title = thinking

    title = _strip_thinking(title)
    words = title.split()
    if not words:
        return ""
    title = ' '.join(words[:6]).strip().rstrip('.,;:!?\\-"\'')
    return title[:60]




async def _core_stream_handler(
    request_id: str,
    conversation_id: str,
    enable_rag: bool = False,
    model: Optional[str] = None,
    document_ids: Optional[list] = None,
    version: int = 1,
    version_group: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Universal SSE handler for streaming LLM responses and tool execution.
    
    Args:
        version: Version number for regenerated responses
        version_group: UUID shared by all versions of the same response
    """
    current_document_ids.set(document_ids)
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
                            "enable_rag": bool(agent.enable_rag),
                            "enabled_tools": agent.enabled_tools or [],
                            "enabled_mcp_servers": agent.enabled_mcp_servers or []
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
            # Build LLM messages — convert user messages with file attachments to multimodal format
            llm_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                metadata = msg.get("metadata") or {}
                files = metadata.get("files", [])
                
                if role == "user" and files:
                    content_parts = []
                    if content.strip():
                        content_parts.append({"type": "text", "text": content})
                    for f in files:
                        f_type = f.get("type", "") or ""
                        f_url = f.get("url", "")
                        f_name = f.get("filename", "file")
                        if f_type.startswith("image/") and f_url:
                            # Convert image to base64 data URI for vision models
                            f_path = os.path.join(UPLOAD_DIR, os.path.basename(f_url))
                            if os.path.exists(f_path):
                                try:
                                    import base64
                                    with open(f_path, "rb") as img_f:
                                        b64_data = base64.b64encode(img_f.read()).decode()
                                    content_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{f_type};base64,{b64_data}"}
                                    })
                                except Exception as e:
                                    print(f"[MULTIMODAL] Error reading image {f_path}: {e}")
                                    content_parts.append({"type": "text", "text": f"\n[Image: {f_name}]"})
                            else:
                                content_parts.append({"type": "text", "text": f"\n[Image file not found: {f_name}]"})
                        else:
                            content_parts.append({"type": "text", "text": f"\n[Attached file: {f_name}]"})
                    llm_messages.append({"role": role, "content": content_parts})
                else:
                    llm_messages.append({"role": role, "content": content})
            
            # Prepend system prompt to messages
            if system_prompt_content:
                llm_messages.insert(0, {"role": "system", "content": system_prompt_content})

            tool_calls_history = []

            # Track message blocks for sequential display (content, thinking, tool calls)
            message_blocks = []

            # Get MCP tools for LLM function calling
            mcp_tools = []
            if mcp_manager:
                # Get MCP tools, excluding disabled ones to save tokens in the LLM prompt.
                mcp_tools = await mcp_manager.list_all_tools(include_disabled=False)
                if mcp_tools:
                    print(f"[TOOLS] {len(mcp_tools)} MCP tools available for LLM")

            # ── Filter tools by agent binding ──────────────────────────
            # Agent can restrict which MCP servers and custom tools are sent to the LLM.
            # Empty list = allow everything (backward compatible).
            # Non-empty list = only allow those specific servers/tools.
            enabled_mcp_servers = []
            if agent_config and agent_config.get("enabled_mcp_servers"):
                enabled_mcp_servers = agent_config["enabled_mcp_servers"]
            if enabled_mcp_servers:
                allowed_servers = set(enabled_mcp_servers)
                mcp_tools = [t for t in mcp_tools if t.get("server") in allowed_servers]
                print(f"[TOOLS] Filtered to {len(mcp_tools)} MCP tools from servers: {allowed_servers}")

            # ── Filter custom tools by agent's enabled_tools ──────────────
            # Custom tools are controlled differently:
            #   - query_documents (RAG) → controlled by enable_rag flag
            #   - generate_speech (TTS) → controlled by exclude_tools param
            # If enabled_tools is configured, exclude tools not in the list.
            exclude_tools = []
            effective_enable_rag = enable_rag
            if agent_config and agent_config.get("enabled_tools"):
                enabled_custom = set(agent_config["enabled_tools"])
                if "query_documents" not in enabled_custom:
                    effective_enable_rag = False
                if "generate_speech" not in enabled_custom:
                    exclude_tools.append("generate_speech")
            elif agent_config is not None:
                # No specific tools restriction — use agent's enable_rag setting
                if agent_config.get("enable_rag"):
                    effective_enable_rag = True

            # Get tool definitions with agent-based filtering
            all_tools = tool_executor.get_tool_definitions(
                exclude_tools=exclude_tools,
                mcp_tools=mcp_tools,
                enable_rag=effective_enable_rag
            )

            # Debug logging - show exactly what tools are being sent
            print(f"[TOOLS] MCP tools discovered: {len(mcp_tools)}")
            print(f"[TOOLS] Total tools sent to LLM: {len(all_tools)}")
            print(f"[TOOLS] enable_rag={effective_enable_rag}")
            for tool in all_tools:
                tool_name = tool.get("function", {}).get("name", "unknown")
                tool_server = tool.get("server", "builtin")
                print(f"[TOOLS]   - {tool_name} ({'MCP: ' + tool_server if tool_server != 'builtin' else 'builtin'})")
            if mcp_tools:
                for tool in mcp_tools:
                    print(f"  - MCP: {tool['name']} from {tool['server']}")
            
            # Main conversation loop - handles multiple tool calls with content in between
            max_tool_iterations = 35  # Prevent infinite loops
            tool_iteration = 0

            while tool_iteration < max_tool_iterations:
                tool_iteration += 1
                print(f"[DEBUG] Conversation loop iteration {tool_iteration}")

                # Stream LLM response
                assistant_message, thinking_content = "", ""
                pending_tool_calls = []

                # Watchdog: if the stream stalls (no chunks for 30s) or exceeds total 120s,
                # break out to save partial response and yield done instead of hanging forever.
                # This is critical for MTP models that can stop sending chunks mid-stream.
                try:
                    async with asyncio.timeout(120):
                        async for chunk in llm_client.stream_chat(llm_messages, model=model, tools=all_tools):
                            try:
                                chunk_type = chunk.get("type")

                                if chunk_type == "content":
                                    content = chunk.get("content", "")
                                    assistant_message += content
                                    if content:
                                        message_blocks.append({
                                            "type": "content",
                                            "content": content
                                        })
                                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                                elif chunk_type == "thinking":
                                    thinking = chunk.get("content", "")
                                    thinking_content += thinking
                                    if thinking:
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

                                        if tool_name:
                                            call_key = f"{tool_name}_{len(pending_tool_calls)}"
                                            pending_tool_calls.append({
                                                "name": tool_name,
                                                "arguments": tool_args if isinstance(tool_args, dict) else {},
                                                "status": "pending",
                                                "result": None,
                                                "progress_history": [],
                                                "key": call_key
                                            })
                                            print(f"[DEBUG] Pending tool call #{len(pending_tool_calls)-1}: {tool_name}, args: {pending_tool_calls[-1]['arguments']}")

                                elif chunk_type == "tool_error":
                                    # Premature EOS from MTP — tool call was started but never completed
                                    tool_name = chunk.get("tool", "unknown")
                                    error_msg = chunk.get("error", "Incomplete tool call")
                                    print(f"[DEBUG] Incomplete tool call from stream: {tool_name}: {error_msg}")
                                    # Add an error block so the user sees what happened
                                    message_blocks.append({
                                        "type": "tool_call",
                                        "name": tool_name,
                                        "arguments": {},
                                        "status": "error",
                                        "result": {"error": error_msg},
                                        "progress_history": []
                                    })
                                    yield f"data: {json.dumps({'type': 'tool_error', 'tool': tool_name, 'error': error_msg})}\n\n"

                            except Exception as e:
                                print(f"[DEBUG] Error processing chunk: {e}")
                                print(f"[DEBUG] Chunk data: {chunk}")
                                import traceback
                                traceback.print_exc()
                            await asyncio.sleep(0)
                except asyncio.TimeoutError:
                    print(f"[WATCHDOG] Stream timeout after 120s for request {request_id} — saving partial response")
                    # Fall through to save partial response and yield done

                # If we have pending tool calls, execute them and continue the loop
                if pending_tool_calls:
                    for i, pending_tool_call in enumerate(pending_tool_calls):
                        print(f"Executing tool {i+1}/{len(pending_tool_calls)}: {pending_tool_call['name']}")

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
                                # Check for result in various possible locations
                                # For MCP tools: result is in 'content' field (parsed CallToolResult)
                                # For custom tools: result might be in 'result' field
                                result = progress_event.get("result") or progress_event.get("content")
                                if result:
                                    pending_tool_call["result"] = result
                                    pending_tool_call["status"] = "completed"
                                    tool_result = result
                            elif progress_event.get("type") == "tool_error":
                                pending_tool_call["status"] = "error"
                                pending_tool_call["result"] = {"error": progress_event.get("error")}
                                tool_result = {"error": progress_event.get("error")}

                        # Add tool call block to message blocks for sequential display
                        tool_call_block = {
                            "type": "tool_call",
                            "name": pending_tool_call['name'],
                            "arguments": pending_tool_call['arguments'],
                            "status": pending_tool_call['status'],
                            "result": pending_tool_call['result'],
                            "progress_history": pending_tool_call['progress_history']
                        }
                        # Extract sources from result for bottom-of-chat display
                        if pending_tool_call['result'] and isinstance(pending_tool_call['result'], dict):
                            sources = pending_tool_call['result'].get('sources', [])
                            if sources:
                                tool_call_block['sources'] = sources
                        message_blocks.append(tool_call_block)

                        # Add tool result to conversation for LLM to continue
                        # Format for llama.cpp: role=tool with content as string
                        tool_result_str = json.dumps(tool_result, default=str) if tool_result else "No result"
                        print(f"[DEBUG] Tool result {i+1} preview: {tool_result_str[:500] if tool_result_str else 'None'}...")
                        llm_messages.append({
                            "role": "tool",
                            "content": tool_result_str,
                            "tool_call_id": f"{pending_tool_call['name']}_{i}"
                        })

                    print(f"[DEBUG] All {len(pending_tool_calls)} tools executed, continuing conversation with results")
                    # Continue the while loop to get LLM's response to the tool result
                    # (LLM may respond with content, thinking, or another tool call)

                else:
                    # No tool call, conversation is complete
                    print(f"[DEBUG] No pending tool call, conversation complete")
                    break
            
            # Log total messages sent to LLM after all iterations
            print(f"[DEBUG] Total messages in llm_messages after {tool_iteration} iterations: {len(llm_messages)}")
            
            # Save assistant message with message blocks for sequential display
            # Always save if we have any content, thinking, or message blocks
            assistant_saved = False
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
                await add_message(
                    db, conversation_id, "assistant", assistant_message,
                    blocks=consolidated_blocks or None,
                    extra_metadata=message_extra_metadata,
                    version=version,
                    version_group=version_group
                )
                # Commit immediately so the message is persisted even if client disconnects
                await db.commit()
                assistant_saved = True

            # Generate title using model (reuses KV cache via cache_prompt: true)
            # Skip if the conversation already has a meaningful title (not the default)
            existing_title = conversation.title if conversation else None
            if assistant_saved and (not existing_title or existing_title == 'New Chat'):
                async with get_db() as title_db:
                    msgs = await get_conversation_messages(title_db, conversation_id)
                    user_count = len([m for m in msgs if m["role"] == "user"])
                    assistant_count = len([m for m in msgs if m["role"] == "assistant"])

                if user_count == 1 and assistant_count == 1:
                    title = await _generate_title_with_model(
                        llm_messages, assistant_message, llm_client, tools=all_tools, model=model
                    )
                    if title:
                        await update_conversation_title(db, conversation_id, title)
                        await db.commit()
                        yield f"data: {json.dumps({'type': 'title_update', 'title': title})}\n\n"

            # Yield done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except asyncio.CancelledError:
        # Request was cancelled by client - this is normal, don't log as error
        print(f"Request {request_id} cancelled by client")
        raise  # Re-raise to properly propagate cancellation
    except Exception as e:
        print(f"Error in event generator: {e}")
        import traceback
        traceback.print_exc()
        try:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        except Exception:
            pass  # Client may have disconnected


@app.get("/api/stream/{request_id}")
async def stream_response(
    request_id: str,
    conversation_id: str,
    enable_rag: bool = False,
    model: str = None,
    document_ids: str = None
):
    """Stream LLM response with real-time tool execution updates."""
    doc_ids = document_ids.split(",") if document_ids else None
    return StreamingResponse(
        _core_stream_handler(request_id, conversation_id, enable_rag, model, doc_ids),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/stream/regenerate/{request_id}")
async def stream_regenerate_response(
    request_id: str,
    conversation_id: str,
    model: str = None,
    version: int = 1,
    version_group: str = None
):
    """Stream regenerated LLM response using unified handler.
    Supports versioned regeneration through version/version_group params.
    """
    return StreamingResponse(
        _core_stream_handler(
            request_id, conversation_id,
            enable_rag=False, model=model,
            version=version, version_group=version_group
        ),
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
    # Get runtime server status from MCP manager
    runtime_servers = await mcp_manager.list_servers()

    # Get all servers from database (both enabled and disabled)
    async with get_db() as db:
        db_servers = await get_all_mcp_servers(db)
        db_enabled = {s["name"]: s["enabled"] for s in db_servers}

    # Merge runtime info with enabled status and disabled_tools
    servers_with_status = []
    for server in runtime_servers:
        db_info = next((s for s in db_servers if s["name"] == server["name"]), {})
        servers_with_status.append({
            **server,
            "enabled": db_enabled.get(server["name"], True),
            "disabled_tools": db_info.get("disabled_tools", []),
            "timeout": db_info.get("timeout", 60.0)
        })

    # Also include servers from DB that might not be connected
    for db_server in db_servers:
        if not any(s["name"] == db_server["name"] for s in runtime_servers):
            servers_with_status.append({
                "name": db_server["name"],
                "transport_type": db_server.get("transport_type", "stdio"),
                "command": db_server.get("command"),
                "args": db_server.get("args"),
                "env": db_server.get("env"),
                "url": db_server.get("url"),
                "tool_count": 0,
                "is_connected": False,
                "is_initialized": False,
                "error": None,
                "enabled": db_server.get("enabled", True),
                "disabled_tools": db_server.get("disabled_tools", [])
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
    timeout = data.get("timeout", 60.0)
    
    # Validate based on transport type
    if transport_type in ("sse", "streamable-http"):
        if not url:
            raise HTTPException(status_code=400, detail="URL is required for SSE/StreamableHTTP transport")
    elif transport_type == "stdio":
        if not command:
            raise HTTPException(status_code=400, detail="Command is required for stdio transport")

    success, error = await mcp_manager.add_server(name, command, args, env, transport_type, url, timeout=timeout)

    if success:
        return {"status": "success", "message": f"Server '{name}' added and connected successfully", "connected": True}
    else:
        return {
            "status": "warning",
            "message": f"Server '{name}' added but connection failed.",
            "connected": False,
            "error": error
        }


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
    timeout = data.get("timeout", 60.0)
    
    # Validate based on transport type
    if transport_type in ("sse", "http", "streamable-http"):
        if not url:
            raise HTTPException(status_code=400, detail="URL is required for SSE/HTTP transport")
    elif transport_type == "stdio":
        if not command:
            raise HTTPException(status_code=400, detail="Command is required for stdio transport")
    
    # First remove the old server
    await mcp_manager.remove_server(server_name)
    
    # Add with new configuration (using new name if provided)
    success, error = await mcp_manager.add_server(
        name=new_name,
        command=command,
        args=args,
        env=env,
        transport_type=transport_type,
        url=url,
        timeout=timeout
    )

    if success:
        return {"status": "success", "message": f"Server '{new_name}' updated successfully", "connected": True}
    else:
        return {
            "status": "warning",
            "message": f"Server '{new_name}' updated but connection failed.",
            "connected": False,
            "error": error
        }


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
    When disabling, disconnects the server from the runtime manager.
    When enabling, reconnects the server with stored config.
    """
    from database.crud import toggle_mcp_server as db_toggle_mcp_server
    from database.crud import get_all_mcp_servers

    data = await request.json()
    enabled = data.get("enabled", True)

    async with get_db() as db:
        await db_toggle_mcp_server(db, server_name, enabled)

    if enabled:
        # Reconnect: fetch config from DB and connect
        async with get_db() as db:
            all_servers = await get_all_mcp_servers(db)
        config_data = next((s for s in all_servers if s["name"] == server_name), None)
        if config_data:
            config = MCPServerConfig(
                name=config_data["name"],
                transport_type=config_data.get("transport_type", "stdio"),
                command=config_data.get("command"),
                args=config_data.get("args", []),
                env=config_data.get("env", {}),
                url=config_data.get("url")
            )
            await mcp_manager._connect_server(config)
    else:
        # Disconnect: remove from manager but keep in DB
        if server_name in mcp_manager.servers:
            instance = mcp_manager.servers[server_name]
            if instance.client and instance.is_connected:
                try:
                    await instance.client.close()
                except Exception:
                    pass
            del mcp_manager.servers[server_name]

    return {"status": "success", "message": f"Server '{server_name}' {'enabled' if enabled else 'disabled'}"}


@app.put("/api/mcp/servers/{server_name}/tools/toggle")
async def toggle_mcp_tool_endpoint(server_name: str, request: Request):
    """
    Enable or disable a specific tool on an MCP server.
    Disabled tools are excluded from the LLM prompt, saving context tokens.
    """
    data = await request.json()
    tool_name = data.get("tool_name")
    disabled = data.get("disabled", True)
    
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool_name is required")
    
    # Update in database
    async with get_db() as db:
        await update_mcp_server_disabled_tools(db, server_name, tool_name, disabled)
    
    # Update runtime config so filtering takes effect immediately
    if server_name in mcp_manager.servers:
        instance = mcp_manager.servers[server_name]
        current = set(instance.config.disabled_tools or [])
        if disabled:
            current.add(tool_name)
        else:
            current.discard(tool_name)
        instance.config.disabled_tools = list(current)
    
    return {
        "status": "success",
        "message": f"Tool '{tool_name}' {'disabled' if disabled else 'enabled'} on '{server_name}'"
    }


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


@app.get("/api/messages/{message_id}/versions")
async def get_message_versions_endpoint(message_id: str):
    """Get all versions of a message by message_id.
    Returns empty list if the message has no version_group.
    """
    async with get_db() as db:
        versions = await get_message_versions(db, message_id)
        return {"versions": versions}


@app.get("/api/versions/{version_group}")
async def get_versions_by_group(version_group: str):
    """Get all versions in a version_group.
    Uses the version_group directly (no message_id lookup needed).
    """
    from sqlalchemy import select
    from database.models import Message
    async with get_db() as db:
        result = await db.execute(
            select(Message)
            .where(Message.version_group == version_group)
            .order_by(Message.version)
        )
        versions_obj = result.scalars().all()
        versions = [
            {
                "id": v.id,
                "role": v.role,
                "content": v.content,
                "tool_calls": v.tool_calls,
                "thinking": v.thinking,
                "metadata": v.extra_metadata,
                "blocks": v.extra_metadata.get('blocks') if v.extra_metadata else None,
                "version": v.version,
                "version_group": v.version_group,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions_obj
        ]
        return {"versions": versions}


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
    """Regenerate an assistant response with versioning.
    Instead of deleting old messages, creates a new version of the response.
    """
    import uuid as uuid_lib
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
            messages = await get_conversation_messages(db, conversation_id, only_latest_versions=False)
            # Find last assistant message
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            if not assistant_messages:
                raise HTTPException(status_code=400, detail="No assistant message to regenerate")
            message = assistant_messages[-1]
        
        # Find the user message that preceded this assistant message
        # Use all messages (not just latest versions) to find the right position
        messages = await get_conversation_messages(db, conversation_id, only_latest_versions=False)
        
        # Find the index of the current message (last version in its group if it has a group)
        msg_target_id = message.get("id")
        msg_index = -1
        for i, m in enumerate(messages):
            if m.get("id") == msg_target_id:
                msg_index = i
                break
        
        if msg_index <= 0:
            raise HTTPException(status_code=400, detail="Could not find preceding user message")
        
        # Walk backwards from msg_index to find the actual preceding user message,
        # skipping any old versions of the same assistant response (same version_group)
        target_version_group = message.get("version_group")
        user_message = None
        for i in range(msg_index - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "user":
                user_message = m
                break
            # If this is an old version of the same response, skip it
            if m.get("version_group") == target_version_group:
                continue
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Could not find preceding user message")
        
        # Determine version_group and next version number
        current_version_group = message.get("version_group")
        current_version = message.get("version", 1)
        
        if current_version_group:
            # This message already has versions — increment
            new_version = current_version + 1
            version_group = current_version_group
        else:
            # First regenerate — create a version_group
            version_group = str(uuid_lib.uuid4())
            new_version = current_version + 1
            
            # Update the original message with the version_group
            from sqlalchemy import select
            from database.models import Message as MsgModel
            result = await db.execute(
                select(MsgModel).where(MsgModel.id == msg_target_id)
            )
            original_msg = result.scalar_one_or_none()
            if original_msg:
                original_msg.version = current_version
                original_msg.version_group = version_group
        
        # Create new request ID
        request_id = str(uuid.uuid4())
        
        return {
            "request_id": request_id,
            "status": "processing",
            "conversation_id": conversation_id,
            "version": new_version,
            "version_group": version_group,
            "superseded_message_id": msg_target_id
        }


# Chat File Upload
@app.post("/api/upload/chat-file")
async def upload_chat_file(file: UploadFile = File(...)):
    """Upload a file for use in a chat message (images, documents, etc.)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_UPLOAD_SIZE} bytes")
    
    # Detect content type from file or use provided
    import mimetypes
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    
    # Generate unique filename preserving extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    
    # Save file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)
    
    return {
        "url": f"/uploads/{unique_name}",
        "filename": file.filename,
        "type": content_type,
        "size": file_size
    }


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


# Agent Management
def _agent_to_dict(agent) -> Dict:
    """Serialize an Agent model to a dictionary."""
    return {
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


@app.get("/api/agents")
async def list_agents():
    """List all agents"""
    async with get_db() as db:
        agents = await get_all_agents(db)
        return {"agents": [_agent_to_dict(a) for a in agents]}


@app.get("/api/agents/{agent_id}")
async def get_agent_detail(agent_id: int):
    """Get agent details"""
    async with get_db() as db:
        agent = await get_agent(db, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"agent": _agent_to_dict(agent)}


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
        return {"agent": _agent_to_dict(agent)}


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
        return {"agent": _agent_to_dict(agent)}


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


# Backup Management
@app.get("/api/backup/status")
async def get_backup_status():
    """Get database backup status and history"""
    from backend.database.backup import get_backup_status
    return get_backup_status()


@app.post("/api/backup/run")
async def trigger_backup():
    """Trigger a manual database backup"""
    from backend.database.backup import backup_database
    result = backup_database()
    if result.get("success"):
        return {"status": "success", "message": "Backup completed", "file": result["file"], "size": result["size"]}
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Backup failed"))


@app.post("/api/backup/scheduler/restart")
async def restart_backup_scheduler():
    """Restart the backup scheduler (e.g., after settings change)"""
    backup_scheduler.restart()
    return {"status": "success", "message": "Backup scheduler restarted"}


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


# ─── Tags ─────────────────────────────────────────────────
@app.put("/api/conversations/{conversation_id}/tags")
async def update_tags(conversation_id: str, request: Request):
    """Update tags for a conversation"""
    data = await request.json()
    tags = data.get("tags", [])
    async with get_db() as db:
        result = await update_conversation_tags(db, conversation_id, tags)
        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result


@app.put("/api/conversations/{conversation_id}/agent")
async def update_conversation_agent_endpoint(conversation_id: str, request: Request):
    """Update the agent associated with a conversation"""
    data = await request.json()
    agent_id = data.get("agent_id")
    async with get_db() as db:
        result = await update_conversation_agent(db, conversation_id, agent_id)
        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result


# ─── Notes ────────────────────────────────────────────────
@app.get("/api/notes")
async def list_notes():
    """Get all notes"""
    async with get_db() as db:
        notes = await get_all_notes(db)
        return {"notes": notes}


@app.get("/api/conversations/{conversation_id}/notes")
async def get_conversation_notes(conversation_id: str):
    """Get notes for a specific conversation"""
    async with get_db() as db:
        notes = await get_notes_for_conversation(db, conversation_id)
        return {"notes": notes}


@app.post("/api/notes")
async def add_note(request: Request):
    """Create a new note"""
    data = await request.json()
    conversation_id = data.get("conversation_id")
    message_id = data.get("message_id")
    content = data.get("content", "")
    source_text = data.get("source_text")
    
    if not conversation_id or not content.strip():
        raise HTTPException(status_code=400, detail="conversation_id and content are required")
    
    async with get_db() as db:
        note = await create_note(db, conversation_id, message_id, content.strip(), source_text)
        return {"note": note}


@app.delete("/api/notes/{note_id}")
async def delete_note_endpoint(note_id: str):
    """Delete a note"""
    async with get_db() as db:
        success = await delete_note(db, note_id)
        if not success:
            raise HTTPException(status_code=404, detail="Note not found")
        return {"status": "success", "message": "Note deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, debug=DEBUG)

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import asyncio
import json
import os
import threading
import uuid
import contextvars
from typing import AsyncGenerator, Dict, List, Optional

from tools.base import current_document_ids

from settings import APP_HOST, APP_PORT, DEBUG, MAX_UPLOAD_SIZE, UPLOAD_DIR, OUTPUTS_DIR
from database.models import init_db, get_db, shutdown_db
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


async def _bootstrap_default_provider():
    """On first run, seed a default LLM provider from the legacy settings
    (llama_cpp_base_url) so existing installs keep working unchanged."""
    try:
        from database.provider_crud import list_providers, create_provider
        from tools.provider_service import fetch_models
        async with get_db() as db:
            providers = await list_providers(db)
            if providers:
                return
            base_url = settings_manager.get_settings().get("llama_cpp_base_url", "http://localhost:8080")
            try:
                models = await fetch_models(base_url, None, timeout=15)
                print(f"[PROVIDER] bootstrapped default provider with {len(models)} models")
            except Exception as e:
                models = []
                print(f"[PROVIDER] bootstrap fetch failed (models will be empty): {e}")
            await create_provider(
                db, "llama.cpp", base_url.rstrip("/"), api_key=None,
                models=models, is_default=1, enabled=1
            )
            await db.commit()
    except Exception as e:
        print(f"[PROVIDER] bootstrap failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await _bootstrap_default_provider()
    await mcp_manager.initialize()
    backup_scheduler.start()
    yield
    # Shutdown
    await backup_scheduler.stop()
    await mcp_manager.cleanup()
    await shutdown_db()

app = FastAPI(title="LLM UI with MCP Support", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
# Mount uploads directory so uploaded files are publicly accessible
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
# Mount outputs directory (agent platform: job outputs, generated files)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
templates = Jinja2Templates(directory="frontend/templates")

# Initialize components
llm_client = LLMClient()
tool_executor = ToolExecutor(mcp_manager)

# Set TTS service in settings manager
settings_manager.set_tts_service(tool_executor.tts_service)

# Initialize STT service
from tools.stt_service import STTService, STTConfig
stt_service = STTService(STTConfig.from_settings(settings_manager.get_settings()))


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
            "files": files,
            "provider_id": data.get("provider_id")
        }




async def _generate_title_with_model(
    llm_messages: list,
    assistant_message: str,
    llm_client,
    tools: list = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    thinking_content: str = None
) -> str:
    """
    Generate a title by appending to the existing conversation messages.
    KV cache reuses the prefix from the main response.
    The appended messages are not saved to DB — only the title is returned.
    The title generation request is built from a copy of llm_messages, so the
    original is untouched and future requests can still reuse the KV cache.
    """
    title_messages = list(llm_messages)
    title_assistant = {"role": "assistant", "content": assistant_message}
    if thinking_content:
        title_assistant["reasoning_content"] = thinking_content
    title_messages.append(title_assistant)
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
            tool_choice="none",
            base_url=base_url,
            api_key=api_key
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

    import re
    title = re.sub(r'<think>.*?</think>', '', title, flags=re.DOTALL)
    title = re.sub(r'<[^>]+>', '', title).strip()
    words = title.split()
    if not words:
        return ""
    title = ' '.join(words[:6]).strip().rstrip('.,;:!?\\-"\'')
    return title[:60]


async def _maybe_reflect_and_propose_skill(db, conversation_id: str, llm_client, model=None,
                                           base_url=None, api_key=None):
    """Self-improvement loop (Phase 4): propose a skill draft after a
    multi-tool task.

    Runs on the same cadence as memory extraction (every N assistant turns),
    only for turns that used tools. Output is always a DRAFT under
    skills/_drafts/ — the user accepts or rejects it in the Skills modal.
    Never writes live skills silently.
    """
    try:
        from settings import settings_manager
        interval = settings_manager.get_settings().get("memory_auto_extract_interval", 3) or 0
        if interval <= 0:
            return
        msgs = await get_conversation_messages(db, conversation_id)
        assistant_count = len([m for m in msgs if m["role"] == "assistant"])
        if assistant_count % interval != 0:
            return
        last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), None)
        last_assistant = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None)
        if not last_user or not last_assistant:
            return
        # Only reflect when tools were used in the last assistant message.
        last_msg = msgs[-1]
        blocks = (last_msg.get("metadata") or {}).get("blocks") or []
        tool_names = [b.get("name") for b in blocks if b.get("type") == "tool_call"]
        if not tool_names:
            return
        prompt = (
            "You are a skill-builder for an AI assistant. Decide whether this task is worth "
            "turning into a reusable skill.\n\n"
            f"USER: {last_user[:1500]}\n\nASSISTANT: {last_assistant[:1500]}\n\n"
            f"TOOLS USED: {', '.join(tool_names[:10])}\n\n"
            "Rules:\n"
            "- CREATE a skill when the task was a repeatable, multi-step procedure "
            "(at least 2 concrete steps).\n"
            "- IMPROVE an existing skill only when load_skill was used and the run "
            "seems to have gone wrong or could clearly be better.\n"
            "- Otherwise output none.\n"
            'Output ONLY a JSON object: {"action": "create"|"improve"|"none", '
            '"name": "short-slug", "description": "one line", '
            '"instructions": "numbered steps", "reason": "short justification"}'
        )
        raw = ""
        async for chunk in llm_client.stream_chat(
            [{"role": "user", "content": prompt}],
            model=model, temperature=0.0, max_tokens=700, tools=None,
            base_url=base_url, api_key=api_key
        ):
            if chunk.get("type") == "content":
                raw += chunk.get("content", "")
        import re as _re
        import json as _json
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            print("[SKILLS] reflection: no JSON in model output")
            return
        try:
            decision = _json.loads(m.group(0))
        except _json.JSONDecodeError as e:
            print(f"[SKILLS] reflection: unparseable JSON: {e}")
            return
        action = str(decision.get("action") or "none").strip().lower()
        if action not in ("create", "improve"):
            print(f"[SKILLS] reflection: action={action} — no draft")
            return
        name = str(decision.get("name") or "").strip()
        description = str(decision.get("description") or "").strip()
        instructions = str(decision.get("instructions") or "").strip()
        reason = str(decision.get("reason") or "").strip()
        if not name or not instructions:
            print("[SKILLS] reflection: missing name/instructions")
            return
        from tools.skills_tool import write_skill, get_skill
        if action == "improve" and not get_skill(name):
            print(f"[SKILLS] reflection: improve target '{name}' does not exist — skipping")
            return
        body = instructions
        if reason:
            body = f"<!-- reflection reason: {reason} -->\n\n" + body
        skill = write_skill(name, description or name, body, draft=True)
        print(f"[SKILLS] reflection: draft proposed: {name} ({action}) — review in Skills modal")
    except Exception as e:
        print(f"[SKILLS] reflection failed: {e}")


async def _extract_memory_from_exchange(db, conversation_id: str, llm_client, agent_id=None, model=None,
                                        base_url=None, api_key=None):
    """Auto-extract durable facts from the last user↔assistant exchange (Phase 2).

    Runs every `memory_auto_extract_interval` assistant turns. Uses a fast,
    low-max-token completion; failures are logged, never propagated.
    """
    try:
        from settings import settings_manager
        interval = settings_manager.get_settings().get("memory_auto_extract_interval", 3) or 0
        if interval <= 0:
            return
        msgs = await get_conversation_messages(db, conversation_id)
        assistant_count = len([m for m in msgs if m["role"] == "assistant"])
        if assistant_count % interval != 0:
            return
        last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), None)
        last_assistant = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None)
        if not last_user or not last_assistant:
            return
        prompt = (
            "Extract durable, reusable facts or user preferences from this conversation exchange. "
            'Output ONLY a JSON array of strings. Each string must be a concise, standalone fact '
            "(max ~120 chars) that would still be useful in future sessions. If nothing durable, output [].\n\n"
            f"User: {last_user[:1500]}\n\nAssistant: {last_assistant[:1500]}"
        )
        parts = []
        async for chunk in llm_client.stream_chat(
            [{"role": "user", "content": prompt}],
            model=model, temperature=0.0, max_tokens=512, tools=None,
            base_url=base_url, api_key=api_key
        ):
            if chunk.get("type") == "content":
                parts.append(chunk.get("content", ""))
        raw = "".join(parts)
        import re
        import json as _json
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                parsed = _json.loads(m.group(0))
                items = [str(x).strip() for x in parsed if str(x).strip()]
            except _json.JSONDecodeError:
                items = [l.strip("- ").strip() for l in raw.splitlines() if l.strip().startswith("-")]
        else:
            items = [l.strip("- ").strip() for l in raw.splitlines()
                     if l.strip() and len(l.strip()) > 5 and not l.strip().startswith("```")]
        if not items:
            return
        from database.memory_crud import create_memory_entry
        scope = f"agent:{agent_id}" if agent_id is not None else "global"
        added = 0
        for item in items[:10]:
            await create_memory_entry(db, item, scope=scope, source="auto")
            added += 1
        if added:
            await db.commit()
            print(f"[MEMORY] auto-extracted {added} fact(s) (scope={scope})")
    except Exception as e:
        print(f"[MEMORY] extraction failed: {e}")


async def _stream_with_stall_timeout(generator, initial_timeout=600, stall_timeout=60):
    """Wrap an async generator with two-phase stall detection.

    Replaces a total-time watchdog so the LLM can generate for arbitrarily long
    responses as long as it keeps producing chunks. Compatible with both MTP
    and non-MTP models.

    Two phases:
      1. Initial: generous timeout (default 600s) to allow for long prompt
         processing / prefill before the first chunk arrives.
      2. Stall: shorter timeout (default 60s) for subsequent chunks — catches
         genuine stalls where the model stops generating.

    Args:
        generator: An async generator (e.g. llm_client.stream_chat(...)).
        initial_timeout: Max seconds to wait for the FIRST chunk (prompt processing).
        stall_timeout: Max seconds to wait for the NEXT chunk after the first one.

    Yields:
        Each chunk from the wrapped generator.
    """
    first_chunk = True
    try:
        while True:
            timeout = initial_timeout if first_chunk else stall_timeout
            chunk = await asyncio.wait_for(
                generator.__anext__(),
                timeout=timeout
            )
            first_chunk = False
            yield chunk
    except StopAsyncIteration:
        pass
    except asyncio.TimeoutError:
        which = "initial (prompt processing)" if first_chunk else "inter-chunk"
        print(f"[WATCHDOG] Stream stalled for {timeout}s ({which}) — no chunks received")


async def _core_stream_handler(
    request_id: str,
    conversation_id: str,
    enable_rag: bool = False,
    model: Optional[str] = None,
    document_ids: Optional[list] = None,
    version: int = 1,
    version_group: Optional[str] = None,
    provider_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Universal SSE handler for streaming LLM responses and tool execution.
    
    Args:
        version: Version number for regenerated responses
        version_group: UUID shared by all versions of the same response
        provider_id: LLM provider to use (falls back to the default provider)
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
                            "provider_id": getattr(agent, "provider_id", None),
                            "temperature": agent.temperature,
                            "top_k": agent.top_k,
                            "max_tokens": agent.max_tokens,
                            "enable_rag": bool(agent.enable_rag),
                            "enabled_tools": agent.enabled_tools or [],
                            "enabled_mcp_servers": agent.enabled_mcp_servers or [],
                            "enabled_skills": agent.enabled_skills or []
                        }

            # Resolve LLM provider: agent's provider > requested provider > default.
            from database.provider_crud import get_provider as _get_provider, get_default_provider
            resolved_provider_id = None
            if agent_config and agent_config.get("provider_id"):
                resolved_provider_id = agent_config["provider_id"]
            elif provider_id:
                resolved_provider_id = provider_id
            provider = None
            try:
                if resolved_provider_id:
                    provider = await _get_provider(db, resolved_provider_id, include_api_key=True)
                if provider is None:
                    provider = await get_default_provider(db, include_api_key=True)
            except Exception as e:
                print(f"[PROVIDER] resolution failed: {e}")
            provider_base_url = provider.get("base_url") if provider else None
            provider_api_key = provider.get("api_key") if provider else None
            if provider:
                print(f"[PROVIDER] using '{provider['name']}' ({provider_base_url}) model={model}")
            
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

            # Inject persistent agent memory (Phase 2)
            try:
                from database.memory_crud import get_memory_for_injection
                memory_block = await get_memory_for_injection(
                    db,
                    agent_id=conversation.agent_id if conversation else None,
                    conversation_id=conversation_id
                )
                if memory_block:
                    system_prompt_content += "\n\n" + memory_block
            except Exception as e:
                print(f"[MEMORY] injection failed: {e}")

            # Inject skills index (Phase 3) — compact, one line per skill.
            try:
                from tools.skills_tool import skill_index
                idx = skill_index()
                if idx:
                    # Agent can restrict which skills are visible; empty = all.
                    enabled_skills = set()
                    if agent_config and agent_config.get("enabled_skills"):
                        enabled_skills = set(agent_config["enabled_skills"])
                    if enabled_skills:
                        idx = "\n".join(
                            line for line in idx.splitlines()
                            if line.startswith("- ")
                            and line[2:].split(":", 1)[0].strip() in enabled_skills
                        )
                    if idx:
                        system_prompt_content += (
                            "\n\n### Available skills\n"
                            "Call load_skill(name) to load a skill's full instructions "
                            "when it is relevant to the current task.\n" + idx
                        )
            except Exception as e:
                print(f"[SKILLS] index injection failed: {e}")
            
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
                            # Try to read and include file content for text-based files and PDFs
                            f_path = os.path.join(UPLOAD_DIR, os.path.basename(f_url))
                            text_exts = {'.txt','.md','.json','.csv','.yaml','.yml','.xml','.log',
                                         '.py','.js','.ts','.jsx','.tsx','.html','.css','.scss','.less',
                                         '.sh','.bash','.zsh','.env','.ini','.cfg','.conf',
                                         '.sql','.r','.rb','.php','.go','.rs','.java','.kt','.swift',
                                         '.c','.cpp','.h','.hpp','.toml','.gradle','.makefile','.dockerfile',
                                         '.diff','.patch','.svg'}
                            file_ext = os.path.splitext(f_name)[1].lower()

                            if file_ext == '.pdf' and os.path.exists(f_path):
                                try:
                                    from pypdf import PdfReader
                                    pdf_content = []
                                    char_count = 0
                                    for page in PdfReader(f_path).pages:
                                        text = (page.extract_text() or '').strip()
                                        if text:
                                            remaining = 100000 - char_count
                                            if remaining <= 0:
                                                pdf_content.append('[...truncated...]')
                                                break
                                            pdf_content.append(text[:remaining])
                                            char_count += len(text[:remaining])
                                    pdf_text = '\n\n'.join(pdf_content)
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"\n\nThe user attached the PDF '{f_name}'. Its content:\n\n{pdf_text}"
                                    })
                                except ImportError:
                                    content_parts.append({"type": "text", "text": f"\n[Attached PDF: {f_name}]"})
                                except Exception as e:
                                    print(f"[FILE] Error reading PDF {f_path}: {e}")
                                    content_parts.append({"type": "text", "text": f"\n[Attached PDF: {f_name}]"})

                            elif file_ext == '.docx' and os.path.exists(f_path):
                                try:
                                    from docx import Document
                                    paras = [p.text for p in Document(f_path).paragraphs if p.text]
                                    doc_text = '\n\n'.join(paras)[:100000]
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"\n\nThe user attached the Word document '{f_name}'. Its content:\n\n{doc_text}"
                                    })
                                except ImportError:
                                    content_parts.append({"type": "text", "text": f"\n[Attached Word document: {f_name}]"})
                                except Exception as e:
                                    print(f"[FILE] Error reading DOCX {f_path}: {e}")
                                    content_parts.append({"type": "text", "text": f"\n[Attached Word document: {f_name}]"})

                            elif file_ext == '.xlsx' and os.path.exists(f_path):
                                try:
                                    from openpyxl import load_workbook
                                    wb = load_workbook(f_path, read_only=True, data_only=True)
                                    sheets_text = []
                                    char_count = 0
                                    for sheet_name in wb.sheetnames:
                                        ws = wb[sheet_name]
                                        rows = []
                                        max_rows = 60
                                        for i, row in enumerate(ws.iter_rows(values_only=True)):
                                            if i >= max_rows:
                                                rows.append(f'... ({ws.max_row or "?"} total rows)')
                                                break
                                            row_vals = [str(c) if c is not None else '' for c in row]
                                            rows.append(' | '.join(row_vals))
                                        sheet_text = f'--- Sheet: {sheet_name} ---\n' + '\n'.join(rows)
                                        remaining = 100000 - char_count
                                        if remaining <= 0:
                                            break
                                        sheets_text.append(sheet_text[:remaining])
                                        char_count += len(sheet_text[:remaining])
                                    wb.close()
                                    xlsx_text = '\n\n'.join(sheets_text)
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"\n\nThe user attached the spreadsheet '{f_name}'. Its content:\n\n{xlsx_text}"
                                    })
                                except ImportError:
                                    content_parts.append({"type": "text", "text": f"\n[Attached spreadsheet: {f_name}]"})
                                except Exception as e:
                                    print(f"[FILE] Error reading XLSX {f_path}: {e}")
                                    content_parts.append({"type": "text", "text": f"\n[Attached spreadsheet: {f_name}]"})

                            elif file_ext in text_exts and os.path.exists(f_path):
                                try:
                                    with open(f_path, "r", encoding="utf-8", errors="replace") as text_f:
                                        file_content = text_f.read(100000)
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"\n\nThe user attached the file '{f_name}'. Its content:\n\n```{file_ext.lstrip('.')}\n{file_content}\n```"
                                    })
                                except Exception as e:
                                    print(f"[FILE] Error reading {f_path}: {e}")
                                    content_parts.append({"type": "text", "text": f"\n[Attached file: {f_name}]"})

                            else:
                                content_parts.append({"type": "text", "text": f"\n[Attached file: {f_name}]"})
                    llm_messages.append({"role": role, "content": content_parts})
                else:
                    llm_msg = {"role": role, "content": content}
                    # Strict reasoning providers require the original reasoning
                    # (thinking mode) to be echoed back with assistant messages.
                    if role == "assistant" and msg.get("thinking"):
                        llm_msg["reasoning_content"] = msg["thinking"]
                    llm_messages.append(llm_msg)
            
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
                for custom_tool in (
                    "generate_speech", "run_command",
                    "memory_write", "memory_read", "memory_search", "memory_delete",
                    "load_skill", "create_skill", "run_job",
                    "list_agents", "create_agent", "delete_agent",
                    "list_mcp_servers", "add_mcp_server", "remove_mcp_server",
                    "list_providers", "add_provider", "search_skills", "install_skill",
                ):
                    if custom_tool not in enabled_custom:
                        exclude_tools.append(custom_tool)
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

            # Context transparency event (Phase 6) — what the model sees this turn.
            try:
                yield f"data: {json.dumps({'type': 'context_info', 'context': {
                    'model': model or 'default',
                    'system_prompt': system_prompt_content,
                    'message_count': len(llm_messages),
                    'tool_count': len(all_tools),
                    'tools': [t.get('function', {}).get('name', '?') for t in all_tools],
                }})}\n\n"
            except Exception as e:
                print(f"[CONTEXT] event failed: {e}")
            
            # Main conversation loop - handles multiple tool calls with content in between
            max_tool_iterations = 35  # Prevent infinite loops
            tool_iteration = 0

            while tool_iteration < max_tool_iterations:
                tool_iteration += 1
                print(f"[DEBUG] Conversation loop iteration {tool_iteration}")

                # Stream LLM response
                assistant_message, thinking_content = "", ""
                pending_tool_calls = []

                # Two-phase stall watchdog: generous initial timeout for prompt
                # processing (long contexts can take 80s+), shorter timeout for
                # inter-chunk stalls. Continues indefinitely while chunks arrive.
                try:
                    async for chunk in _stream_with_stall_timeout(
                        llm_client.stream_chat(llm_messages, model=model, tools=all_tools,
                                               base_url=provider_base_url, api_key=provider_api_key),
                        initial_timeout=600,
                        stall_timeout=60
                    ):
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

                            elif chunk_type == "error":
                                # LLM request failure (connection, auth, invalid model...) —
                                # surface it instead of silently ending with an empty reply.
                                error_msg = chunk.get("error", "LLM request failed")
                                print(f"[LLM ERROR] {error_msg}")
                                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                                break

                        except Exception as e:
                            print(f"[DEBUG] Error processing chunk: {e}")
                            print(f"[DEBUG] Chunk data: {chunk}")
                            import traceback
                            traceback.print_exc()
                        await asyncio.sleep(0)
                except asyncio.TimeoutError:
                    # Safety net — _stream_with_stall_timeout already catches this,
                    # but handle it here to prevent unhandled exception propagation.
                    print(f"[WATCHDOG] Unhandled timeout in stream processing for request {request_id}")
                    # Fall through to save partial response and yield done

                # If we have pending tool calls, execute them and continue the loop
                if pending_tool_calls:
                    # The OpenAI-compatible protocol requires a role='tool' message to be
                    # the response to a preceding assistant message carrying tool_calls.
                    # Some backends (llama.cpp proxying to strict upstream providers)
                    # reject the request with HTTP 400 if this pairing is missing.
                    assistant_tool_calls = []
                    for i, tc in enumerate(pending_tool_calls):
                        assistant_tool_calls.append({
                            "id": f"{tc['name']}_{i}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else str(tc.get("arguments") or "{}"),
                            },
                        })
                    assistant_tc_msg = {"role": "assistant", "tool_calls": assistant_tool_calls}
                    if assistant_message.strip():
                        assistant_tc_msg["content"] = assistant_message
                    # Strict reasoning providers require the original reasoning
                    # (thinking mode) to be echoed back with the assistant message.
                    if thinking_content:
                        assistant_tc_msg["reasoning_content"] = thinking_content
                    llm_messages.append(assistant_tc_msg)

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
                            request_id,
                            call_key=pending_tool_call.get('key'),
                            conversation_id=conversation_id,
                            skill_allowlist=(agent_config or {}).get("enabled_skills") or None
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
                    # Record skill usage (Phase 4) for the improvement loop.
                    try:
                        from database.skill_crud import record_skill_run
                        for tc in pending_tool_calls:
                            if tc.get("name") == "load_skill":
                                skill_name = str((tc.get("arguments") or {}).get("name", "")).strip()
                                if skill_name:
                                    await record_skill_run(
                                        db, skill_name, conversation_id,
                                        success=tc.get("status") not in ("error",)
                                    )
                        await db.commit()
                    except Exception as e:
                        print(f"[SKILLS] run-log failed: {e}")
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

            # Finalize open job runs for this conversation (Phase 5).
            if assistant_saved:
                try:
                    from database.job_crud import list_job_runs, finish_job_run
                    import os as _os
                    from settings import OUTPUTS_DIR
                    runs = await list_job_runs(db, limit=200)
                    open_runs = [r for r in runs
                                 if r["conversation_id"] == conversation_id and r["status"] == "running"]
                    if open_runs:
                        jobs_dir = _os.path.join(OUTPUTS_DIR, "jobs")
                        _os.makedirs(jobs_dir, exist_ok=True)
                        for run in open_runs:
                            if assistant_message.strip():
                                output_path = _os.path.join(jobs_dir, f"{run['id']}.md")
                                try:
                                    with open(output_path, "w", encoding="utf-8") as f:
                                        f.write(f"# Job: {run['job_name']}\n\n{assistant_message}")
                                except Exception as e:
                                    print(f"[JOBS] output write failed: {e}")
                                    output_path = None
                                await finish_job_run(db, run["id"], "completed", output_path=output_path)
                            else:
                                await finish_job_run(db, run["id"], "failed",
                                                     error="No assistant output produced")
                        await db.commit()
                        print(f"[JOBS] finalized {len(open_runs)} run(s) for conversation {conversation_id[:8]}")
                except Exception as e:
                    print(f"[JOBS] finalize failed: {e}")

            # Auto memory extraction (Phase 2) — every N assistant turns.
            if assistant_saved:
                try:
                    await asyncio.wait_for(
                        _extract_memory_from_exchange(
                            db, conversation_id, llm_client,
                            agent_id=conversation.agent_id if conversation else None,
                            model=model,
                            base_url=provider_base_url, api_key=provider_api_key
                        ),
                        timeout=90
                    )
                except asyncio.TimeoutError:
                    print("[MEMORY] extraction timed out")
                except Exception as e:
                    print(f"[MEMORY] extraction error: {e}")

                # Self-improvement reflection (Phase 4) — proposes skill drafts.
                try:
                    await asyncio.wait_for(
                        _maybe_reflect_and_propose_skill(db, conversation_id, llm_client, model=model,
                                                         base_url=provider_base_url, api_key=provider_api_key),
                        timeout=90
                    )
                except asyncio.TimeoutError:
                    print("[SKILLS] reflection timed out")
                except Exception as e:
                    print(f"[SKILLS] reflection error: {e}")

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
                        llm_messages, assistant_message, llm_client, tools=all_tools, model=model,
                        base_url=provider_base_url, api_key=provider_api_key,
                        thinking_content=thinking_content
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
    document_ids: str = None,
    provider_id: str = None
):
    """Stream LLM response with real-time tool execution updates."""
    doc_ids = document_ids.split(",") if document_ids else None
    return StreamingResponse(
        _core_stream_handler(request_id, conversation_id, enable_rag, model, doc_ids,
                             provider_id=provider_id),
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
    version_group: str = None,
    provider_id: str = None
):
    """Stream regenerated LLM response using unified handler.
    Supports versioned regeneration through version/version_group params.
    """
    return StreamingResponse(
        _core_stream_handler(
            request_id, conversation_id,
            enable_rag=False, model=model,
            version=version, version_group=version_group,
            provider_id=provider_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# MCP Registry Endpoints
@app.get("/api/mcp/registry")
async def search_mcp_registry(query: str = "", limit: int = 24):
    """Search installable MCP servers on the Smithery registry. Blank = most used."""
    from tools.mcp_registry import search_mcp_registry, enrich_mcp_servers
    try:
        servers = await search_mcp_registry(query, limit)
        servers = await enrich_mcp_servers(servers)
        return {"servers": servers, "error": None}
    except Exception as e:
        print(f"[MCP REGISTRY] search failed: {e}")
        return {"servers": [], "error": str(e)[:300]}


@app.post("/api/mcp/registry/install")
async def install_mcp_registry_server(request: Request):
    """Install an MCP server from the registry by its qualified name."""
    from tools.mcp_registry import install_mcp_from_registry
    data = await request.json()
    qualified_name = (data.get("qualified_name") or data.get("id") or "").strip()
    if not qualified_name:
        raise HTTPException(status_code=400, detail="Missing server id")
    try:
        result = await install_mcp_from_registry(qualified_name, mcp_manager)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[MCP REGISTRY] install failed: {e}")
        raise HTTPException(status_code=500, detail=f"Install failed: {str(e)[:300]}")
    return {"server": result}


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
            "headers": db_info.get("headers", {}),
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
                "headers": db_server.get("headers", {}),
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
    headers = data.get("headers", {}) or {}
    
    # Validate based on transport type
    if transport_type in ("sse", "streamable-http"):
        if not url:
            raise HTTPException(status_code=400, detail="URL is required for SSE/StreamableHTTP transport")
    elif transport_type == "stdio":
        if not command:
            raise HTTPException(status_code=400, detail="Command is required for stdio transport")

    success, error = await mcp_manager.add_server(name, command, args, env, transport_type, url, timeout=timeout, headers=headers)

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
    headers = data.get("headers", {}) or {}
    
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
        timeout=timeout,
        headers=headers
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
                url=config_data.get("url"),
                headers=config_data.get("headers", {})
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
    elif file_ext in [".docx"]:
        file_type = "document"
    elif file_ext in [".json", ".yaml", ".yml"]:
        file_type = "data"

    if file_type == "unknown":
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported: txt, md, pdf, docx, json, yaml, yml")
    
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
        await tool_executor.delete_document_from_rag(document_id)
        
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
    """List models across all enabled LLM providers (cached at connect/refresh)."""
    from database.provider_crud import list_providers
    async with get_db() as db:
        providers = await list_providers(db)
    models = []
    provider_list = []
    for p in providers:
        if not p.get("enabled"):
            continue
        provider_list.append({"id": p["id"], "name": p["name"], "is_default": p.get("is_default", 0)})
        for m in (p.get("models") or []):
            models.append({
                "id": m.get("id"),
                "name": m.get("name") or m.get("id"),
                "owned_by": m.get("owned_by") or p.get("name"),
                "provider_id": p.get("id"),
                "provider_name": p.get("name"),
            })
    return {"models": models, "providers": provider_list}


@app.post("/api/rag/query")
async def rag_query_endpoint(request: Request):
    """
    Direct RAG query endpoint for searching documents.
    
    This can be used for explicit document queries without LLM tool calling.
    """
    data = await request.json()
    query = data.get("query", "")
    document_ids = data.get("document_ids")
    section = data.get("section")
    top_k = min(data.get("top_k", 10), 50)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    result = await tool_executor.rag_service.query(
        query=query,
        document_ids=document_ids,
        top_k=top_k,
        section=section
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
        "enabled_skills": agent.enabled_skills or [],
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
        "enabled_skills": data.get("enabled_skills", []),
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
    if "enabled_skills" in data:
        update_data["enabled_skills"] = data["enabled_skills"]
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

        # Watch for client disconnect (pause/stop clicked): when it happens, set
        # the stop flag so long Kokoro generation bails between segments.
        stop_flag = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(request, stop_flag))
        try:
            result = await tool_executor.tts_service.generate_speech(
                text=text,
                voice=voice,
                should_stop=stop_flag.is_set
            )
        finally:
            watcher.cancel()
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "TTS generation failed"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.post("/api/tts/generate/stream")
async def generate_tts_stream(request: Request):
    """Stream TTS audio as it is generated.

    Response is NDJSON (one JSON object per line, media_type application/x-ndjson):
      {"seg": "tts_xxx.wav", "url": "/api/audio/tts_xxx.wav"}   — one segment ready
      {"done": true}                                             — all segments sent
      {"error": "..."}                                          — failure

    The client plays segments as they arrive (first sentence ≈ first audio).
    Cancelling the fetch marks the request disconnected; the server stops
    generation between sentences.
    """
    from tools.tts_service import HAS_EDGE_TTS, _check_kokoro_available
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    text = data.get("text", "")
    voice = data.get("voice")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    svc = tool_executor.tts_service
    engine = svc.config.engine
    if engine == "kokoro" and not _check_kokoro_available():
        raise HTTPException(status_code=500, detail="Kokoro engine is not available")
    if engine == "edge-tts" and not HAS_EDGE_TTS:
        raise HTTPException(status_code=500, detail="Edge TTS engine is not available")

    stop_flag = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(request, stop_flag))

    async def event_stream():
        try:
            async for filename, url in svc.stream_speech(
                text=text, voice=voice, should_stop=stop_flag.is_set
            ):
                yield json.dumps({"seg": filename, "url": url}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            watcher.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _watch_disconnect(request: Request, stop_flag):
    """Poll request.is_disconnected() from the event loop and set stop_flag."""
    while not stop_flag.is_set():
        if await request.is_disconnected():
            stop_flag.set()
            return
        await asyncio.sleep(0.2)


@app.get("/api/tts/voices")
async def list_tts_voices():
    """List available TTS voices"""
    return tool_executor.tts_service.list_available_voices()


@app.get("/api/tts/status")
async def get_tts_status():
    """Check if TTS is available"""
    from tools.tts_service import HAS_EDGE_TTS, _check_kokoro_available
    kokoro_available = _check_kokoro_available()
    return {
        "available": HAS_EDGE_TTS or kokoro_available,
        "edge_tts": HAS_EDGE_TTS,
        "kokoro": kokoro_available,
        "engine": tool_executor.tts_service.config.engine
    }


# Memory Endpoints (agent platform Phase 2)
@app.get("/api/memory")
async def list_memory(scope: str = None, limit: int = 200):
    """List persistent memory entries (optional scope filter)."""
    from database.memory_crud import list_memory_entries
    async with get_db() as db:
        entries = await list_memory_entries(db, scope=scope, limit=limit)
    return {"entries": entries}


@app.post("/api/memory")
async def add_memory(request: Request):
    """Create a memory entry."""
    from database.memory_crud import create_memory_entry
    data = await request.json()
    content = (data.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content required")
    scope = data.get("scope") or "global"
    async with get_db() as db:
        entry = await create_memory_entry(db, content, scope=scope,
                                          tags=data.get("tags") or [], source="manual")
    return {"entry": entry}


@app.patch("/api/memory/{entry_id}")
async def edit_memory(entry_id: str, request: Request):
    """Update a memory entry's content."""
    from database.memory_crud import update_memory_entry
    data = await request.json()
    async with get_db() as db:
        entry = await update_memory_entry(db, entry_id, content=data.get("content"))
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"entry": entry}


@app.delete("/api/memory/{entry_id}")
async def remove_memory(entry_id: str):
    """Delete a memory entry."""
    from database.memory_crud import delete_memory_entry
    async with get_db() as db:
        deleted = await delete_memory_entry(db, entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "ok"}


# Skill Registry Endpoints (Phase 4.5)
@app.get("/api/skills/registry")
async def search_skill_registry(query: str = "", limit: int = 25):
    """Search installable skills on the skills.sh registry. Blank query = most popular."""
    from tools.skill_registry import search_registry, popular_registry, enrich_registry
    try:
        q = (query or "").strip()
        skills = await (popular_registry(limit) if len(q) < 2 else search_registry(q, limit))
        skills = await enrich_registry(skills)
        return {"skills": skills, "error": None}
    except Exception as e:
        print(f"[REGISTRY] search failed: {e}")
        return {"skills": [], "error": str(e)[:300]}


@app.post("/api/skills/install")
async def install_registry_skill(request: Request):
    """Install a registry skill (skills.sh id, e.g. 'owner/repo/path') locally."""
    from tools.skill_registry import install_registry_skill
    data = await request.json()
    skill_id = (data.get("id") or data.get("skill_id") or "").strip()
    if not skill_id:
        raise HTTPException(status_code=400, detail="Missing skill id")
    try:
        result = await install_registry_skill(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[REGISTRY] install failed: {e}")
        raise HTTPException(status_code=500, detail=f"Install failed: {str(e)[:300]}")
    return {"skill": result}


# Skills Endpoints (agent platform Phase 3/4)
@app.get("/api/skills")
async def list_skills_api(include_drafts: bool = False):
    """List installed skills (optionally including _drafts)."""
    from tools.skills_tool import list_skills
    return {"skills": list_skills(include_drafts=include_drafts)}


@app.get("/api/skills/{name}")
async def get_skill_api(name: str):
    """Get a skill's SKILL.md content + file manifest."""
    from tools.skills_tool import get_skill
    skill = get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"skill": skill}


@app.post("/api/skills")
async def create_skill_api(request: Request):
    """Create a skill."""
    from tools.skills_tool import write_skill
    data = await request.json()
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    instructions = (data.get("instructions") or "").strip()
    if not name or not instructions:
        raise HTTPException(status_code=400, detail="name and instructions are required")
    skill = write_skill(name, description or name, instructions)
    return {"skill": skill}


@app.put("/api/skills/{name}")
async def update_skill_api(name: str, request: Request):
    """Update a skill's description/instructions."""
    from tools.skills_tool import write_skill
    data = await request.json()
    description = (data.get("description") or "").strip()
    instructions = (data.get("instructions") or "").strip()
    if not instructions:
        raise HTTPException(status_code=400, detail="instructions are required")
    skill = write_skill(name, description or name, instructions)
    return {"skill": skill}


@app.delete("/api/skills/{name}")
async def delete_skill_api(name: str):
    """Delete a skill."""
    from tools.skills_tool import delete_skill
    if not delete_skill(name):
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "ok"}


@app.post("/api/skills/drafts/{name}/accept")
async def accept_skill_draft_api(name: str):
    """Accept a self-improvement draft: move it into the live skills dir."""
    from tools.skills_tool import accept_draft
    skill = accept_draft(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"skill": skill}


@app.delete("/api/skills/drafts/{name}")
async def reject_skill_draft_api(name: str):
    """Reject a self-improvement draft: delete it."""
    from tools.skills_tool import delete_skill
    if not delete_skill(name, draft=True):
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"status": "ok"}


# LLM Provider Endpoints (multi-provider support)
@app.get("/api/providers")
async def list_llm_providers():
    """List LLM providers (with cached models)."""
    from database.provider_crud import list_providers
    async with get_db() as db:
        providers = await list_providers(db)
    return {"providers": providers}


@app.post("/api/providers")
async def add_llm_provider(request: Request):
    """Add a provider and auto-fetch its models from /v1/models."""
    from database.provider_crud import create_provider, get_provider_by_name
    from tools.provider_service import fetch_models
    data = await request.json()
    name = (data.get("name") or "").strip()
    base_url = (data.get("base_url") or "").strip().rstrip("/")
    api_key = (data.get("api_key") or "").strip() or None
    if not name or not base_url:
        raise HTTPException(status_code=400, detail="name and base_url are required")
    async with get_db() as db:
        existing = await get_provider_by_name(db, name)
        if existing:
            raise HTTPException(status_code=409, detail=f"Provider '{name}' already exists")
        # First provider becomes the default.
        from database.provider_crud import list_providers as _lp
        existing_count = len(await _lp(db))
        try:
            models = await fetch_models(base_url, api_key)
            error = None
        except Exception as e:
            models = []
            error = str(e)[:300]
        provider = await create_provider(
            db, name, base_url, api_key=api_key, models=models,
            is_default=1 if existing_count == 0 else 0
        )
        await db.commit()
    return {"provider": provider, "models_fetched": len(models), "error": error}


@app.post("/api/providers/{provider_id}/refresh")
async def refresh_llm_provider_models(provider_id: str):
    """Re-fetch models from a provider."""
    from database.provider_crud import get_provider, update_provider
    from tools.provider_service import fetch_models
    async with get_db() as db:
        provider = await get_provider(db, provider_id, include_api_key=True)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        try:
            models = await fetch_models(provider["base_url"], provider.get("api_key"))
            error = None
        except Exception as e:
            models = provider.get("models") or []
            error = str(e)[:300]
        updated = await update_provider(db, provider_id, models=models)
        await db.commit()
    return {"provider": updated, "models_fetched": len(models), "error": error}


@app.post("/api/providers/{provider_id}/default")
async def set_default_llm_provider(provider_id: str):
    """Make a provider the default for conversations/agents without one."""
    from database.provider_crud import set_default_provider
    async with get_db() as db:
        provider = await set_default_provider(db, provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        await db.commit()
    return {"provider": provider}


@app.put("/api/providers/{provider_id}")
async def update_llm_provider(provider_id: str, request: Request):
    """Update provider details and re-fetch models."""
    from database.provider_crud import get_provider, update_provider
    from tools.provider_service import fetch_models
    data = await request.json()
    async with get_db() as db:
        provider = await get_provider(db, provider_id, include_api_key=True)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        fields = {}
        if "name" in data and str(data.get("name") or "").strip():
            fields["name"] = str(data["name"]).strip()
        if "base_url" in data and str(data.get("base_url") or "").strip():
            fields["base_url"] = str(data["base_url"]).strip().rstrip("/")
        if "api_key" in data:
            fields["api_key"] = str(data["api_key"]).strip() or None
        if "enabled" in data:
            fields["enabled"] = 1 if data["enabled"] else 0
        if fields.get("base_url") and fields["base_url"] != provider["base_url"]:
            try:
                fields["models"] = await fetch_models(fields["base_url"], fields.get("api_key", provider.get("api_key")))
            except Exception as e:
                fields["models"] = []
                fields["fetch_error"] = str(e)[:300]
        updated = await update_provider(db, provider_id, **fields)
        await db.commit()
    return {"provider": updated}


@app.delete("/api/providers/{provider_id}")
async def delete_llm_provider(provider_id: str):
    """Delete a provider."""
    from database.provider_crud import delete_provider, get_default_provider, set_default_provider, list_providers
    async with get_db() as db:
        provider = await delete_provider(db, provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        # If we deleted the default, promote the first remaining provider.
        remaining = await list_providers(db)
        if remaining and not any(p.get("is_default") for p in remaining):
            await set_default_provider(db, remaining[0]["id"])
        await db.commit()
    return {"status": "ok"}


# Terminal Endpoints
@app.get("/api/terminal/blocked-patterns")
async def get_terminal_blocked_patterns():
    """List the hard-coded dangerous command patterns (read-only, cannot be disabled)."""
    from tools.terminal_tool import HARD_BLOCKED_PATTERNS
    return {"patterns": HARD_BLOCKED_PATTERNS}


# Jobs Endpoints (agent platform Phase 5)
@app.get("/api/jobs")
async def list_jobs_api(limit: int = 50):
    """List recent job runs."""
    from database.job_crud import list_job_runs
    async with get_db() as db:
        runs = await list_job_runs(db, limit=limit)
    return {"runs": runs}


async def _pick_jobs_model() -> Optional[str]:
    """jobs_model setting, else the default provider's first loaded model."""
    from settings import settings_manager
    m = (settings_manager.get_settings().get("jobs_model") or "").strip()
    if m:
        return m
    try:
        from database.provider_crud import get_default_provider
        async with get_db() as db:
            provider = await get_default_provider(db)
        if not provider:
            return None
        cached = provider.get("models") or []
        base = provider.get("base_url")
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{base}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    for mod in data.get("data", []):
                        if mod.get("status", {}).get("value") == "loaded":
                            return mod["id"]
        if cached:
            return cached[0]["id"]
    except Exception as e:
        print(f"[JOBS] model pick failed: {e}")
    return None


@app.post("/api/jobs/run")
async def run_job_now(request: Request):
    """Run an on-demand job to completion (inline) and write its output.

    Body: {job: <skill name>, params?: {...}, agent_id?: int}
    Returns the job run record. This endpoint is the future cron hook —
    jobs only run while the app process is up.
    """
    import uuid as _uuid
    from database.job_crud import create_job_run, finish_job_run, get_job_run
    from database.crud import create_conversation, add_message
    from tools.skills_tool import get_skill
    data = await request.json()
    job_name = (data.get("job") or data.get("name") or "").strip()
    params = data.get("params") or {}
    agent_id = data.get("agent_id")

    skill = get_skill(job_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Job/skill '{job_name}' not found")

    model = None
    if agent_id is not None:
        async with get_db() as db:
            agent = await get_agent(db, agent_id)
            if agent:
                model = agent.model
    if not model:
        model = await _pick_jobs_model()
    if not model:
        raise HTTPException(status_code=500, detail="No model available for the job run")

    async with get_db() as db:
        conversation = await create_conversation(db, title=f"Job: {job_name}", agent_id=agent_id)
        conversation_id = conversation["id"]
        run = await create_job_run(db, job_name, params=params, conversation_id=conversation_id)
        user_message = (
            f"Run the job '{job_name}' now. Follow the job instructions exactly "
            f"and deliver the output. Job parameters: {json.dumps(params)}"
        )
        await add_message(db, conversation_id, "user", user_message)
        await db.commit()

    request_id = str(_uuid.uuid4())
    assistant_parts = []
    try:
        async for event in _core_stream_handler(request_id, conversation_id, model=model):
            line = event.strip()
            if line.startswith("data: "):
                try:
                    ev = json.loads(line[6:])
                    if ev.get("type") == "content":
                        assistant_parts.append(ev.get("content", ""))
                except Exception:
                    pass
    except Exception as e:
        print(f"[JOBS] run failed: {e}")
        async with get_db() as db:
            await finish_job_run(db, run["id"], "failed", error=str(e))
            await db.commit()
        raise HTTPException(status_code=500, detail=f"Job run failed: {e}")

    assistant_text = "".join(assistant_parts).strip()
    output_path = None
    async with get_db() as db:
        if assistant_text:
            import os as _os
            from settings import OUTPUTS_DIR
            jobs_dir = _os.path.join(OUTPUTS_DIR, "jobs")
            _os.makedirs(jobs_dir, exist_ok=True)
            output_path = _os.path.join(jobs_dir, f"{run['id']}.md")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"# Job: {job_name}\n\n{assistant_text}")
            except Exception as e:
                print(f"[JOBS] output write failed: {e}")
                output_path = None
            await finish_job_run(db, run["id"], "completed", output_path=output_path)
        else:
            await finish_job_run(db, run["id"], "failed", error="No assistant output produced")
        await db.commit()
        final = await get_job_run(db, run["id"])
    return {"run": final, "output": assistant_text[:2000]}


# Tool Approval Endpoints
@app.post("/api/tools/{request_id}/approve")
async def approve_tool_request(request_id: str, payload: dict = None):
    """Approve or deny a pending tool execution (e.g. a run_command request).

    The terminal tool yields a `tool_approval_required` SSE event and pauses;
    the frontend calls this endpoint to resolve it.
    """
    from tools.terminal_tool import approval_manager
    payload = payload or {}
    approval_key = payload.get("approval_key") or f"{request_id}:0"
    approved = bool(payload.get("decision", True))
    if not approval_manager.decide(approval_key, approved):
        raise HTTPException(status_code=404, detail="No pending approval found for this request")
    return {"status": "ok", "decision": approved}


# STT Endpoints
@app.post("/api/stt/transcribe")
async def transcribe_audio(request: Request):
    """
    Transcribe audio to text using STT.
    Accepts multipart/form-data with an audio file.
    """
    try:
        form = await request.form()
        audio_file = form.get("audio")
        if not audio_file:
            raise HTTPException(status_code=400, detail="No audio file provided")

        audio_data = await audio_file.read()
        filename = getattr(audio_file, "filename", "recording.webm") or "recording.webm"

        result = await stt_service.transcribe(audio_data, filename=filename)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Transcription failed"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")


@app.get("/api/stt/status")
async def stt_status():
    """Check STT availability and current config"""
    status = await stt_service.check_availability()
    status["config"] = {
        "engine": stt_service.config.engine,
        "model": stt_service.config.model,
        "language": stt_service.config.language,
    }
    return status


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
    await backup_scheduler.restart()
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

    # Update STT service config if STT settings changed
    stt_keys = {'stt_engine', 'stt_model', 'stt_language', 'stt_openai_api_key'}
    if stt_keys & set(data.keys()):
        global stt_service
        stt_service = STTService(STTConfig.from_settings(updated_settings))
        print(f"Updated STT service: engine={stt_service.config.engine}, model={stt_service.config.model}")

    return updated_settings


@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated TTS audio files"""
    import mimetypes
    audio_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    mime_type, _ = mimetypes.guess_type(filename)
    media_type = mime_type or "audio/mpeg"

    return FileResponse(
        audio_path,
        media_type=media_type,
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

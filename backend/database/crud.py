from sqlalchemy import select, desc, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import os

from .models import Conversation, Message, MCPServer, Document, Agent, Note


# Conversation operations
async def create_conversation(db: AsyncSession, title: str = "New Chat", agent_id: int = None) -> Dict:
    """Create a new conversation"""
    conversation = Conversation(title=title, agent_id=agent_id)
    db.add(conversation)
    await db.flush()
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "agent_id": conversation.agent_id,
        "tags": conversation.tags or [],
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


async def get_conversation(db: AsyncSession, conversation_id: str) -> Optional[Dict]:
    """Get a conversation by ID"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        return None
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "agent_id": conversation.agent_id,
        "tags": conversation.tags or [],
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


async def get_all_conversations(db: AsyncSession, limit: Optional[int] = None) -> List[Dict]:
    """Get all conversations ordered by most recent (no limit by default;
    all conversations are returned so none are hidden from the UI)."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.agent))
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    conversations = result.scalars().all()
    
    return [
        {
            "id": conv.id,
            "title": conv.title,
            "agent_id": conv.agent_id,
            "agent_name": conv.agent.name if conv.agent else None,
            "tags": conv.tags or [],
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        }
        for conv in conversations
    ]


async def update_conversation_title(db: AsyncSession, conversation_id: str, title: str):
    """Update conversation title"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if conversation:
        conversation.title = title
        conversation.updated_at = datetime.utcnow()


async def delete_conversation(db: AsyncSession, conversation_id: str):
    """Delete a conversation"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if conversation:
        await db.delete(conversation)


# Message operations
async def add_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    blocks: Optional[List[Dict]] = None,
    extra_metadata: Optional[Dict] = None,
    version: int = 1,
    version_group: Optional[str] = None
) -> Dict:
    """Add a message to a conversation.
    
    Args:
        blocks: List of message blocks (content, thinking, tool_call) in sequential order
        extra_metadata: Additional metadata (blocks will be stored here under 'blocks' key)
        version: Message version number (for versioned regeneration)
        version_group: UUID shared by all versions of the same response
    
    For backward compatibility, thinking and tool_calls are also extracted and stored
    in their respective columns.
    """
    # Initialize metadata
    metadata = extra_metadata or {}
    
    # Store blocks in metadata
    if blocks:
        metadata['blocks'] = blocks
    
    # Extract thinking and tool_calls for backward compatibility
    thinking = None
    tool_calls = None
    
    if blocks:
        # Extract thinking from blocks
        thinking_parts = []
        for block in blocks:
            if block.get('type') == 'thinking':
                thinking_parts.append(block.get('content', ''))
        if thinking_parts:
            thinking = '\n'.join(thinking_parts)
        
        # Extract tool_calls from blocks. Keep only lightweight call metadata
        # in the column — the full result/sources/progress payload lives in
        # metadata.blocks (the render source of truth), so storing it twice
        # roughly doubled message size for every tool-using message.
        # Preserve tool_call `id` for KV cache exact replay (prompt must match byte-for-byte).
        tool_call_blocks = [b for b in blocks if b.get('type') == 'tool_call']
        if tool_call_blocks:
            tool_calls = []
            for idx, block in enumerate(tool_call_blocks):
                # Use stored id if present (set during live tool loop), else synthesize deterministically
                tid = block.get('id') or block.get('tool_call_id') or f"{block.get('name','tool')}_{idx}"
                tool_calls.append({
                    'id': tid,
                    'name': block.get('name'),
                    'arguments': block.get('arguments', {}),
                    'status': block.get('status', 'completed'),
                    'progress': block.get('progress', 100),
                })
    
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
        extra_metadata=metadata,
        version=version,
        version_group=version_group
    )
    db.add(message)

    # Update conversation's updated_at timestamp
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation:
        conversation.updated_at = datetime.utcnow()

    await db.flush()

    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "tool_calls": message.tool_calls,
        "thinking": message.thinking,
        "metadata": message.extra_metadata,
        "blocks": message.extra_metadata.get('blocks') if message.extra_metadata else None,
        "version": message.version,
        "version_group": message.version_group,
        "created_at": message.created_at.isoformat(),
    }


async def get_conversation_messages(
    db: AsyncSession,
    conversation_id: str,
    only_latest_versions: bool = True
) -> List[Dict]:
    """Get all messages for a conversation.
    
    Args:
        only_latest_versions: If True, filters to show only the latest version
                              of each version_group. Messages without a version_group
                              are always included.
    
    Returns messages with blocks from metadata for sequential rendering.
    """
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    
    # Filter to keep only the latest version of each version_group
    if only_latest_versions:
        filtered = []
        seen_groups = {}  # version_group -> (index_in_filtered, version)
        for msg in messages:
            vg = msg.version_group
            if vg is None:
                # No version_group (legacy message or single-version) — always include
                filtered.append(msg)
            elif vg in seen_groups:
                idx, existing_version = seen_groups[vg]
                if msg.version > existing_version:
                    # Replace with newer version
                    filtered[idx] = msg
                    seen_groups[vg] = (idx, msg.version)
            else:
                seen_groups[vg] = (len(filtered), msg.version)
                filtered.append(msg)
        messages = filtered

    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "tool_calls": msg.tool_calls,
            "thinking": msg.thinking,
            "metadata": msg.extra_metadata,
            "blocks": msg.extra_metadata.get('blocks') if msg.extra_metadata else None,
            "version": msg.version,
            "version_group": msg.version_group,
            "created_at": msg.created_at.isoformat(),
        }
        for msg in messages
    ]


async def get_message_versions(db: AsyncSession, message_id: str) -> List[Dict]:
    """Get all versions of a message by finding its version_group.
    Returns empty list if the message has no version_group.
    """
    # First get the message to find its version_group
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    msg = result.scalar_one_or_none()
    
    if not msg or not msg.version_group:
        return []
    
    # Get all messages in the same version_group
    result = await db.execute(
        select(Message)
        .where(
            Message.version_group == msg.version_group,
            Message.conversation_id == msg.conversation_id
        )
        .order_by(Message.version)
    )
    versions = result.scalars().all()
    
    return [
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
        for v in versions
    ]


# MCP Server operations
async def add_mcp_server(
    db: AsyncSession,
    name: str,
    command: str,
    args: List[str],
    env: Dict,
    transport_type: str = "stdio",
    url: Optional[str] = None,
    timeout: float = 60.0,
    headers: Optional[Dict] = None
) -> Dict:
    """Add an MCP server configuration. Updates if exists."""
    # Check if server already exists
    result = await db.execute(
        select(MCPServer).where(MCPServer.name == name)
    )
    server = result.scalar_one_or_none()

    if server:
        # Update existing server
        server.transport_type = transport_type
        server.command = command
        server.args = args
        server.env = env
        server.url = url
        server.timeout = timeout
        server.headers = headers or {}
        server.enabled = 1  # Re-enable if it was disabled
    else:
        # Create new server
        server = MCPServer(
            name=name,
            transport_type=transport_type,
            command=command,
            args=args,
            env=env,
            url=url,
            timeout=timeout,
            headers=headers or {}
        )
        db.add(server)

    await db.flush()

    return {
        "id": server.id,
        "name": server.name,
        "transport_type": server.transport_type,
        "command": server.command,
        "args": server.args,
        "env": server.env,
        "url": server.url,
        "headers": server.headers or {},
        "enabled": bool(server.enabled),
        "disabled_tools": server.disabled_tools or [],
        "timeout": server.timeout,
    }


async def get_all_mcp_servers(db: AsyncSession) -> List[Dict]:
    """Get all MCP servers (both enabled and disabled)"""
    result = await db.execute(
        select(MCPServer)
    )
    servers = result.scalars().all()

    return [
        {
            "id": server.id,
            "name": server.name,
            "transport_type": server.transport_type,
            "command": server.command,
            "args": server.args,
            "env": server.env,
            "url": server.url,
            "headers": server.headers or {},
            "enabled": bool(server.enabled),
            "disabled_tools": server.disabled_tools or [],
            "timeout": server.timeout,
        }
        for server in servers
    ]


async def get_enabled_mcp_servers(db: AsyncSession) -> List[Dict]:
    """Get all enabled MCP servers"""
    result = await db.execute(
        select(MCPServer).where(MCPServer.enabled == 1)
    )
    servers = result.scalars().all()

    return [
        {
            "id": server.id,
            "name": server.name,
            "transport_type": server.transport_type,
            "command": server.command,
            "args": server.args,
            "env": server.env,
            "url": server.url,
            "headers": server.headers or {},
            "disabled_tools": server.disabled_tools or [],
            "timeout": server.timeout,
        }
        for server in servers
    ]


async def update_mcp_server_disabled_tools(db: AsyncSession, server_name: str, tool_name: str, disabled: bool) -> bool:
    """Toggle a tool's disabled state for a server."""
    result = await db.execute(
        select(MCPServer).where(MCPServer.name == server_name)
    )
    server = result.scalar_one_or_none()
    
    if not server:
        return False
    
    current = set(server.disabled_tools or [])
    if disabled:
        current.add(tool_name)
    else:
        current.discard(tool_name)
    server.disabled_tools = list(current)
    return True


async def toggle_mcp_server(db: AsyncSession, server_name: str, enabled: bool):
    """Enable or disable an MCP server"""
    result = await db.execute(
        select(MCPServer).where(MCPServer.name == server_name)
    )
    server = result.scalar_one_or_none()
    
    if server:
        server.enabled = 1 if enabled else 0


async def remove_mcp_server(db: AsyncSession, server_name: str) -> bool:
    """Remove an MCP server"""
    result = await db.execute(
        select(MCPServer).where(MCPServer.name == server_name)
    )
    server = result.scalar_one_or_none()
    
    if server:
        await db.delete(server)
        return True
    return False


# Document operations
async def create_document(
    db: AsyncSession,
    filename: str,
    filepath: str,
    file_type: str,
    size_bytes: int,
    metadata: Optional[Dict] = None
) -> Dict:
    """Create a document record"""
    document = Document(
        filename=filename,
        filepath=filepath,
        file_type=file_type,
        size_bytes=size_bytes,
        metadata=metadata
    )
    db.add(document)
    await db.flush()
    
    return {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "size_bytes": document.size_bytes,
        "status": document.status,
        "uploaded_at": document.uploaded_at.isoformat(),
    }


async def update_document_status(
    db: AsyncSession,
    document_id: str,
    status: str,
    metadata: Optional[Dict] = None
):
    """Update document processing status"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if document:
        document.status = status
        if metadata:
            document.metadata = metadata
        if status == "completed":
            document.processed_at = datetime.utcnow()


async def get_documents(db: AsyncSession, limit: int = 50) -> List[Dict]:
    """Get all documents"""
    result = await db.execute(
        select(Document)
        .order_by(desc(Document.uploaded_at))
        .limit(limit)
    )
    documents = result.scalars().all()
    
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "size_bytes": doc.size_bytes,
            "status": doc.status,
            "uploaded_at": doc.uploaded_at.isoformat(),
        }
        for doc in documents
    ]


async def get_message(db: AsyncSession, message_id: str) -> Optional[Dict]:
    """Get a message by ID"""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()

    if not message:
        return None

    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "tool_calls": message.tool_calls,
        "thinking": message.thinking,
        "metadata": message.extra_metadata,
        "version": message.version,
        "version_group": message.version_group,
        "created_at": message.created_at.isoformat(),
    }


async def update_message(db: AsyncSession, message_id: str, content: str) -> Optional[Dict]:
    """Update a message's content"""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()

    if not message:
        return None

    message.content = content
    await db.flush()

    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "tool_calls": message.tool_calls,
        "thinking": message.thinking,
        "metadata": message.extra_metadata,
        "created_at": message.created_at.isoformat(),
    }


async def delete_message(db: AsyncSession, message_id: str) -> bool:
    """Delete a message"""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()
    
    if not message:
        return False
    
    await db.delete(message)
    return True


async def get_document(db: AsyncSession, document_id: str) -> Optional[Dict]:
    """Get a document by ID"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        return None
    
    return {
        "id": document.id,
        "filename": document.filename,
        "filepath": document.filepath,
        "file_type": document.file_type,
        "size_bytes": document.size_bytes,
        "status": document.status,
        "metadata": document.metadata,
        "uploaded_at": document.uploaded_at.isoformat(),
        "processed_at": document.processed_at.isoformat() if document.processed_at else None,
    }


async def delete_document(db: AsyncSession, document_id: str) -> bool:
    """Delete a document"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        return False
    
    # Delete the file from filesystem
    if document.filepath and os.path.exists(document.filepath):
        try:
            os.remove(document.filepath)
        except Exception as e:
            print(f"Error deleting file: {e}")
    
    await db.delete(document)
    return True


# ============================================================================
# Agent operations
# ============================================================================

async def create_agent(db: AsyncSession, agent_data: Dict[str, Any]) -> Agent:
    """Create a new agent"""
    agent = Agent(**agent_data)
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return agent


async def get_agent(db: AsyncSession, agent_id: int) -> Optional[Agent]:
    """Get agent by ID"""
    result = await db.execute(
        select(Agent)
        .options(selectinload(Agent.conversations))
        .where(Agent.id == agent_id)
    )
    return result.scalar_one_or_none()


async def get_agent_by_name(db: AsyncSession, name: str) -> Optional[Agent]:
    """Get agent by name"""
    result = await db.execute(
        select(Agent)
        .options(selectinload(Agent.conversations))
        .where(Agent.name == name)
    )
    return result.scalar_one_or_none()


async def get_all_agents(db: AsyncSession, active_only: bool = True) -> List[Agent]:
    """Get all agents"""
    query = select(Agent).options(selectinload(Agent.conversations))
    if active_only:
        query = query.where(Agent.is_active == 1)
    query = query.order_by(Agent.created_at.desc())
    result = await db.execute(query)
    return result.scalars().all()


async def update_agent(db: AsyncSession, agent_id: int, agent_data: Dict[str, Any]) -> Optional[Agent]:
    """Update an existing agent"""
    agent = await get_agent(db, agent_id)
    if not agent:
        return None
    
    for key, value in agent_data.items():
        if hasattr(agent, key):
            setattr(agent, key, value)
    
    agent.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(agent)
    return agent


async def delete_agent(db: AsyncSession, agent_id: int) -> bool:
    """Delete an agent (soft delete by setting is_active=False)"""
    agent = await get_agent(db, agent_id)
    if not agent:
        return False
    
    agent.is_active = 0
    agent.updated_at = datetime.utcnow()
    await db.commit()
    return True


async def update_conversation_tags(db: AsyncSession, conversation_id: str, tags: List[str]) -> Optional[Dict]:
    """Update tags for a conversation"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        return None
    
    conversation.tags = tags
    conversation.updated_at = datetime.utcnow()
    await db.flush()
    
    return {
        "id": conversation.id,
        "tags": conversation.tags or [],
    }


async def update_conversation_agent(db: AsyncSession, conversation_id: str, agent_id: Optional[int]) -> Optional[Dict]:
    """Update the agent associated with a conversation"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        return None
    
    conversation.agent_id = agent_id
    conversation.updated_at = datetime.utcnow()
    await db.flush()
    
    return {
        "id": conversation.id,
        "agent_id": conversation.agent_id,
    }


# Note operations

async def create_note(
    db: AsyncSession,
    conversation_id: str,
    message_id: Optional[str],
    content: str,
    source_text: Optional[str] = None
) -> Dict:
    """Create a note for a conversation/message"""
    note = Note(
        conversation_id=conversation_id,
        message_id=message_id,
        content=content,
        source_text=source_text
    )
    db.add(note)
    await db.flush()
    
    return {
        "id": note.id,
        "conversation_id": note.conversation_id,
        "message_id": note.message_id,
        "content": note.content,
        "source_text": note.source_text,
        "created_at": note.created_at.isoformat(),
    }


async def get_all_notes(db: AsyncSession, limit: int = 100) -> List[Dict]:
    """Get all notes ordered by most recent"""
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Note)
        .options(selectinload(Note.conversation))
        .order_by(desc(Note.created_at))
        .limit(limit)
    )
    notes = result.scalars().all()
    
    return [
        {
            "id": note.id,
            "conversation_id": note.conversation_id,
            "message_id": note.message_id,
            "content": note.content,
            "source_text": note.source_text,
            "conversation_title": note.conversation.title if note.conversation else "Unknown",
            "created_at": note.created_at.isoformat(),
        }
        for note in notes
    ]


async def delete_note(db: AsyncSession, note_id: str) -> bool:
    """Delete a note"""
    result = await db.execute(
        select(Note).where(Note.id == note_id)
    )
    note = result.scalar_one_or_none()
    
    if not note:
        return False
    
    await db.delete(note)
    return True


async def get_notes_for_conversation(db: AsyncSession, conversation_id: str) -> List[Dict]:
    """Get all notes for a conversation"""
    result = await db.execute(
        select(Note)
        .where(Note.conversation_id == conversation_id)
        .order_by(desc(Note.created_at))
    )
    notes = result.scalars().all()
    
    return [
        {
            "id": note.id,
            "conversation_id": note.conversation_id,
            "message_id": note.message_id,
            "content": note.content,
            "source_text": note.source_text,
            "created_at": note.created_at.isoformat(),
        }
        for note in notes
    ]




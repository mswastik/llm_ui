from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime
from typing import List, Optional, Dict
import os

from .models import Conversation, Message, MCPServer, Document


# Conversation operations
async def create_conversation(db: AsyncSession, title: str = "New Chat") -> Dict:
    """Create a new conversation"""
    conversation = Conversation(title=title)
    db.add(conversation)
    await db.flush()
    
    return {
        "id": conversation.id,
        "title": conversation.title,
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
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


async def get_all_conversations(db: AsyncSession, limit: int = 50) -> List[Dict]:
    """Get all conversations ordered by most recent"""
    result = await db.execute(
        select(Conversation)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    conversations = result.scalars().all()
    
    return [
        {
            "id": conv.id,
            "title": conv.title,
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
    tool_calls: Optional[List] = None,
    thinking: Optional[str] = None,
    extra_metadata: Optional[Dict] = None
) -> Dict:
    """Add a message to a conversation"""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
        extra_metadata=extra_metadata
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
        "created_at": message.created_at.isoformat(),
    }


async def get_conversation_messages(
    db: AsyncSession,
    conversation_id: str
) -> List[Dict]:
    """Get all messages for a conversation"""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "tool_calls": msg.tool_calls,
            "thinking": msg.thinking,
            "metadata": msg.extra_metadata,
            "created_at": msg.created_at.isoformat(),
        }
        for msg in messages
    ]


# MCP Server operations
async def add_mcp_server(
    db: AsyncSession,
    name: str,
    command: str,
    args: List[str],
    env: Dict
) -> Dict:
    """Add an MCP server configuration"""
    server = MCPServer(
        name=name,
        command=command,
        args=args,
        env=env
    )
    db.add(server)
    await db.flush()
    
    return {
        "id": server.id,
        "name": server.name,
        "command": server.command,
        "args": server.args,
        "env": server.env,
        "enabled": bool(server.enabled),
    }


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
            "command": server.command,
            "args": server.args,
            "env": server.env,
        }
        for server in servers
    ]


async def toggle_mcp_server(db: AsyncSession, server_name: str, enabled: bool):
    """Enable or disable an MCP server"""
    result = await db.execute(
        select(MCPServer).where(MCPServer.name == server_name)
    )
    server = result.scalar_one_or_none()
    
    if server:
        server.enabled = 1 if enabled else 0


async def remove_mcp_server(db: AsyncSession, server_name: str):
    """Remove an MCP server"""
    result = await db.execute(
        select(MCPServer).where(MCPServer.name == server_name)
    )
    server = result.scalar_one_or_none()
    
    if server:
        await db.delete(server)


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

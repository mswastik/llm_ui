from sqlalchemy import Column, String, Text, DateTime, Integer, Float, JSON, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import logging

from settings import DATABASE_URL, SQLALCHEMY_ECHO

# Suppress verbose SQLAlchemy logs
if not SQLALCHEMY_ECHO:
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

Base = declarative_base()

# Database URL from config
engine = create_async_engine(DATABASE_URL, echo=SQLALCHEMY_ECHO)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, default="New Chat")
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    # Relationship to agent
    agent = relationship("Agent", back_populates="conversations")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Store thinking content from reasoning models (e.g., DeepSeek)
    thinking = Column(Text, nullable=True)

    # Store tool calls and results if any
    tool_calls = Column(JSON, nullable=True)

    # Additional metadata for the message
    extra_metadata = Column(JSON, nullable=True, name="metadata")

    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")


class MCPServer(Base):
    __tablename__ = "mcp_servers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    
    # Transport type: 'stdio', 'sse', or 'streamable-http'
    transport_type = Column(String, default="stdio", nullable=False)
    
    # For stdio transport
    command = Column(String, nullable=True)  # Optional for URL-based transports
    args = Column(JSON, default=list)
    env = Column(JSON, default=dict)
    
    # For SSE and StreamableHTTP transports
    url = Column(String, nullable=True)
    
    enabled = Column(Integer, default=1)  # SQLite doesn't have native boolean
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)

    # Processing status
    status = Column(String, default="pending")  # pending, processing, completed, failed

    # Metadata extracted from document
    _metadata = Column(JSON, nullable=True)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    
    # LLM Configuration
    model = Column(String(255), nullable=False)
    temperature = Column(Float, default=0.7)
    top_k = Column(Integer, default=40)
    max_tokens = Column(Integer, default=16048)
    
    # System prompt
    system_prompt = Column(Text, nullable=True)
    
    # Tools configuration
    enabled_tools = Column(JSON, default=list)  # List of tool names
    enabled_mcp_servers = Column(JSON, default=list)  # List of MCP server names
    
    # RAG configuration
    enable_rag = Column(Integer, default=0)  # SQLite doesn't have native boolean
    rag_similarity_threshold = Column(Float, default=0.4)
    
    # Web search configuration
    enable_web_search = Column(Integer, default=0)
    
    # Default conversation starters
    conversation_starters = Column(JSON, default=list)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan", lazy="selectin")


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_db():
    """Get database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

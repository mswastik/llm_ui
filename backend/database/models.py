from sqlalchemy import Column, String, Text, DateTime, Integer, Float, JSON, ForeignKey
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import uuid
import logging
import asyncio

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
    tags = Column(JSON, default=list)  # Array of tag strings
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

    # Versioning for regeneration
    version = Column(Integer, default=1)
    version_group = Column(String, nullable=True)  # UUID shared by all versions of the same response

    # Stable timeline slot. Every version of a regenerated response inherits the
    # original message's turn_index so mid-thread regenerations render in place;
    # normal messages get max(turn_index) + 1000 at insert time.
    turn_index = Column(Float, nullable=True)

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
    headers = Column(JSON, default=dict)  # Custom HTTP headers (e.g. Authorization) for SSE/HTTP transports
    
    enabled = Column(Integer, default=1)  # SQLite doesn't have native boolean
    disabled_tools = Column(JSON, default=list)  # List of tool names disabled for this server
    timeout = Column(Float, default=60.0)  # Connection timeout in seconds
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


class Note(Base):
    __tablename__ = "notes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    message_id = Column(String, ForeignKey("messages.id"), nullable=True)
    content = Column(Text, nullable=False)
    source_text = Column(Text, nullable=True)  # The original LLM text that the note was taken from
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation")
    message = relationship("Message")


class MemoryEntry(Base):
    """Persistent agent memory (agent platform, Phase 2).

    scope: 'global' | 'agent:<id>' | 'conversation:<id>'
    source: 'manual' (user/LLM explicit) | 'auto' (extracted by the app)
    """
    __tablename__ = "memory_entries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scope = Column(String, default="global", nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSON, default=list)
    source = Column(String, default="manual", nullable=False)
    importance = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SkillRun(Base):
    """Skill usage log (agent platform, Phase 4) — feeds the improvement loop."""
    __tablename__ = "skill_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    skill_name = Column(String, nullable=False)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True)
    success = Column(Integer, default=1)
    user_correction = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class JobRun(Base):
    """On-demand job execution record (agent platform, Phase 5).

    A job is a skill with an input/output contract; run_job tracks one
    execution here so the Jobs UI can show history and outputs.
    """
    __tablename__ = "job_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_name = Column(String, nullable=False)
    params = Column(JSON, default=dict)
    status = Column(String, default="running")  # running | completed | failed
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    output_path = Column(String, nullable=True)
    conversation_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)


class LLMProvider(Base):
    """LLM provider (OpenAI-compatible endpoint) for multi-provider support.

    Models are fetched from the provider's /v1/models endpoint when the
    provider is added/refreshed and cached in the `models` JSON column.
    """
    __tablename__ = "llm_providers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    base_url = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    models = Column(JSON, default=list)  # [{id, name, owned_by}]
    enabled = Column(Integer, default=1)
    is_default = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    
    # LLM Configuration
    model = Column(String(255), nullable=False)
    provider_id = Column(String, nullable=True)  # LLMProvider id (multi-provider support)
    temperature = Column(Float, default=0.7)
    top_k = Column(Integer, default=40)
    max_tokens = Column(Integer, default=16048)
    
    # System prompt
    system_prompt = Column(Text, nullable=True)
    
    # Tools configuration
    enabled_tools = Column(JSON, default=list)  # List of tool names
    enabled_mcp_servers = Column(JSON, default=list)  # List of MCP server names
    enabled_skills = Column(JSON, default=list)  # List of skill names (empty = all)
    
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
    """Initialize database tables and run migrations"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ── FTS5 for memory search (VRAM-free, no embedding model) ──────────
    # Replaces the embedding-based semantic search for memory. FTS5 gives
    # BM25-ranked full-text search without loading Qwen3-4B-Embedding into
    # VRAM (which evicts the 35B main model on limited VRAM and forces a
    # full 27k prompt re-process next turn).
    try:
        from sqlalchemy import text as _text
        import json as _json
        async with engine.begin() as conn:
            await conn.execute(_text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
                "USING fts5(id UNINDEXED, content, tags, scope UNINDEXED, tokenize='porter unicode61')"
            ))
            # Backfill existing rows if FTS is empty (e.g. upgrade from pre-FTS install)
            cnt = (await conn.execute(_text("SELECT count(*) FROM memory_fts"))).scalar() or 0
            if cnt == 0:
                rows = (await conn.execute(_text("SELECT id, content, tags, scope FROM memory_entries"))).fetchall()
                for r in rows:
                    try:
                        _tags = r[2]
                        if isinstance(_tags, str):
                            _tags = _json.loads(_tags)
                        _tags_str = " ".join(t for t in (_tags or []) if isinstance(t, str))
                    except Exception:
                        _tags_str = ""
                    await conn.execute(_text(
                        "INSERT OR IGNORE INTO memory_fts (id, content, tags, scope) VALUES (:id, :content, :tags, :scope)"
                    ), {"id": r[0], "content": r[1] or "", "tags": _tags_str, "scope": r[3] or "global"})
                if rows:
                    print(f"[DB] Backfilled memory_fts with {len(rows)} rows")
    except Exception as e:
        print(f"[DB] memory_fts init failed (non-fatal, falls back to LIKE): {e}")

    # ── Migration: add disabled_tools column to mcp_servers ──────────────
    # SQLite's create_all does not alter existing tables, so new columns on
    # an already-created table must be added via ALTER TABLE.
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE mcp_servers ADD COLUMN disabled_tools JSON DEFAULT '[]'"
                )
            )
        print("[DB] Migrated mcp_servers: added disabled_tools column")
    except OperationalError:
        # Column already exists — that's fine
        pass

    # ── Migration: add headers column to mcp_servers ──────────────────────
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE mcp_servers ADD COLUMN headers JSON DEFAULT '{}'"
                )
            )
        print("[DB] Migrated mcp_servers: added headers column")
    except OperationalError:
        pass

    # ── Migration: add provider_id column to agents ────────────────────
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE agents ADD COLUMN provider_id VARCHAR"
                )
            )
        print("[DB] Migrated agents: added provider_id column")
    except OperationalError:
        pass

    # ── Migration: add enabled_skills column to agents ────────────────
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE agents ADD COLUMN enabled_skills JSON DEFAULT '[]'"
                )
            )
        print("[DB] Migrated agents: added enabled_skills column")
    except OperationalError:
        pass

    # ── Migration: add version column to messages (if not present) ─────
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE messages ADD COLUMN version INTEGER DEFAULT 1"
                )
            )
        print("[DB] Migrated messages: added version column")
    except OperationalError:
        pass

    # ── Migration: add turn_index column to messages ───────────────────
    # Stable timeline slot: versions of a regenerated answer share the slot,
    # so history ordering is independent of when a version was created.
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE messages ADD COLUMN turn_index FLOAT"
                )
            )
        print("[DB] Migrated messages: added turn_index column")
    except OperationalError:
        pass

    # Backfill legacy rows: one monotonic slot per conversation in creation order.
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql("""
                    WITH ranked AS (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY conversation_id ORDER BY created_at, rowid
                        ) AS rn
                        FROM messages WHERE turn_index IS NULL
                    )
                    UPDATE messages SET turn_index = (
                        SELECT rn * 1000.0 FROM ranked WHERE ranked.id = messages.id
                    )
                    WHERE turn_index IS NULL AND id IN (SELECT id FROM ranked)
                """)
            )
        print("[DB] Backfilled messages.turn_index for legacy rows")
    except Exception as e:
        print(f"[DB] turn_index backfill failed (non-fatal): {e}")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.exec_driver_sql(
                    "ALTER TABLE messages ADD COLUMN version_group VARCHAR"
                )
            )
        print("[DB] Migrated messages: added version_group column")
    except OperationalError:
        pass


async def shutdown_db():
    """Checkpoint pending WAL data into the main DB file and close the pool.

    Ensures llm_ui.db contains every committed write and the -wal/-shm
    files are cleared when the app shuts down cleanly.
    """
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        await engine.dispose()
    except Exception as e:
        print(f"[DB] WAL checkpoint on shutdown failed: {e}")


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
            try:
                await session.close()
            except asyncio.CancelledError:
                # Ignore cancellation during close - connection is being cleaned up anyway
                pass


# ============================================================================
# Suppress "Exception terminating connection" noise from aiosqlite pool
# when client disconnects during SSE streaming.
#
# SQLAlchemy's Pool._close_connection catches BaseException from do_terminate/
# do_close, logs it as "Exception terminating/closing connection", then
# re-raises if it's not an Exception (CancelledError is BaseException-only).
# We patch it to silently ignore CancelledError since it's expected when
# a client disconnects during SSE streaming.
# ============================================================================
import sqlalchemy.pool as _pool

_original_close_connection = _pool.Pool._close_connection

def _suppressed_close_connection(self, connection, *, terminate=False):
    try:
        if terminate:
            self._dialect.do_terminate(connection)
        else:
            self._dialect.do_close(connection)
    except asyncio.CancelledError:
        # Client disconnected during streaming - connection close was cancelled.
        # This is expected and harmless; no need to log or re-raise.
        pass
    except BaseException as e:
        self.logger.error(
            f"Exception {'terminating' if terminate else 'closing'} "
            f"connection %r",
            connection,
            exc_info=True,
        )
        if not isinstance(e, Exception):
            raise

_pool.Pool._close_connection = _suppressed_close_connection

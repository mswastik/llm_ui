"""
Database models for Agent configuration
"""
from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime


class AgentModel:
    """Agent configuration model for database"""
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
    enable_rag = Column(Boolean, default=False)
    rag_similarity_threshold = Column(Float, default=0.4)
    
    # Web search configuration
    enable_web_search = Column(Boolean, default=False)
    
    # Default conversation starters
    conversation_starters = Column(JSON, default=list)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    conversations = relationship("ConversationModel", back_populates="agent", lazy="selectin")


# Schema for API responses
AGENT_SCHEMA = {
    "id": int,
    "name": str,
    "description": str,
    "model": str,
    "temperature": float,
    "top_k": int,
    "max_tokens": int,
    "system_prompt": str,
    "enabled_tools": list,
    "enabled_mcp_servers": list,
    "enable_rag": bool,
    "rag_similarity_threshold": float,
    "enable_web_search": bool,
    "conversation_starters": list,
    "created_at": str,
    "updated_at": str,
    "is_active": bool
}

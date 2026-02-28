"""
Database CRUD operations for Agents
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from datetime import datetime
from typing import List, Optional, Dict, Any
from .models import Agent, Conversation


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


async def hard_delete_agent(db: AsyncSession, agent_id: int) -> bool:
    """Permanently delete an agent"""
    agent = await get_agent(db, agent_id)
    if not agent:
        return False
    
    await db.delete(agent)
    await db.commit()
    return True


async def get_default_agent(db: AsyncSession) -> Optional[Agent]:
    """Get the default agent (first active agent or create one)"""
    agents = await get_all_agents(db)
    if agents:
        return agents[0]
    
    # Create default agent if none exists
    default_data = {
        "name": "Default Assistant",
        "description": "Default AI assistant agent",
        "model": "qwen3-4b",
        "temperature": 0.7,
        "top_k": 40,
        "max_tokens": 16048,
        "system_prompt": "You are a helpful AI assistant. When you use tools, explain what you're doing and why.",
        "enabled_tools": [],
        "enabled_mcp_servers": [],
        "enable_rag": 0,
        "enable_web_search": 0,
        "conversation_starters": [
            "What can you help me with?",
            "Tell me about your capabilities",
            "How do I get started?"
        ]
    }
    return await create_agent(db, default_data)

#!/usr/bin/env python3
"""
Database migration script to add agent_id column to conversations table
"""
import asyncio
import sys
sys.path.insert(0, 'backend')

from sqlalchemy import text
from database.models import engine

async def migrate():
    """Run database migrations"""
    async with engine.begin() as conn:
        # Check if agent_id column already exists
        result = await conn.execute(text("PRAGMA table_info(conversations)"))
        columns = [row[1] for row in result.fetchall()]
        
        if 'agent_id' not in columns:
            # Add agent_id column to conversations table
            await conn.execute(text("""
                ALTER TABLE conversations ADD COLUMN agent_id INTEGER
            """))
            print("[OK] Added agent_id column to conversations table")
        else:
            print("[INFO] agent_id column already exists")
        
        # Create agents table if it doesn't exist
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                model VARCHAR(255) NOT NULL,
                temperature FLOAT DEFAULT 0.7,
                top_k INTEGER DEFAULT 40,
                max_tokens INTEGER DEFAULT 16048,
                system_prompt TEXT,
                enabled_tools JSON DEFAULT '[]',
                enabled_mcp_servers JSON DEFAULT '[]',
                enable_rag INTEGER DEFAULT 0,
                rag_similarity_threshold FLOAT DEFAULT 0.4,
                enable_web_search INTEGER DEFAULT 0,
                conversation_starters JSON DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """))
        print("[OK] Agents table created/verified")
        
        # Create index on agent_id
        try:
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_conversations_agent_id 
                ON conversations(agent_id)
            """))
            print("[OK] Created index on conversations.agent_id")
        except Exception as e:
            print(f"[WARN] Could not create index: {e}")
        
        print("\n[OK] Database migration completed successfully!")

if __name__ == "__main__":
    asyncio.run(migrate())

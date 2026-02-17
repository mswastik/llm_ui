#!/usr/bin/env python3
"""
Database migration script to add new columns for MCP server transport support
"""
import sqlite3
import os

# Get the database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "llm_ui.db")

def migrate():
    """Add new columns for MCP server transport support"""
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if columns already exist in mcp_servers table
    cursor.execute("PRAGMA table_info(mcp_servers)")
    mcp_columns = [column[1] for column in cursor.fetchall()]

    # Add transport_type column
    if "transport_type" in mcp_columns:
        print("Column 'transport_type' already exists in mcp_servers table")
    else:
        print("Adding 'transport_type' column to mcp_servers table...")
        cursor.execute("ALTER TABLE mcp_servers ADD COLUMN transport_type TEXT DEFAULT 'stdio'")
        conn.commit()
        print("Successfully added 'transport_type' column to mcp_servers table")

    # Add url column
    if "url" in mcp_columns:
        print("Column 'url' already exists in mcp_servers table")
    else:
        print("Adding 'url' column to mcp_servers table...")
        cursor.execute("ALTER TABLE mcp_servers ADD COLUMN url TEXT")
        conn.commit()
        print("Successfully added 'url' column to mcp_servers table")

    # Check if metadata column exists in messages table (previous migration)
    cursor.execute("PRAGMA table_info(messages)")
    message_columns = [column[1] for column in cursor.fetchall()]

    if "metadata" in message_columns:
        print("Column 'metadata' already exists in messages table")
    else:
        print("Adding 'metadata' column to messages table...")
        cursor.execute("ALTER TABLE messages ADD COLUMN metadata JSON")
        conn.commit()
        print("Successfully added 'metadata' column to messages table")

    conn.close()
    print("\nMigration completed successfully!")

if __name__ == "__main__":
    migrate()

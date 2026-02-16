#!/usr/bin/env python3
"""
Database migration script to add metadata column to messages table
"""
import sqlite3
import os

# Get the database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "llm_ui.db")

def migrate():
    """Add metadata column to messages table if it doesn't exist"""
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if column already exists
    cursor.execute("PRAGMA table_info(messages)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if "metadata" in columns:
        print("Column 'metadata' already exists in messages table")
    else:
        print("Adding 'metadata' column to messages table...")
        cursor.execute("ALTER TABLE messages ADD COLUMN metadata JSON")
        conn.commit()
        print("Successfully added 'metadata' column to messages table")
    
    conn.close()

if __name__ == "__main__":
    migrate()

"""
Database migration script.
Adds missing columns/tables after schema changes.
"""
import sqlite3
import os

DB_PATH = "llm_ui.db"

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if tags column exists in conversations
cursor.execute("PRAGMA table_info(conversations)")
columns = {row[1] for row in cursor.fetchall()}

print(f"Existing columns: {columns}")

if "tags" not in columns:
    print("Adding 'tags' column to conversations table...")
    cursor.execute("ALTER TABLE conversations ADD COLUMN tags JSON DEFAULT '[]'")
    print("  -> 'tags' column added.")
else:
    print("  -> 'tags' column already exists.")

# Check if notes table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
if not cursor.fetchone():
    print("Creating 'notes' table...")
    cursor.execute("""
        CREATE TABLE notes (
            id VARCHAR NOT NULL PRIMARY KEY,
            conversation_id VARCHAR NOT NULL REFERENCES conversations(id),
            message_id VARCHAR REFERENCES messages(id),
            content TEXT NOT NULL,
            source_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("  -> 'notes' table created.")
else:
    print("  -> 'notes' table already exists.")

conn.commit()
conn.close()
print("\nMigration complete!")

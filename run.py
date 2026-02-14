#!/usr/bin/env python3
"""
Startup script for LLM UI application
"""
import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import uvicorn
from config import APP_HOST, APP_PORT, DEBUG

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting LLM UI Application")
    print("=" * 60)
    print(f"📍 Server: http://{APP_HOST}:{APP_PORT}")
    print(f"🔧 Debug Mode: {DEBUG}")
    print(f"💾 Database: SQLite (llm_ui.db)")
    print("=" * 60)
    print()
    print("📌 Make sure llama.cpp is running on port 8080")
    print("📌 Add MCP servers via the Settings menu in the UI")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug"
    )

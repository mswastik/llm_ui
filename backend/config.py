"""
Configuration settings for the LLM UI application
"""
import os
from typing import Dict, Any

# To fix Kokoro Cuda memory allocation
os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./llm_ui.db")

# llama.cpp Configuration
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "glm4.7-30ba3b")
QUERY_MODEL = os.getenv("QUERY_MODEL", "qwen3-30ba3b") # Query model should be non-thinking model

# Application Settings
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# LLM Generation Defaults
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16048"))

# File Upload Settings
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB default
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CORS Settings (if needed)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# System Prompt (optional - can be customized per conversation)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant. When you use tools, explain what you're doing and why."
)


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        "database_url": DATABASE_URL,
        "llama_cpp_base_url": LLAMA_CPP_BASE_URL,
        "llama_cpp_model": LLAMA_CPP_MODEL,
        "query_model": QUERY_MODEL,
        "app_host": APP_HOST,
        "app_port": APP_PORT,
        "debug": DEBUG,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "max_upload_size": MAX_UPLOAD_SIZE,
        "upload_dir": UPLOAD_DIR,
        "cors_origins": CORS_ORIGINS,
        "system_prompt": SYSTEM_PROMPT,
    }

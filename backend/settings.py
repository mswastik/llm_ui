"""
Application settings management module

Single source of truth for all configuration settings.
Settings are loaded from settings.json on startup, with environment variables as fallback defaults.
"""
import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel

# To fix Kokoro Cuda memory allocation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")

# Default values (used if not in settings.json or environment variables)
DEFAULTS = {
    # Database Configuration
    "database_url": os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./llm_ui.db"),

    # Llama.cpp Configuration
    "llama_cpp_base_url": os.getenv("LLAMA_CPP_URL", "http://localhost:8001/v3"),
    "llama_cpp_model": os.getenv("LLAMA_CPP_MODEL", "qwen3-4b"),
    "query_model": os.getenv("QUERY_MODEL", "qwen3-4b"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "Qwen3-4B-Embedding"),
    "reranking_model": os.getenv("RERANKING_MODEL", "Qwen3-4B-Embedding"),

    # Application Settings
    "app_host": os.getenv("APP_HOST", "0.0.0.0"),
    "app_port": int(os.getenv("APP_PORT", "8002")),
    "debug": os.getenv("DEBUG", "false").lower() == "true",

    # LLM Generation Defaults
    "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
    "default_max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "16048")),

    # File Upload Settings
    "max_upload_size": int(os.getenv("MAX_UPLOAD_SIZE", "10485760")),
    "upload_dir": os.getenv("UPLOAD_DIR", "./uploads"),

    # CORS Settings
    "cors_origins": os.getenv("CORS_ORIGINS", "*"),

    # System Prompt
    "system_prompt": os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful AI assistant. When you use tools, explain what you're doing and why. "
        "You have persistent memory: call memory_search before answering questions about the user's "
        "past work, preferences, projects or people, and call memory_write when the user states a "
        "durable fact or preference. Never store secret values (keys, passwords, tokens) in memory."
    ),

    # SQLAlchemy Logging
    "sqlalchemy_echo": os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true",

    # Database Backup Settings
    "backup_enabled": False,
    "backup_path": "./backups",
    "backup_interval_hours": 4,
    "backup_max_keep": 24,

    # STT Settings
    "stt_engine": "faster-whisper",
    "stt_model": "base",
    "stt_language": None,
    "stt_openai_api_key": None,

    # TTS Settings
    "tts_engine": "edge-tts",
    "tts_voice": "en-IN-NeerjaNeural",
    "tts_rate": "+0%",
    "tts_volume": 1.0,
    "kokoro_lang": "a",
    "kokoro_device": "cpu",
    "kokoro_volume": 1.0,
    "kokoro_speed": 1.0,
    "tts_auto_read": False,

    # Agent Platform: Terminal Tool Settings
    "terminal_allowed_dirs": [".", "./uploads", "./outputs"],
    "terminal_allowed_commands": [
        "python3", "python", "node", "npm", "npx", "git", "curl", "wget",
        "grep", "rg", "find", "sed", "awk", "ls", "cat", "head", "tail",
        "mkdir", "cp", "mv", "rm", "touch", "date", "jq", "echo", "printf",
        "wc", "sort", "uniq", "diff", "patch", "tar", "unzip", "zip",
        "pip", "pip3", "uv", "sqlite3", "env", "pwd", "file", "stat",
        "basename", "dirname", "xargs", "tee", "tr", "cut", "paste"
    ],
    "terminal_blocked_patterns": [],
    "terminal_require_approval": True,
    "terminal_default_timeout": 120,
    "terminal_audit_log": "./terminal_audit.jsonl",

    # Agent Platform: Skills / Jobs / Memory
    "skills_dir": "./skills",
    "outputs_dir": "./outputs",
    "memory_auto_extract_interval": 3,
    "memory_search_use_embedding": False,  # VRAM-safe: keyword search only; embedding would evict main model on limited VRAM
    "jobs_model": "",
}

# Create agent platform directories if they don't exist
os.makedirs(DEFAULTS["skills_dir"], exist_ok=True)
os.makedirs(DEFAULTS["outputs_dir"], exist_ok=True)

# Create upload directory if it doesn't exist
os.makedirs(DEFAULTS["upload_dir"], exist_ok=True)


class Settings(BaseModel):
    """Application settings model"""
    database_url: str = DEFAULTS["database_url"]
    llama_cpp_base_url: str = DEFAULTS["llama_cpp_base_url"]
    llama_cpp_model: str = DEFAULTS["llama_cpp_model"]
    query_model: str = DEFAULTS["query_model"]
    embedding_model: str = DEFAULTS["embedding_model"]
    reranking_model: str = DEFAULTS["reranking_model"]
    app_host: str = DEFAULTS["app_host"]
    app_port: int = DEFAULTS["app_port"]
    debug: bool = DEFAULTS["debug"]
    max_upload_size: int = DEFAULTS["max_upload_size"]
    upload_dir: str = DEFAULTS["upload_dir"]
    cors_origins: str = DEFAULTS["cors_origins"]
    system_prompt: str = DEFAULTS["system_prompt"]
    default_temperature: float = DEFAULTS["default_temperature"]
    default_max_tokens: int = DEFAULTS["default_max_tokens"]
    sqlalchemy_echo: bool = DEFAULTS["sqlalchemy_echo"]
    backup_enabled: bool = DEFAULTS["backup_enabled"]
    backup_path: str = DEFAULTS["backup_path"]
    backup_interval_hours: int = DEFAULTS["backup_interval_hours"]
    backup_max_keep: int = DEFAULTS["backup_max_keep"]
    stt_engine: str = DEFAULTS["stt_engine"]
    stt_model: str = DEFAULTS["stt_model"]
    stt_language: Optional[str] = None
    stt_openai_api_key: Optional[str] = None
    tts_engine: str = DEFAULTS["tts_engine"]
    tts_voice: str = DEFAULTS["tts_voice"]
    tts_rate: str = DEFAULTS["tts_rate"]
    tts_volume: float = DEFAULTS["tts_volume"]
    kokoro_lang: str = DEFAULTS["kokoro_lang"]
    kokoro_device: str = DEFAULTS["kokoro_device"]
    kokoro_volume: float = DEFAULTS["kokoro_volume"]
    kokoro_speed: float = DEFAULTS["kokoro_speed"]
    tts_auto_read: bool = DEFAULTS["tts_auto_read"]
    terminal_allowed_dirs: list = DEFAULTS["terminal_allowed_dirs"]
    terminal_allowed_commands: list = DEFAULTS["terminal_allowed_commands"]
    terminal_blocked_patterns: list = DEFAULTS["terminal_blocked_patterns"]
    terminal_require_approval: bool = DEFAULTS["terminal_require_approval"]
    terminal_default_timeout: int = DEFAULTS["terminal_default_timeout"]
    terminal_audit_log: str = DEFAULTS["terminal_audit_log"]
    skills_dir: str = DEFAULTS["skills_dir"]
    outputs_dir: str = DEFAULTS["outputs_dir"]
    memory_auto_extract_interval: int = DEFAULTS["memory_auto_extract_interval"]
    memory_search_use_embedding: bool = DEFAULTS["memory_search_use_embedding"]
    jobs_model: str = DEFAULTS["jobs_model"]


def _coerce_setting(key: str, value: Any) -> Any:
    """Coerce string values to the Settings field's type (setattr bypasses Pydantic validation)."""
    if not isinstance(value, str):
        return value
    field = Settings.model_fields.get(key)
    if field is None:
        return value
    ann = field.annotation
    if ann is float or ann is int:
        try:
            return float(value) if ann is float else int(value)
        except ValueError:
            return value
    return value


# Module-level constants set once from DEFAULTS (used by importers)
DATABASE_URL = DEFAULTS["database_url"]
LLAMA_CPP_BASE_URL = DEFAULTS["llama_cpp_base_url"]
LLAMA_CPP_MODEL = DEFAULTS["llama_cpp_model"]
QUERY_MODEL = DEFAULTS["query_model"]
EMBEDDING_MODEL = DEFAULTS["embedding_model"]
RERANKING_MODEL = DEFAULTS["reranking_model"]
APP_HOST = DEFAULTS["app_host"]
APP_PORT = DEFAULTS["app_port"]
DEBUG = DEFAULTS["debug"]
DEFAULT_TEMPERATURE = DEFAULTS["default_temperature"]
DEFAULT_MAX_TOKENS = DEFAULTS["default_max_tokens"]
MAX_UPLOAD_SIZE = DEFAULTS["max_upload_size"]
UPLOAD_DIR = DEFAULTS["upload_dir"]
CORS_ORIGINS = DEFAULTS["cors_origins"].split(',')
SYSTEM_PROMPT = DEFAULTS["system_prompt"]
SQLALCHEMY_ECHO = DEFAULTS["sqlalchemy_echo"]
BACKUP_ENABLED = DEFAULTS["backup_enabled"]
BACKUP_PATH = DEFAULTS["backup_path"]
BACKUP_INTERVAL_HOURS = DEFAULTS["backup_interval_hours"]
BACKUP_MAX_KEEP = DEFAULTS["backup_max_keep"]
SKILLS_DIR = DEFAULTS["skills_dir"]
OUTPUTS_DIR = DEFAULTS["outputs_dir"]


# ponytail: env keys that map 1:1 with settings keys
_ENV_MAP = {
    "llama_cpp_base_url": "LLAMA_CPP_URL",
    "llama_cpp_model": "LLAMA_CPP_MODEL",
    "query_model": "QUERY_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "reranking_model": "RERANKING_MODEL",
    "app_host": "APP_HOST",
    "app_port": "APP_PORT",
    "debug": "DEBUG",
    "default_temperature": "DEFAULT_TEMPERATURE",
    "default_max_tokens": "DEFAULT_MAX_TOKENS",
    "max_upload_size": "MAX_UPLOAD_SIZE",
    "upload_dir": "UPLOAD_DIR",
    "system_prompt": "SYSTEM_PROMPT",
    "cors_origins": "CORS_ORIGINS",
    "tts_engine": "TTS_ENGINE",
    "tts_voice": "TTS_VOICE",
    "tts_rate": "TTS_RATE",
    "tts_volume": "TTS_VOLUME",
    "kokoro_lang": "KOKORO_LANG",
    "kokoro_device": "KOKORO_DEVICE",
    "kokoro_volume": "KOKORO_VOLUME",
    "kokoro_speed": "KOKORO_SPEED",
    "tts_auto_read": "TTS_AUTO_READ",
    "stt_engine": "STT_ENGINE",
    "stt_model": "STT_MODEL",
    "terminal_require_approval": "TERMINAL_REQUIRE_APPROVAL",
    "skills_dir": "SKILLS_DIR",
    "outputs_dir": "OUTPUTS_DIR",
    "jobs_model": "JOBS_MODEL",
}


def _sync_env(settings_dict: dict):
    """Sync settings dict to environment variables."""
    for key, env_var in _ENV_MAP.items():
        if key in settings_dict:
            val = settings_dict[key]
            os.environ[env_var] = str(val) if not isinstance(val, str) else val


class SettingsManager:
    """Manages application settings"""

    def __init__(self):
        self.settings = Settings()
        self.tts_service = None  # Will be set later
        # Load settings from file if it exists
        self.load_settings_from_file()

    def set_tts_service(self, tts_service):
        """Set the TTS service for configuration updates"""
        self.tts_service = tts_service

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self.settings.dict()

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings with new values"""
        # Update the settings object
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, _coerce_setting(key, value))

        # Sync changed env vars
        _sync_env(new_settings)

        # Update TTS settings if changed and TTS service is available
        tts_keys = {'tts_engine', 'tts_voice', 'tts_rate', 'tts_volume',
                    'kokoro_lang', 'kokoro_device', 'kokoro_volume', 'kokoro_speed',
                    'tts_auto_read'}
        if tts_keys & set(new_settings.keys()) and self.tts_service:
            from .tools.tts_service import TTSConfig
            tts_config = TTSConfig.from_settings(new_settings)
            self.tts_service.update_config(tts_config)

        # Save settings to file
        self.save_settings_to_file()

        return self.get_settings()

    def load_settings_from_file(self):
        """Load settings from file if it exists"""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    saved_settings = json.load(f)

                # Update settings object with saved values
                for key, value in saved_settings.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, _coerce_setting(key, value))

                # Sync env vars from saved settings
                _sync_env(saved_settings)
            except Exception as e:
                print(f"Error loading settings from file: {e}")

    def save_settings_to_file(self):
        """Save current settings to file"""
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.get_settings(), f, indent=2)
        except Exception as e:
            print(f"Error saving settings to file: {e}")


# Global settings manager instance
settings_manager = SettingsManager()

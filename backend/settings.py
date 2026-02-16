"""
Application settings management module
"""
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import importlib.util
import os

# Dynamically load config module
config_spec = importlib.util.spec_from_file_location("config", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "config.py"))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)

# Import the variables from the loaded module
DATABASE_URL = config_module.DATABASE_URL
LLAMA_CPP_BASE_URL = config_module.LLAMA_CPP_BASE_URL
LLAMA_CPP_MODEL = config_module.LLAMA_CPP_MODEL
QUERY_MODEL = config_module.QUERY_MODEL
APP_HOST = config_module.APP_HOST
APP_PORT = config_module.APP_PORT
DEBUG = config_module.DEBUG
DEFAULT_TEMPERATURE = config_module.DEFAULT_TEMPERATURE
DEFAULT_MAX_TOKENS = config_module.DEFAULT_MAX_TOKENS
MAX_UPLOAD_SIZE = config_module.MAX_UPLOAD_SIZE
UPLOAD_DIR = config_module.UPLOAD_DIR
CORS_ORIGINS = config_module.CORS_ORIGINS
SYSTEM_PROMPT = config_module.SYSTEM_PROMPT


class Settings(BaseModel):
    """Application settings model"""
    # Database Configuration
    database_url: str = Field(default=DATABASE_URL, description="Database URL")
    
    # Llama.cpp Configuration
    llama_cpp_base_url: str = Field(default=LLAMA_CPP_BASE_URL, description="Base URL for llama.cpp server")
    llama_cpp_model: str = Field(default=LLAMA_CPP_MODEL, description="Default model for llama.cpp")
    query_model: str = Field(default=QUERY_MODEL, description="Model used for query processing and title generation")
    
    # Application Settings
    app_host: str = Field(default=APP_HOST, description="Host address for the application")
    app_port: int = Field(default=APP_PORT, description="Port for the application")
    debug: bool = Field(default=DEBUG, description="Debug mode")
    
    # File Upload Settings
    max_upload_size: int = Field(default=MAX_UPLOAD_SIZE, description="Maximum upload size in bytes")
    upload_dir: str = Field(default=UPLOAD_DIR, description="Directory for uploads")
    
    # CORS Settings
    cors_origins: str = Field(default=",".join(CORS_ORIGINS), description="Comma-separated list of CORS origins")
    
    # System Prompt
    system_prompt: str = Field(default=SYSTEM_PROMPT, description="Default system prompt")
    
    # LLM Generation Defaults
    default_temperature: float = Field(default=DEFAULT_TEMPERATURE, description="Default temperature for LLM generation")
    default_max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, description="Default max tokens for LLM generation")
    
    # TTS Settings
    tts_engine: str = Field(default="edge-tts", description="TTS engine to use (edge-tts, pyttsx3, kokoro)")
    tts_voice: str = Field(default="en-IN-NeerjaNeural", description="Voice to use for TTS")
    tts_rate: str = Field(default="+0%", description="Speech rate adjustment")
    tts_volume: float = Field(default=1.0, description="Volume level (0.0 to 1.0)")
    kokoro_lang: str = Field(default="a", description="Kokoro language code (a=American English, b=British English)")
    kokoro_device: str = Field(default="cpu", description="Kokoro device (cpu or cuda)")


class SettingsManager:
    """Manages application settings"""
    
    def __init__(self):
        self.settings = Settings()
        self.tts_service = None  # Will be set later
    
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
                setattr(self.settings, key, value)
        
        # Update environment variables for runtime changes where applicable
        if 'llama_cpp_base_url' in new_settings:
            os.environ['LLAMA_CPP_URL'] = new_settings['llama_cpp_base_url']
        if 'llama_cpp_model' in new_settings:
            os.environ['LLAMA_CPP_MODEL'] = new_settings['llama_cpp_model']
        if 'query_model' in new_settings:
            os.environ['QUERY_MODEL'] = new_settings['query_model']
        if 'app_host' in new_settings:
            os.environ['APP_HOST'] = new_settings['app_host']
        if 'app_port' in new_settings:
            os.environ['APP_PORT'] = str(new_settings['app_port'])
        if 'debug' in new_settings:
            os.environ['DEBUG'] = str(new_settings['debug']).lower()
        if 'default_temperature' in new_settings:
            os.environ['DEFAULT_TEMPERATURE'] = str(new_settings['default_temperature'])
        if 'default_max_tokens' in new_settings:
            os.environ['DEFAULT_MAX_TOKENS'] = str(new_settings['default_max_tokens'])
        if 'max_upload_size' in new_settings:
            os.environ['MAX_UPLOAD_SIZE'] = str(new_settings['max_upload_size'])
        if 'upload_dir' in new_settings:
            os.environ['UPLOAD_DIR'] = new_settings['upload_dir']
        if 'system_prompt' in new_settings:
            os.environ['SYSTEM_PROMPT'] = new_settings['system_prompt']
        
        # Update CORS origins if changed
        if 'cors_origins' in new_settings:
            os.environ['CORS_ORIGINS'] = new_settings['cors_origins']
        
        # Update TTS settings if changed and TTS service is available
        if ('tts_engine' in new_settings or 'tts_voice' in new_settings or 
            'tts_rate' in new_settings or 'tts_volume' in new_settings or
            'kokoro_lang' in new_settings or 'kokoro_device' in new_settings) and self.tts_service:
            from .tools.tts_service import TTSConfig
            tts_config = TTSConfig.from_settings(new_settings)
            self.tts_service.update_config(tts_config)
        
        if 'tts_engine' in new_settings:
            os.environ['TTS_ENGINE'] = new_settings['tts_engine']
        if 'tts_voice' in new_settings:
            os.environ['TTS_VOICE'] = new_settings['tts_voice']
        if 'tts_rate' in new_settings:
            os.environ['TTS_RATE'] = new_settings['tts_rate']
        if 'tts_volume' in new_settings:
            os.environ['TTS_VOLUME'] = str(new_settings['tts_volume'])
        if 'kokoro_lang' in new_settings:
            os.environ['KOKORO_LANG'] = new_settings['kokoro_lang']
        if 'kokoro_device' in new_settings:
            os.environ['KOKORO_DEVICE'] = new_settings['kokoro_device']
        
        return self.get_settings()


# Global settings manager instance
settings_manager = SettingsManager()
"""
Text-to-Speech (TTS) Service for the LLM UI.

Provides lightweight TTS capabilities using either:
- edge-tts (Microsoft Edge TTS API - requires internet but high quality)
- pyttsx3 (offline, cross-platform)
- or a local model like Piper TTS (very lightweight, ~50MB)
"""

import asyncio
import os
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from config import UPLOAD_DIR

# Try to import TTS backends
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


@dataclass
class TTSConfig:
    """Configuration for TTS service"""
    engine: str = "edge-tts"  # Options: "edge-tts", "pyttsx3", "piper"
    #voice: str = "en-US-ChristopherNeural"  # Default Edge TTS voice
    voice: str = "en-IN-NeerjaNeural" #en-IN-PrabhatNeural en-IN-NeerjaNeural
    rate: str = "+0%"  # Speech rate adjustment
    volume: float = 1.0  # Volume (0.0 to 1.0)
    output_dir: str = UPLOAD_DIR


class TTSService:
    """Text-to-Speech service supporting multiple backends"""
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self._ensure_output_dir()
        
        # Initialize pyttsx3 engine if needed
        if self.config.engine == "pyttsx3" and HAS_PYTTSX3:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', self.config.volume)
        else:
            self.engine = None
    
    def _ensure_output_dir(self):
        """Ensure TTS output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    async def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        output_format: str = "mp3"
    ) -> Dict[str, Any]:
        """
        Generate speech audio from text.
        
        Args:
            text: The text to convert to speech
            voice: Optional voice override
            rate: Optional rate override (e.g., "+10%", "-20%")
            output_format: Audio format (mp3, wav, etc.)
            
        Returns:
            Dict with 'success', 'filepath', 'url', and optional 'error'
        """
        if not text.strip():
            return {"success": False, "error": "No text provided"}
        
        voice = voice or self.config.voice
        rate = rate or self.config.rate
        
        # Generate unique filename
        filename = f"tts_{uuid.uuid4()}.{output_format}"
        filepath = os.path.join(self.config.output_dir, filename)
        
        try:
            if self.config.engine == "edge-tts" and HAS_EDGE_TTS:
                return await self._generate_with_edge_tts(text, voice, rate, filepath)
            elif self.config.engine == "pyttsx3" and HAS_PYTTSX3:
                return await self._generate_with_pyttsx3(text, filepath)
            else:
                # Fallback: try edge-tts if available
                if HAS_EDGE_TTS:
                    return await self._generate_with_edge_tts(text, voice, rate, filepath)
                else:
                    return {"success": False, "error": "No TTS engine available. Install edge-tts or pyttsx3."}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_with_edge_tts(
        self,
        text: str,
        voice: str,
        rate: str,
        filepath: str
    ) -> Dict[str, Any]:
        """Generate speech using Microsoft Edge TTS (high quality, requires internet)"""
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(filepath)
            
            return {
                "success": True,
                "filepath": filepath,
                "audio_url": f"/api/audio/{os.path.basename(filepath)}",
                "engine": "edge-tts",
                "voice": voice
            }
        except Exception as e:
            return {"success": False, "error": f"Edge TTS error: {str(e)}"}
    
    async def _generate_with_pyttsx3(
        self,
        text: str,
        filepath: str
    ) -> Dict[str, Any]:
        """Generate speech using pyttsx3 (offline, lower quality)"""
        try:
            # pyttsx3 is synchronous, run in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.engine.save_to_file(text, filepath)
            )
            self.engine.runAndWait()
            
            return {
                "success": True,
                "filepath": filepath,
                "audio_url": f"/api/audio/{os.path.basename(filepath)}",
                "engine": "pyttsx3"
            }
        except Exception as e:
            return {"success": False, "error": f"pyttsx3 error: {str(e)}"}
    
    def list_available_voices(self) -> Dict[str, Any]:
        """List available voices for the configured engine"""
        voices = []
        
        if self.config.engine == "edge-tts" and HAS_EDGE_TTS:
            # Edge TTS voices are fetched asynchronously
            # This is a simplified list of common voices
            voices = [
                {"id": "en-US-ChristopherNeural", "name": "Christopher (Male, US)", "gender": "male", "locale": "en-US"},
                {"id": "en-US-JennyNeural", "name": "Jenny (Female, US)", "gender": "female", "locale": "en-US"},
                {"id": "en-GB-SoniaNeural", "name": "Sonia (Female, UK)", "gender": "female", "locale": "en-GB"},
                {"id": "en-AU-NatashaNeural", "name": "Natasha (Female, AU)", "gender": "female", "locale": "en-AU"},
            ]
        elif self.config.engine == "pyttsx3" and HAS_PYTTSX3:
            for voice in self.engine.getProperty('voices'):
                voices.append({
                    "id": voice.id,
                    "name": voice.name,
                    "gender": "unknown",
                    "locale": voice.languages[0] if voice.languages else "unknown"
                })
        
        return {
            "engine": self.config.engine,
            "voices": voices,
            "default_voice": self.config.voice
        }


# Tool definition for LLM function calling
TTS_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_speech",
        "description": "Generate speech audio from text. Use this to provide audio output for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert to speech"
                },
                "voice": {
                    "type": "string",
                    "description": "Optional voice ID (if not specified, uses default)"
                }
            },
            "required": ["text"]
        }
    }
}

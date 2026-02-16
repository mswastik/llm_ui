"""
Text-to-Speech (TTS) Service for the LLM UI.

Provides lightweight TTS capabilities using either:
- edge-tts (Microsoft Edge TTS API - requires internet but high quality)
- pyttsx3 (offline, cross-platform)
- kokoro (high-quality local TTS, requires model download from HuggingFace)
- or a local model like Piper TTS (very lightweight, ~50MB)
"""

import asyncio
import hashlib
import os
import uuid
from typing import Optional, Dict, Any, List
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

# Kokoro is imported lazily to avoid import-time errors with spacy/pydantic on Python 3.14
# We check availability when actually needed
HAS_KOKORO = None  # None means "not checked yet"

def _check_kokoro_available():
    """Check if kokoro is available (lazy check to avoid import-time errors)"""
    global HAS_KOKORO
    if HAS_KOKORO is None:
        try:
            from kokoro import KPipeline
            HAS_KOKORO = True
        except ImportError:
            HAS_KOKORO = False
        except Exception:
            # Catch any other import-time errors (e.g., pydantic/spacy compatibility)
            HAS_KOKORO = False
    return HAS_KOKORO

# Kokoro voice definitions
KOKORO_VOICES = {
    # American English voices
    "af_bella": {"name": "Bella (Female, American)", "gender": "female", "locale": "en-US"},
    "af_sarah": {"name": "Sarah (Female, American)", "gender": "female", "locale": "en-US"},
    "af_sky": {"name": "Sky (Female, American)", "gender": "female", "locale": "en-US"},
    "am_adam": {"name": "Adam (Male, American)", "gender": "male", "locale": "en-US"},
    "am_michael": {"name": "Michael (Male, American)", "gender": "male", "locale": "en-US"},
    # British English voices
    "bf_emma": {"name": "Emma (Female, British)", "gender": "female", "locale": "en-GB"},
    "bf_isabella": {"name": "Isabella (Female, British)", "gender": "female", "locale": "en-GB"},
    "bm_george": {"name": "George (Male, British)", "gender": "male", "locale": "en-GB"},
    "bm_lewis": {"name": "Lewis (Male, British)", "gender": "male", "locale": "en-GB"},
}


@dataclass
class TTSConfig:
    """Configuration for TTS service"""
    engine: str = "edge-tts"  # Options: "edge-tts", "pyttsx3", "kokoro"
    #voice: str = "en-US-ChristopherNeural"  # Default Edge TTS voice
    voice: str = "en-IN-NeerjaNeural" #en-IN-PrabhatNeural en-IN-NeerjaNeural
    rate: str = "+0%"  # Speech rate adjustment
    volume: float = 1.0  # Volume (0.0 to 1.0)
    output_dir: str = UPLOAD_DIR
    kokoro_lang: str = "a"  # Kokoro language code: 'a' for American English, 'b' for British English
    kokoro_device: str = "cpu"  # Kokoro device: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    
    @classmethod
    def from_settings(cls, settings_dict: dict):
        """Create TTSConfig from settings dictionary"""
        return cls(
            engine=settings_dict.get('tts_engine', 'edge-tts'),
            voice=settings_dict.get('tts_voice', 'en-IN-NeerjaNeural'),
            rate=settings_dict.get('tts_rate', '+0%'),
            volume=float(settings_dict.get('tts_volume', 1.0)),
            output_dir=settings_dict.get('upload_dir', UPLOAD_DIR),
            kokoro_lang=settings_dict.get('kokoro_lang', 'a'),
            kokoro_device=settings_dict.get('kokoro_device', 'cpu')
        )


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
        
        # Initialize Kokoro pipeline if needed (lazy loading)
        self._kokoro_pipeline = None
    
    def _ensure_output_dir(self):
        """Ensure TTS output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _get_cache_filename(self, text: str, voice: str, rate: str, output_format: str) -> str:
        """Generate a consistent filename based on text content and parameters"""
        # Create a hash of the text content along with voice and rate parameters
        text_hash = hashlib.md5(f"{text}_{voice}_{rate}".encode()).hexdigest()
        
        # Kokoro outputs WAV format, so we use wav for kokoro
        actual_format = "wav" if self.config.engine == "kokoro" else output_format
        
        return f"tts_{text_hash}.{actual_format}"

    def _get_kokoro_pipeline(self):
        """Get or create Kokoro pipeline (lazy loading)"""
        if self._kokoro_pipeline is None and _check_kokoro_available():
            from kokoro import KPipeline
            lang_code = self.config.kokoro_lang
            device = self.config.kokoro_device

            # Validate device setting - allow cpu, cuda, cuda:0, cuda:1, etc.
            valid_devices = ('cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3')
            if device not in valid_devices and not device.startswith('cuda:'):
                device = 'cpu'

            try:
                self._kokoro_pipeline = KPipeline(lang_code=lang_code, device=device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    print(f"Warning: CUDA OOM error occurred: {e}. Falling back to CPU.")
                    # Fall back to CPU
                    self._kokoro_pipeline = KPipeline(lang_code=lang_code, device='cpu')
                else:
                    raise e
        return self._kokoro_pipeline
    
    def update_config(self, new_config: TTSConfig):
        """Update the TTS configuration"""
        old_device = self.config.kokoro_device if self.config else None
        old_lang = self.config.kokoro_lang if self.config else None
        
        self.config = new_config
        self._ensure_output_dir()
        
        # Reset Kokoro pipeline if device or language changes
        if (old_device != new_config.kokoro_device or old_lang != new_config.kokoro_lang):
            self._kokoro_pipeline = None
    
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

        # Generate cache filename based on text content and parameters
        filename = self._get_cache_filename(text, voice, rate, output_format)
        filepath = os.path.join(self.config.output_dir, filename)

        # Check if cached file already exists
        if os.path.exists(filepath):
            print(f"Using cached TTS file: {filepath}")
            return {
                "success": True,
                "filepath": filepath,
                "audio_url": f"/api/audio/{os.path.basename(filepath)}",
                "engine": self.config.engine,
                "voice": voice,
                "cached": True
            }

        try:
            if self.config.engine == "edge-tts" and HAS_EDGE_TTS:
                return await self._generate_with_edge_tts(text, voice, rate, filepath)
            elif self.config.engine == "pyttsx3" and HAS_PYTTSX3:
                return await self._generate_with_pyttsx3(text, filepath)
            elif self.config.engine == "kokoro" and _check_kokoro_available():
                return await self._generate_with_kokoro(text, voice, filepath)
            else:
                # Fallback: try edge-tts if available
                if HAS_EDGE_TTS:
                    return await self._generate_with_edge_tts(text, voice, rate, filepath)
                elif _check_kokoro_available():
                    return await self._generate_with_kokoro(text, voice, filepath)
                else:
                    return {"success": False, "error": "No TTS engine available. Install edge-tts, pyttsx3, or kokoro."}

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
            # Set volume property before generating speech
            self.engine.setProperty('volume', self.config.volume)
            
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
    
    async def _generate_with_kokoro(
        self,
        text: str,
        voice: Optional[str],
        filepath: str
    ) -> Dict[str, Any]:
        """Generate speech using Kokoro TTS (high quality, local, requires model download)"""
        try:
            import soundfile as sf
            import numpy as np

            # Get the Kokoro pipeline
            pipeline = self._get_kokoro_pipeline()
            if pipeline is None:
                return {"success": False, "error": "Kokoro pipeline not available. Install kokoro."}

            # Use default voice if not specified
            voice = voice or "af_bella"

            # Validate voice
            if voice not in KOKORO_VOICES:
                # Try to use a similar voice based on language code
                lang_code = self.config.kokoro_lang
                if lang_code == 'a':
                    voice = "af_bella"  # Default American female
                elif lang_code == 'b':
                    voice = "bf_emma"  # Default British female
                else:
                    voice = "af_bella"  # Fallback

            # Generate speech in a thread pool (Kokoro is synchronous)
            loop = asyncio.get_event_loop()

            def _generate():
                # Kokoro returns generator of (graphemes, phonemes, audio)
                audio_segments = []
                for _, _, audio in pipeline(text, voice=voice):
                    audio_segments.append(audio)

                # Concatenate all audio segments
                if audio_segments:
                    full_audio = np.concatenate(audio_segments)
                    return full_audio
                return None

            try:
                audio_data = await loop.run_in_executor(None, _generate)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    # Clear the pipeline and force recreation with CPU
                    self._kokoro_pipeline = None
                    pipeline = self._get_kokoro_pipeline()  # This will use CPU fallback
                    if pipeline is None:
                        return {"success": False, "error": "Kokoro pipeline not available after CPU fallback."}
                    
                    # Retry generation with CPU pipeline
                    def _generate_cpu():
                        audio_segments = []
                        for _, _, audio in pipeline(text, voice=voice):
                            audio_segments.append(audio)
                        
                        if audio_segments:
                            full_audio = np.concatenate(audio_segments)
                            return full_audio
                        return None
                    
                    audio_data = await loop.run_in_executor(None, _generate_cpu)
                else:
                    raise e

            if audio_data is None:
                return {"success": False, "error": "Kokoro generated no audio"}

            # Apply volume adjustment if needed
            if self.config.volume != 1.0:
                audio_data = audio_data * self.config.volume

            # Save as WAV file (Kokoro outputs at 24kHz)
            sf.write(filepath, audio_data, 24000)

            return {
                "success": True,
                "filepath": filepath,
                "audio_url": f"/api/audio/{os.path.basename(filepath)}",
                "engine": "kokoro",
                "voice": voice
            }
        except ImportError as e:
            missing_pkg = str(e).split("'")[-2] if "'" in str(e) else "required package"
            return {"success": False, "error": f"Missing dependency for Kokoro: {missing_pkg}. Install with: pip install {missing_pkg}"}
        except Exception as e:
            return {"success": False, "error": f"Kokoro TTS error: {str(e)}"}
    
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
        elif self.config.engine == "kokoro" and _check_kokoro_available():
            # Return Kokoro voices
            for voice_id, voice_info in KOKORO_VOICES.items():
                voices.append({
                    "id": voice_id,
                    "name": voice_info["name"],
                    "gender": voice_info["gender"],
                    "locale": voice_info["locale"]
                })
        
        return {
            "engine": self.config.engine,
            "voices": voices,
            "default_voice": self.config.voice
        }
    
    @staticmethod
    def get_available_engines() -> Dict[str, Any]:
        """Get list of available TTS engines and their status"""
        engines = []
        
        if HAS_EDGE_TTS:
            engines.append({
                "id": "edge-tts",
                "name": "Edge TTS",
                "description": "Microsoft Edge TTS - high quality, requires internet",
                "available": True
            })
        else:
            engines.append({
                "id": "edge-tts",
                "name": "Edge TTS",
                "description": "Microsoft Edge TTS - high quality, requires internet (not installed)",
                "available": False
            })
        
        if HAS_PYTTSX3:
            engines.append({
                "id": "pyttsx3",
                "name": "pyttsx3",
                "description": "Offline TTS - lower quality, no internet required",
                "available": True
            })
        else:
            engines.append({
                "id": "pyttsx3",
                "name": "pyttsx3",
                "description": "Offline TTS - lower quality, no internet required (not installed)",
                "available": False
            })
        
        if _check_kokoro_available():
            engines.append({
                "id": "kokoro",
                "name": "Kokoro TTS",
                "description": "High-quality local TTS - requires model download from HuggingFace",
                "available": True
            })
        else:
            engines.append({
                "id": "kokoro",
                "name": "Kokoro TTS",
                "description": "High-quality local TTS - requires model download from HuggingFace (not installed)",
                "available": False
            })
        
        return {
            "engines": engines,
            "default_engine": "edge-tts" if HAS_EDGE_TTS else ("kokoro" if _check_kokoro_available() else "pyttsx3")
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

"""
Speech-to-Text (STT) Service for the LLM UI.

Provides voice input transcription using either:
- faster-whisper (local, fast, no API key)
- OpenAI Whisper API (optional, requires API key)
"""

import os
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# Try to import openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class STTConfig:
    """Configuration for STT service"""
    engine: str = "faster-whisper"  # "faster-whisper" or "openai"
    model: str = "base"  # tiny, base, small, medium, large-v3
    language: Optional[str] = None  # None = auto-detect
    openai_api_key: Optional[str] = None
    openai_model: str = "whisper-1"

    @classmethod
    def from_settings(cls, settings_dict: dict):
        """Create STTConfig from settings dictionary"""
        return cls(
            engine=settings_dict.get('stt_engine', 'faster-whisper'),
            model=settings_dict.get('stt_model', 'base'),
            language=settings_dict.get('stt_language') or None,
            openai_api_key=settings_dict.get('stt_openai_api_key') or None,
        )


class STTService:
    """Speech-to-Text service supporting multiple backends"""

    def __init__(self, config: STTConfig = None):
        self.config = config or STTConfig()
        self._model = None  # lazy loaded

    def _get_model(self):
        """Get or create the faster-whisper model (lazy loading)"""
        if self._model is None and HAS_FASTER_WHISPER and self.config.engine == "faster-whisper":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                self._model = WhisperModel(self.config.model, device=device, compute_type=compute_type)
                print(f"[STT] Loaded faster-whisper model '{self.config.model}' on {device} ({compute_type})")
            except Exception as e:
                print(f"[STT] Failed to load faster-whisper on CUDA, falling back to CPU: {e}")
                try:
                    self._model = WhisperModel(self.config.model, device="cpu", compute_type="int8")
                    print(f"[STT] Loaded faster-whisper model '{self.config.model}' on CPU (int8)")
                except Exception as e2:
                    print(f"[STT] Failed to load faster-whisper: {e2}")
                    raise
        return self._model

    async def transcribe(self, audio_data: bytes, filename: str = "recording.webm") -> Dict[str, Any]:
        """Transcribe audio bytes to text.

        Accepts raw audio bytes (WebM/ogg/wav/mp3). Saves temporarily,
        transcribes, returns result.
        """
        if not audio_data:
            return {"success": False, "error": "No audio data provided"}

        if self.config.engine == "openai" and HAS_OPENAI and self.config.openai_api_key:
            return await self._transcribe_with_openai(audio_data, filename)
        elif HAS_FASTER_WHISPER:
            return await self._transcribe_with_faster_whisper(audio_data)
        else:
            return {"success": False, "error": "No STT engine available. Install faster-whisper or configure OpenAI API key."}

    async def _transcribe_with_faster_whisper(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe using local faster-whisper model."""
        import asyncio

        model = self._get_model()
        if model is None:
            return {"success": False, "error": "faster-whisper model not loaded"}

        # Write bytes to a temp file (faster-whisper reads from disk)
        suffix = ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_event_loop()

            def _run():
                segments, info = model.transcribe(
                    tmp_path,
                    language=self.config.language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                segments_list = list(segments)
                text = " ".join(s.text.strip() for s in segments_list)
                return {
                    "text": text,
                    "language": info.language,
                    "duration": info.duration,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text.strip()}
                        for s in segments_list
                    ],
                }

            result = await loop.run_in_executor(None, _run)
            result["success"] = True
            result["engine"] = "faster-whisper"
            return result

        except Exception as e:
            return {"success": False, "error": f"faster-whisper error: {str(e)}"}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _transcribe_with_openai(self, audio_data: bytes, filename: str) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        import asyncio

        client = OpenAI(api_key=self.config.openai_api_key)

        try:
            loop = asyncio.get_event_loop()

            def _run():
                # Determine content type from filename
                if filename.endswith(".webm"):
                    content_type = "audio/webm"
                elif filename.endswith(".ogg"):
                    content_type = "audio/ogg"
                elif filename.endswith(".wav"):
                    content_type = "audio/wav"
                elif filename.endswith(".mp3"):
                    content_type = "audio/mpeg"
                elif filename.endswith(".m4a"):
                    content_type = "audio/mp4"
                else:
                    content_type = "audio/webm"

                transcript = client.audio.transcriptions.create(
                    model=self.config.openai_model,
                    file=(filename, audio_data, content_type),
                    language=self.config.language,
                    response_format="verbose_json",
                )
                return transcript

            result = await loop.run_in_executor(None, _run)

            segments = []
            if hasattr(result, 'segments') and result.segments:
                for s in result.segments:
                    segments.append({
                        "start": getattr(s, 'start', 0),
                        "end": getattr(s, 'end', 0),
                        "text": getattr(s, 'text', ''),
                    })

            return {
                "success": True,
                "text": result.text,
                "language": getattr(result, 'language', None),
                "duration": getattr(result, 'duration', None),
                "segments": segments,
                "engine": "openai",
            }

        except Exception as e:
            return {"success": False, "error": f"OpenAI STT error: {str(e)}"}

    async def check_availability(self) -> Dict[str, Any]:
        """Check which STT engines are available."""
        engines = []

        if HAS_FASTER_WHISPER:
            engines.append({
                "id": "faster-whisper",
                "name": "faster-whisper",
                "description": "Local Whisper transcription (fast, no API key)",
                "available": True,
            })
        else:
            engines.append({
                "id": "faster-whisper",
                "name": "faster-whisper",
                "description": "Local Whisper transcription (not installed)",
                "available": False,
            })

        if HAS_OPENAI:
            engines.append({
                "id": "openai",
                "name": "OpenAI Whisper API",
                "description": "OpenAI Whisper API (requires API key)",
                "available": True,
            })
        else:
            engines.append({
                "id": "openai",
                "name": "OpenAI Whisper API",
                "description": "OpenAI Whisper API - requires openai package (not installed)",
                "available": False,
            })

        return {
            "engines": engines,
            "current_engine": self.config.engine,
            "default_engine": "faster-whisper" if HAS_FASTER_WHISPER else ("openai" if HAS_OPENAI else None),
            "available": HAS_FASTER_WHISPER or (HAS_OPENAI and bool(self.config.openai_api_key)),
        }

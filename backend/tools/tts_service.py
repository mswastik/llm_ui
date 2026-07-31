"""
Text-to-Speech (TTS) Service for the LLM UI.

Provides lightweight TTS capabilities using either:
- edge-tts (Microsoft Edge TTS API - requires internet but high quality)
- kokoro (high-quality local TTS, requires model download from HuggingFace)
"""

import asyncio
import hashlib
import os
import re
import uuid
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from settings import UPLOAD_DIR

# Try to import TTS backends
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False


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

# Kokoro voice definitions (validated against hexgrad/Kokoro-82M voices/)
KOKORO_VOICES = {
    # American English female
    "af_bella": {"name": "Bella (Female, American)", "gender": "female", "locale": "en-US"},
    "af_sarah": {"name": "Sarah (Female, American)", "gender": "female", "locale": "en-US"},
    "af_sky": {"name": "Sky (Female, American)", "gender": "female", "locale": "en-US"},
    "af_heart": {"name": "Heart (Female, American)", "gender": "female", "locale": "en-US"},
    "af_nicole": {"name": "Nicole (Female, American)", "gender": "female", "locale": "en-US"},
    "af_aoede": {"name": "Aoede (Female, American)", "gender": "female", "locale": "en-US"},
    "af_kore": {"name": "Kore (Female, American)", "gender": "female", "locale": "en-US"},
    "af_nova": {"name": "Nova (Female, American)", "gender": "female", "locale": "en-US"},
    "af_alloy": {"name": "Alloy (Female, American)", "gender": "female", "locale": "en-US"},
    "af_jessica": {"name": "Jessica (Female, American)", "gender": "female", "locale": "en-US"},
    "af_river": {"name": "River (Female, American)", "gender": "female", "locale": "en-US"},
    # American English male
    "am_adam": {"name": "Adam (Male, American)", "gender": "male", "locale": "en-US"},
    "am_michael": {"name": "Michael (Male, American)", "gender": "male", "locale": "en-US"},
    "am_echo": {"name": "Echo (Male, American)", "gender": "male", "locale": "en-US"},
    "am_eric": {"name": "Eric (Male, American)", "gender": "male", "locale": "en-US"},
    "am_fenrir": {"name": "Fenrir (Male, American)", "gender": "male", "locale": "en-US"},
    "am_liam": {"name": "Liam (Male, American)", "gender": "male", "locale": "en-US"},
    "am_onyx": {"name": "Onyx (Male, American)", "gender": "male", "locale": "en-US"},
    "am_puck": {"name": "Puck (Male, American)", "gender": "male", "locale": "en-US"},
    "am_santa": {"name": "Santa (Male, American)", "gender": "male", "locale": "en-US"},
    # British English female
    "bf_emma": {"name": "Emma (Female, British)", "gender": "female", "locale": "en-GB"},
    "bf_isabella": {"name": "Isabella (Female, British)", "gender": "female", "locale": "en-GB"},
    "bf_alice": {"name": "Alice (Female, British)", "gender": "female", "locale": "en-GB"},
    "bf_lily": {"name": "Lily (Female, British)", "gender": "female", "locale": "en-GB"},
    # British English male
    "bm_george": {"name": "George (Male, British)", "gender": "male", "locale": "en-GB"},
    "bm_lewis": {"name": "Lewis (Male, British)", "gender": "male", "locale": "en-GB"},
    "bm_daniel": {"name": "Daniel (Male, British)", "gender": "male", "locale": "en-GB"},
    "bm_fable": {"name": "Fable (Male, British)", "gender": "male", "locale": "en-GB"},
}


@dataclass
class TTSConfig:
    """Configuration for TTS service"""
    engine: str = "edge-tts"  # Options: "edge-tts", "kokoro"
    #voice: str = "en-US-ChristopherNeural"  # Default Edge TTS voice
    voice: str = "en-US-MichelleNeural" #en-IN-PrabhatNeural en-IN-NeerjaNeural
    rate: str = "+0%"  # Speech rate adjustment
    volume: float = 1.0  # Volume (0.0 to 1.0)
    output_dir: str = UPLOAD_DIR
    kokoro_lang: str = "a"  # Kokoro language code: 'a' for American English, 'b' for British English
    kokoro_device: str = "cpu"  # Kokoro device: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    kokoro_volume: float = 1.0  # Kokoro volume (0.0 to 1.0)
    kokoro_speed: float = 1.0  # Kokoro speed multiplier (0.5 to 2.0)

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
            kokoro_device=settings_dict.get('kokoro_device', 'cpu'),
            kokoro_volume=float(settings_dict.get('kokoro_volume', 1.0)),
            kokoro_speed=float(settings_dict.get('kokoro_speed', 1.0))
        )


_MONEY_SUFFIXES = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _money_to_words(raw, suffix):
    """'1.5', 'M' -> 'one point five million dollars'; '100' -> 'one hundred dollars'."""
    from num2words import num2words
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    if suffix:
        value *= _MONEY_SUFFIXES[suffix.lower()]
    dollars = int(value)
    cents = round((value - dollars) * 100)
    def nw(x):  # num2words inserts grouping commas; strip for smooth TTS
        return num2words(x).replace(",", "")
    if dollars and cents:
        return f"{nw(dollars)} dollars and {nw(cents)} cents"
    if dollars:
        unit = "dollar" if dollars == 1 else "dollars"
        return f"{nw(dollars)} {unit}"
    if cents:
        return f"{nw(cents)} cents"
    return "zero dollars"


def normalize_tts_text(text: str) -> str:
    """Rewrite number-heavy text so TTS engines (esp. Kokoro) read it correctly.

    Kokoro/misaki mangles $ amounts, percents, ordinals and bare numbers. The
    rewrites here are conservative — they match what Edge TTS would say anyway
    — so it is safe to run before any engine. Requires num2words; no-ops if missing.
    """
    try:
        from num2words import num2words
    except ImportError:
        return text

    text = re.sub(r"\$\s*([\d,]+(?:\.\d+)?)\s+(million|billion|trillion|thousand)\b",
                  lambda m: f"{num2words(m.group(1).replace(',', '')).replace(',', '')} {m.group(2).lower()} dollars", text)
    text = re.sub(r"\$\s*([\d,]+(?:\.\d+)?)([kmbtKMBT]?)\b",
                  lambda m: _money_to_words(m.group(1), m.group(2)) or m.group(0), text)
    text = re.sub(r"\b([\d,]+(?:\.\d+)?)\s*\$",
                  lambda m: _money_to_words(m.group(1), "") or m.group(0), text)
    text = re.sub(r"\b([\d.,]+)%",
                  lambda m: f"{num2words(m.group(1).replace(',', '')).replace(',', '')} percent", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b",
                  lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    # Bare numbers, but not ones glued to : / - . (times, dates, ranges, versions)
    text = re.sub(r"(?<![\w:./-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w:./%-])",
                  lambda m: num2words(m.group(1).replace(",", "")).replace(",", ""), text)
    return text


class TTSService:
    """Text-to-Speech service supporting multiple backends"""
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self._ensure_output_dir()

        # Initialize Kokoro pipeline if needed (lazy loading)
        self._kokoro_pipeline = None

    def _ensure_output_dir(self):
        """Ensure TTS output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _get_cache_filename(self, text: str, voice: str, rate: str, output_format: str) -> str:
        """Generate a consistent filename based on text content and parameters"""
        # Create a hash of the text content along with voice and rate parameters
        # Include kokoro-specific settings in the hash for Kokoro engine
        if self.config.engine == "kokoro":
            text_hash = hashlib.md5(f"{text}_{voice}_{self.config.kokoro_volume}_{self.config.kokoro_speed}_{self.config.kokoro_lang}".encode()).hexdigest()
        else:
            text_hash = hashlib.md5(f"{text}_{voice}_{rate}".encode()).hexdigest()

        # Kokoro outputs WAV format
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
                msg = str(e).lower()
                # Fall back to CPU when CUDA is unusable (OOM, or torch built without CUDA)
                if "cuda" in msg and ("not available" in msg or "not compiled" in msg or "out of memory" in msg):
                    print(f"Warning: Kokoro CUDA unavailable ({e}). Falling back to CPU.")
                    self._kokoro_pipeline = KPipeline(lang_code=lang_code, device='cpu')
                else:
                    raise e
        return self._kokoro_pipeline
    
    def update_config(self, new_config: TTSConfig):
        """Update the TTS configuration"""
        old_device = self.config.kokoro_device if self.config else None
        old_lang = self.config.kokoro_lang if self.config else None
        old_volume = self.config.kokoro_volume if self.config else None
        old_speed = self.config.kokoro_speed if self.config else None

        self.config = new_config
        self._ensure_output_dir()

        # Log config changes for debugging
        print(f"TTS Config updated: engine={self.config.engine}, kokoro_volume={self.config.kokoro_volume}, kokoro_speed={self.config.kokoro_speed}")

        # Reset Kokoro pipeline if device or language changes
        if (old_device != new_config.kokoro_device or old_lang != new_config.kokoro_lang):
            self._kokoro_pipeline = None
            print(f"Kokoro pipeline reset due to device/lang change")
    

    async def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        output_format: str = "mp3",
        should_stop: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """Generate speech audio from text."""
        if not text.strip():
            return {"success": False, "error": "No text provided"}

        # Rewrite $ amounts, %, ordinals and bare numbers so the engine
        # (especially Kokoro) reads them correctly. Cache key uses the
        # normalized text since that is what gets synthesized.
        text = normalize_tts_text(text)

        voice = voice or self.config.voice
        rate = rate or self.config.rate

        filename = self._get_cache_filename(text, voice, rate, output_format)
        filepath = os.path.join(self.config.output_dir, filename)

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

        engine = self.config.engine

        if engine == "edge-tts" and HAS_EDGE_TTS:
            return await self._generate_with_edge_tts(text, voice, rate, filepath)
        elif engine == "kokoro" and _check_kokoro_available():
            return await self._generate_with_kokoro(text, voice, filepath, should_stop)

        # User explicitly chose an engine but it's not available → tell them, don't silently fall back
        if engine in ("edge-tts", "kokoro"):
            return {"success": False, "error": f"TTS engine '{engine}' is selected but not available. Check Settings → TTS."}

        # Auto-fallback when no engine is explicitly selected (shouldn't happen, but safe)
        if HAS_EDGE_TTS:
            return await self._generate_with_edge_tts(text, voice, rate, filepath)
        elif _check_kokoro_available():
            return await self._generate_with_kokoro(text, voice, filepath, should_stop)
        else:
            return {"success": False, "error": "No TTS engine available. Install edge-tts or kokoro."}
    
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
    
    async def _generate_with_kokoro(
        self,
        text: str,
        voice: Optional[str],
        filepath: str,
        should_stop: Optional[Callable[[], bool]] = None
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
            # Native speed param: shorter durations -> genuinely faster inference
            speed = self.config.kokoro_speed
            loop = asyncio.get_event_loop()

            def _generate():
                # Kokoro returns generator of (graphemes, phonemes, audio)
                # Bail between segments once the client has stopped/disconnected
                audio_segments = []
                for _, _, audio in pipeline(text, voice=voice, speed=speed):
                    if should_stop and should_stop():
                        print("[TTS] Kokoro generation cancelled by client")
                        return None
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
                        for _, _, audio in pipeline(text, voice=voice, speed=speed):
                            if should_stop and should_stop():
                                print("[TTS] Kokoro generation cancelled by client")
                                return None
                            audio_segments.append(audio)

                        if audio_segments:
                            full_audio = np.concatenate(audio_segments)
                            return full_audio
                        return None

                    audio_data = await loop.run_in_executor(None, _generate_cpu)
                else:
                    raise e

            if audio_data is None:
                if should_stop and should_stop():
                    return {"success": False, "error": "TTS generation cancelled"}
                return {"success": False, "error": "Kokoro generated no audio"}

            # Apply volume adjustment if needed (using kokoro-specific volume)
            if self.config.kokoro_volume != 1.0:
                print(f"Applying Kokoro volume adjustment: {self.config.kokoro_volume}")
                audio_data = audio_data * self.config.kokoro_volume

            # Speed is applied natively at inference (pipeline speed= param);
            # the old librosa/scipy time-stretch post-processing is gone.

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
            # ponytail: hardcoded subset; edge_tts.list_voices() is async and
            # this method is sync. Upgrade to async if full voice list is needed.
            voices = [
                {"id": "en-US-ChristopherNeural", "name": "Christopher (Male, US)", "gender": "male", "locale": "en-US"},
                {"id": "en-US-JennyNeural", "name": "Jenny (Female, US)", "gender": "female", "locale": "en-US"},
                {"id": "en-GB-SoniaNeural", "name": "Sonia (Female, UK)", "gender": "female", "locale": "en-GB"},
                {"id": "en-AU-NatashaNeural", "name": "Natasha (Female, AU)", "gender": "female", "locale": "en-AU"},
                {"id": "en-IN-NeerjaNeural", "name": "Neerja (Female, IN)", "gender": "female", "locale": "en-IN"},
                {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Male, IN)", "gender": "male", "locale": "en-IN"},
            ]
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
            "default_engine": "edge-tts" if HAS_EDGE_TTS else "kokoro"
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


if __name__ == "__main__":
    # Self-check: normalization must turn number-heavy text into words a
    # TTS engine would say correctly, and leave dates/times/ranges alone.
    cases = {
        "$100": "one hundred dollars",
        "$1.5M": "one million five hundred thousand dollars",
        "$100 million": "one hundred million dollars",
        "$1.5 billion": "one point five billion dollars",
        "$0.99": "ninety-nine cents",
        "100$": "one hundred dollars",
        "25%": "twenty-five percent",
        "1st place": "first place",
        "42 apples": "forty-two apples",
        "3.14": "three point one four",
        "3.5 stars": "three point five stars",
        "12:30 PM": "12:30 PM",  # untouched: time
        "12/25/2024": "12/25/2024",  # untouched: date
        "10-20 items": "10-20 items",  # untouched: range
        "5km away": "5km away",  # untouched: unit
    }
    for inp, expected in cases.items():
        got = normalize_tts_text(inp)
        assert got == expected, f"{inp!r} -> {got!r}, expected {expected!r}"
    print("normalizer self-check: OK")

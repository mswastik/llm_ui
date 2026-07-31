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
from settings import UPLOAD_DIR

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

def _check_inflect_nano_available(model_path: Optional[str] = None):
    """Check if Inflect-Nano-v1 model files exist at the configured path.

    Always re-checks the filesystem — no global cache, because the user
    may change the path in settings or clone the repo after startup.
    """
    path = model_path or "models/Inflect-Nano-v1"
    acoustic = os.path.join(path, "weights", "inflect_nano_v1_acoustic.pt")
    vocoder = os.path.join(path, "weights", "inflect_nano_v1_vocoder.pt")
    available = os.path.exists(acoustic) and os.path.exists(vocoder)
    if available:
        print(f"[TTS] Inflect-Nano-v1 found at {path}")
    return available


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
    engine: str = "edge-tts"  # Options: "edge-tts", "pyttsx3", "kokoro"
    #voice: str = "en-US-ChristopherNeural"  # Default Edge TTS voice
    voice: str = "en-IN-NeerjaNeural" #en-IN-PrabhatNeural en-IN-NeerjaNeural
    rate: str = "+0%"  # Speech rate adjustment
    volume: float = 1.0  # Volume (0.0 to 1.0)
    output_dir: str = UPLOAD_DIR
    inflect_nano_model_path: str = "models/Inflect-Nano-v1"  # Path to cloned HF repo
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
            inflect_nano_model_path=settings_dict.get('inflect_nano_model_path', 'models/Inflect-Nano-v1'),
            kokoro_lang=settings_dict.get('kokoro_lang', 'a'),
            kokoro_device=settings_dict.get('kokoro_device', 'cpu'),
            kokoro_volume=float(settings_dict.get('kokoro_volume', 1.0)),
            kokoro_speed=float(settings_dict.get('kokoro_speed', 1.0))
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

        # Inflect-Nano model cache (loaded once, reused across calls)
        self._inflect_nano_acoustic = None
        self._inflect_nano_vocoder = None
        self._inflect_nano_speakers = None
        self._inflect_nano_device = None
    
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

        # Kokoro and Inflect-Nano output WAV format
        actual_format = "wav" if self.config.engine in ("kokoro", "inflect-nano") else output_format

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
        output_format: str = "mp3"
    ) -> Dict[str, Any]:
        """Generate speech audio from text."""
        if not text.strip():
            return {"success": False, "error": "No text provided"}

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
        elif engine == "pyttsx3" and HAS_PYTTSX3:
            return await self._generate_with_pyttsx3(text, filepath)
        elif engine == "kokoro" and _check_kokoro_available():
            return await self._generate_with_kokoro(text, voice, filepath)
        elif engine == "inflect-nano" and _check_inflect_nano_available(self.config.inflect_nano_model_path):
            return await self._generate_with_inflect_nano(text, filepath)

        # User explicitly chose an engine but it's not available → tell them, don't silently fall back
        if engine in ("edge-tts", "pyttsx3", "kokoro", "inflect-nano"):
            return {"success": False, "error": f"TTS engine '{engine}' is selected but not available. Check Settings → TTS."}

        # Auto-fallback when no engine is explicitly selected (shouldn't happen, but safe)
        if HAS_EDGE_TTS:
            return await self._generate_with_edge_tts(text, voice, rate, filepath)
        elif _check_kokoro_available():
            return await self._generate_with_kokoro(text, voice, filepath)
        elif _check_inflect_nano_available(self.config.inflect_nano_model_path):
            return await self._generate_with_inflect_nano(text, filepath)
        else:
            return {"success": False, "error": "No TTS engine available. Install edge-tts, pyttsx3, inflect-nano, or kokoro."}
    
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

            # Apply volume adjustment if needed (using kokoro-specific volume)
            if self.config.kokoro_volume != 1.0:
                print(f"Applying Kokoro volume adjustment: {self.config.kokoro_volume}")
                audio_data = audio_data * self.config.kokoro_volume

            # Apply speed adjustment if needed
            if self.config.kokoro_speed != 1.0:
                print(f"Applying Kokoro speed adjustment: {self.config.kokoro_speed}")
                try:
                    import librosa
                    # Resample audio to change speed (pitch-preserving)
                    # Speed up = higher sample rate output, then resample back to original
                    speed_factor = self.config.kokoro_speed
                    # Stretch/compress audio using time-stretch
                    audio_data = librosa.effects.time_stretch(audio_data.astype(np.float32), rate=speed_factor)
                    print(f"Speed adjustment applied using librosa")
                except ImportError:
                    # If librosa is not available, use scipy
                    try:
                        from scipy.interpolate import interp1d
                        original_length = len(audio_data)
                        target_length = int(original_length / self.config.kokoro_speed)
                        
                        if target_length != original_length:
                            # Create interpolation function
                            x_original = np.linspace(0, 1, original_length)
                            x_target = np.linspace(0, 1, target_length)
                            f = interp1d(x_original, audio_data, kind='linear', fill_value='extrapolate')
                            audio_data = f(x_target)
                            print(f"Speed adjustment applied using scipy")
                    except ImportError:
                        # If neither librosa nor scipy is available, warn but continue without speed adjustment
                        print("Warning: Speed adjustment requires librosa or scipy. Continuing without speed adjustment.")

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

    async def _generate_with_inflect_nano(self, text: str, filepath: str) -> Dict[str, Any]:
        """Generate speech using Inflect-Nano-v1 (ultra-small local TTS, ~4.6M params).

        Models are cached in the service instance so they are only loaded once.
        """
        import sys
        import soundfile as sf
        import numpy as np
        import torch
        from pathlib import Path

        model_path = Path(self.config.inflect_nano_model_path).resolve()
        weights_dir = model_path / "weights"
        acoustic_path = weights_dir / "inflect_nano_v1_acoustic.pt"
        vocoder_path = weights_dir / "inflect_nano_v1_vocoder.pt"

        if not acoustic_path.exists() or not vocoder_path.exists():
            return {"success": False, "error": f"Inflect-Nano model not found at {model_path}. Clone: git clone https://huggingface.co/owensong/Inflect-Nano-v1 {model_path}"}

        # Add repo dirs to sys.path for vendored imports (idempotent: duplicates are harmless)
        for p in [str(model_path), str(model_path / "third_party" / "tiny_tts_frontend")]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from inflect_nano.text_cleaning import clean_tinytts_text
            from inflect_nano.vocoder import HifiGanGenerator, make_config
            from inflect_nano.acoustic import MicroFastSpeech, MicroFastSpeechConfig
            from tiny_tts.text import phonemes_to_ids
            from tiny_tts.text.english import grapheme_to_phoneme, normalize_text
            from tiny_tts.nn import commons
            from tiny_tts.utils import ADD_BLANK
        except ImportError as e:
            return {"success": False, "error": f"Inflect-Nano import error: {e}. Make sure the repo is cloned."}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            loop = asyncio.get_event_loop()

            def _generate():
                # Load models into cache if not already loaded for this device
                cache_key = (str(acoustic_path), str(vocoder_path), str(device))
                if (self._inflect_nano_acoustic is None
                        or self._inflect_nano_cache_key != cache_key):
                    acoustic_ckpt = torch.load(str(acoustic_path), map_location=device, weights_only=False)
                    acoustic_cfg = MicroFastSpeechConfig(**acoustic_ckpt["config"])
                    self._inflect_nano_acoustic = MicroFastSpeech(acoustic_cfg).to(device)
                    self._inflect_nano_acoustic.load_state_dict(acoustic_ckpt["model"])
                    self._inflect_nano_acoustic.eval()

                    vocoder_ckpt = torch.load(str(vocoder_path), map_location=device, weights_only=False)
                    vocoder_cfg = make_config((vocoder_ckpt.get("config") or {}).get("variant", "snake_v2mid"))
                    self._inflect_nano_vocoder = HifiGanGenerator(vocoder_cfg).to(device)
                    self._inflect_nano_vocoder.load_state_dict(vocoder_ckpt["generator"])
                    self._inflect_nano_vocoder.remove_weight_norm()
                    self._inflect_nano_vocoder.eval()

                    self._inflect_nano_speakers = acoustic_ckpt.get("speakers") or {"mark": 0}
                    self._inflect_nano_device = str(device)
                    self._inflect_nano_cache_key = cache_key
                    print("[TTS] Inflect-Nano models loaded and cached")

                acoustic = self._inflect_nano_acoustic
                vocoder = self._inflect_nano_vocoder
                speakers = self._inflect_nano_speakers

                # Text processing
                cleaned = clean_tinytts_text(text)
                normalized = normalize_text(cleaned)
                phones, tones, _ = grapheme_to_phoneme(normalized)
                phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")

                if ADD_BLANK:
                    phone_ids = commons.insert_blanks(phone_ids, 0)
                    tone_ids = commons.insert_blanks(tone_ids, 0)
                    lang_ids = commons.insert_blanks(lang_ids, 0)

                phone_t = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
                tone_t = torch.LongTensor(tone_ids).unsqueeze(0).to(device)
                lang_t = torch.LongTensor(lang_ids).unsqueeze(0).to(device)
                speaker_t = torch.LongTensor([int(next(iter(speakers.values())))]).to(device)

                # Synthesize mel -> waveform
                with torch.inference_mode():
                    mel = acoustic.infer(phone_t, tone_t, lang_t, speaker_t,
                                         length_scale=1.0, pitch_scale=1.0, energy_scale=1.0)
                    wav = vocoder(mel).squeeze().detach().cpu().numpy()

                # Normalize
                wav = wav - wav.mean()
                peak = float(np.max(np.abs(wav)) + 1e-9)
                if peak > 0.95:
                    wav *= 0.95 / peak
                wav = np.clip(wav, -1.0, 1.0)

                sf.write(str(filepath), wav, 24000, subtype="PCM_16")

            await loop.run_in_executor(None, _generate)

            return {
                "success": True,
                "filepath": filepath,
                "audio_url": f"/api/audio/{os.path.basename(filepath)}",
                "engine": "inflect-nano",
                "voice": "default (male)"
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"success": False, "error": f"Inflect-Nano error: {str(e)}"}

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
        elif self.config.engine == "inflect-nano" and _check_inflect_nano_available(self.config.inflect_nano_model_path):
            voices.append({
                "id": "default",
                "name": "Default (Male, US)",
                "gender": "male",
                "locale": "en-US"
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

        engines.append({
            "id": "inflect-nano",
            "name": "Inflect-Nano v1",
            "description": "Ultra-small local TTS (~4.6M params) - clone HF repo to models/Inflect-Nano-v1",
            "available": _check_inflect_nano_available()
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

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
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
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
    normalize_enabled: bool = True  # Run the structural/numeric cleanup before synthesis
    # User-defined rules applied last in normalize_tts_text. Each entry is a
    # dict {"pattern": str, "flags": str, "replacement": str}; compile errors
    # are skipped at apply time so one bad rule does not break TTS.
    custom_replacements: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.custom_replacements is None:
            self.custom_replacements = []

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
            kokoro_speed=float(settings_dict.get('kokoro_speed', 1.0)),
            normalize_enabled=bool(settings_dict.get('tts_normalize_enabled', True)),
            custom_replacements=settings_dict.get('tts_custom_replacements') or [],
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


def _bare_number_to_words(m):
    """Rewrite a bare number for TTS. The default 'one thousand nine hundred
    and sixty four' sounds wrong for years; TTS engines also sometimes read
    4-digit years that way. The standard English convention is to split the
    year as 'nineteen sixty four' / 'two thousand four'. A bare 4-digit
    number in 1000–2099 is treated as a year by default (see
    _looks_like_year) unless an explicit reference prefix ("page 1207",
    "line 1812") marks it as a cardinal.

    Already-decorated cases (money, percent, ordinal, glued-to-unit) are
    handled by their own rules above, so this only sees plain integers.
    """
    from num2words import num2words
    raw = m.group(1)
    plural = m.group(2).startswith("s")
    # Decimals ("0.1", "3.14") — read them point-style, "zero point one" /
    # "three point one four". int() below would crash with
    # 'invalid literal for int() with base 10' and silently kill the
    # sentence's TTS.
    if "." in raw:
        whole, _, frac = raw.partition(".")
        whole_w = num2words(int(whole or "0")).replace(",", "") if whole else "zero"
        frac_w = " ".join(num2words(int(d)).replace(",", "") for d in frac)
        return f"{whole_w} point {frac_w}"
    n = int(raw.replace(",", ""))
    is_year = _looks_like_year(m.string, m.start())
    if is_year:
        spoken = _year_to_words(n)
        if plural:
            # Plural-year shorthand: 1990s -> "nineteen nineties",
            # 2000s -> "two thousands", 1900s -> "nineteen hundreds".
            if n % 100 == 0:
                # 1900s, 2000s, 1800s — drop the trailing "hundred"/"thousand"
                # and add s.
                spoken = spoken.replace(" hundred", " hundreds").replace(" thousand", " thousands")
                if " hundreds" not in spoken and " thousands" not in spoken:
                    spoken = spoken + "s"
            else:
                # Decade plural: 1990s -> "nineteen nineties", 1980s -> "nineteen
                # eighties", 2010s -> "twenty tens". Build from the last two
                # digits of n so we don't have to parse the spoken form. The
                # prefix is the century without the trailing "hundred" /
                # "thousand" (we don't say "nineteen hundred nineties").
                century_val = (n // 100) * 100
                decade_last2 = n % 100
                decade = _decade_plural(decade_last2)
                if 1900 <= century_val <= 1999:
                    prefix_words = "nineteen"
                elif 2000 <= century_val <= 2099:
                    prefix_words = "two thousand" if decade_last2 == 0 else "twenty"
                elif 1800 <= century_val <= 1899:
                    prefix_words = "eighteen"
                elif 1700 <= century_val <= 1799:
                    prefix_words = "seventeen"
                elif 1600 <= century_val <= 1699:
                    prefix_words = "sixteen"
                elif 1500 <= century_val <= 1599:
                    prefix_words = "fifteen"
                elif 1400 <= century_val <= 1499:
                    prefix_words = "fourteen"
                elif 1300 <= century_val <= 1399:
                    prefix_words = "thirteen"
                elif 1200 <= century_val <= 1299:
                    prefix_words = "twelve"
                elif 1100 <= century_val <= 1199:
                    prefix_words = "eleven"
                elif 1000 <= century_val <= 1099:
                    prefix_words = "ten"
                else:
                    prefix_words = _year_to_words(century_val)
                spoken = f"{prefix_words} {decade}"
        return spoken
    return num2words(n).replace(",", "")


def _looks_like_year(text: str, pos: int) -> bool:
    """Is the number at text[pos:] a year?

    Default answer for a bare 4-digit number in 1000–2099 is YES — in
    flowing prose that is overwhelmingly a year ("the 1964 edition"), and
    even for street numbers ("1812 Maple St") the split reading "eighteen
    twelve" is what people actually say. The only counter-signal is an
    explicit reference prefix ("page 1207", "line 1912", "chapter 1815")
    where the cardinal is correct. Counts with a thousands comma
    ("1,812 people") and decimals never take the year path.
    """
    m = re.match(r"\d[\d,]*", text[pos:])
    if not m:
        return False
    raw = m.group(0)
    # A thousands comma ("1,812 people") marks a count, not a year; a
    # decimal is a measurement. Both stay on the cardinal path.
    if "," in raw or "." in raw:
        return False
    try:
        n = int(raw)
    except ValueError:
        return False
    if n < 1000 or n > 2099:
        return False
    prefix = text[max(0, pos - 40):pos].rstrip()
    last = re.search(r"(\S+)$", prefix)
    if not last:
        # Number at the very start of the text/sentence — "1964 was ...".
        return True
    prev = last.group(1).lower().rstrip(".")
    if prev in _NOT_YEAR_PREFIXES:
        return False
    return True


# Words that put a following 4-digit number in an explicit non-year
# context (page, line, figure, table, chapter, volume, ... references).
# With these, "1207" reads as "one thousand two hundred seven" — the
# cardinal — instead of the split-year "twelve oh seven". Everything
# else in 1000–2099 is treated as a year.
_NOT_YEAR_PREFIXES = frozenset({
    "page", "pages", "p", "pp", "pg", "pgs",
    "line", "lines", "row", "rows", "column", "columns",
    "figure", "figures", "fig", "figs",
    "table", "tables",
    "chapter", "chapters", "section", "sections", "appendix",
    "equation", "equations", "eq", "eqn", "eqns",
    "step", "steps", "rule", "rules", "item", "items",
    "entry", "entries", "no", "nos", "number", "numbers",
    "vol", "vols", "volume", "volumes", "issue", "issues",
    "id", "ids", "serial", "model", "part", "parts", "code",
})


def _year_to_words(n: int) -> str:
    """Read a year as humans say it: 1964 -> 'nineteen sixty four',
    1812 -> 'eighteen twelve', 2024 -> 'twenty twenty four'."""
    from num2words import num2words
    if n < 1000 or n > 2099:
        return num2words(n)
    century, year = divmod(n, 100)
    # 20xx: 2000-2009 -> "two thousand N"; 2010-2099 -> "twenty N N"
    if 2000 <= n <= 2009:
        return "two thousand" if year == 0 else f"two thousand {num2words(year)}"
    if 2010 <= n <= 2099:
        return f"twenty {num2words(year)}" if year >= 10 else f"twenty oh {num2words(year)}"
    # 10xx-19xx: "<century-word> <year>"
    cent_word = num2words(century)
    if year == 0:
        return cent_word + " hundred"
    if year < 10:
        return f"{cent_word} oh {num2words(year)}"
    # 10-99 just reads as a number (twelve, sixty four, ninety two, etc.)
    return f"{cent_word} {num2words(year)}"


# Plural-form mapping for decade-year shorthand: 1990s -> "nineties".
_DECADE_PLURALS = {
    10: "tens", 20: "twenties", 30: "thirties", 40: "forties", 50: "fifties",
    60: "sixties", 70: "seventies", 80: "eighties", 90: "nineties",
    1: "ones", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes",
    7: "sevens", 8: "eights", 9: "nines", 0: "noughts",
}


def _decade_plural(last_two: int) -> str:
    if last_two in _DECADE_PLURALS:
        return _DECADE_PLURALS[last_two]
    return f"{last_two}s"


# ─── Non-numeric TTS text cleanup ──────────────────────────────
# Engines (esp. Kokoro/misaki, but Edge too) read literal punctuation and
# symbols aloud: "~" -> "tilde", URLs spelled out char-by-char, "->" read as
# "hyphen greater", etc. The rewrites below are conservative — they match what
# a human would say — so they are safe to run before ANY engine.

# Symbols that should become words.
_SYMBOL_WORDS = {
    "\u2192": " to ",      # →
    "\u2190": " to ",      # ←
    "\u2194": " versus ",  # ↔
    "\u00b1": " plus or minus ",  # ±
    "\u00d7": " times ",   # ×
    "\u00f7": " divided by ",     # ÷
    "\u2248": " approximately ",   # ≈
    "\u2264": " less than or equal to ",  # ≤
    "\u2265": " greater than or equal to ",  # ≥
    "\u2260": " not equal to ",   # ≠
    "\u00b0": " degrees ",  # °
}

# Acronyms Kokoro/misaki mangle into nonsense ("appy", "jay-son"). Edge reads
# them fine, so only spell them out letter-by-letter for Kokoro.
_KOKORO_ACRONYMS = {
    "API": "A P I", "APIs": "A P I s", "URL": "U R L", "URLs": "U R L s",
    "SQL": "S Q L", "JSON": "J S O N", "XML": "X M L", "HTML": "H T M L",
    "CSS": "C S S", "HTTP": "H T T P", "HTTPS": "H T T P S",
}

# Glued number+unit tokens. Expanded to words; the digit is then converted to
# words by the numeric pass below.
_UNIT_WORDS = {
    "km": " kilometers", "cm": " centimeters", "mm": " millimeters",
    "m": " meters", "px": " pixels", "gb": " gigabytes", "mb": " megabytes",
    "kb": " kilobytes", "tb": " terabytes", "ms": " milliseconds",
    "kg": " kilograms", "ml": " milliliters", "hz": " hertz",
    "ghz": " gigahertz", "mhz": " megahertz",
}

# Emoji / symbol pictographs to drop entirely (engines read them as
# descriptions like "red heart emoji" or skip them).
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\ufe0f]"
)


def apply_custom_replacements(text: str, rules) -> str:
    """Apply user-defined regex rules. Compile errors are skipped per rule so a
    single bad pattern does not break TTS for the whole message.

    ponytail: re.error caught per-rule; rule surface stays simple. If the user
    later wants a single 'validate all' UI action, compile and cache once on
    save instead of per-call.
    """
    if not rules:
        return text
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        pat = rule.get("pattern")
        rep = rule.get("replacement", "")
        if not pat:
            continue
        flags = 0
        for ch in (rule.get("flags") or "").lower():
            if ch == "i": flags |= re.IGNORECASE
            elif ch == "m": flags |= re.MULTILINE
            elif ch == "s": flags |= re.DOTALL
            elif ch == "x": flags |= re.VERBOSE
        try:
            text = re.sub(pat, rep, text, flags=flags)
        except re.error as e:
            print(f"[TTS] skipping bad custom rule {pat!r}: {e}")
    return text


def normalize_tts_text(text: str, engine: Optional[str] = None,
                       normalize_enabled: bool = True,
                       custom_replacements: Optional[list] = None) -> str:
    """Rewrite text so any TTS engine reads it the way a human would.

    Two layers:
      1. Structural/semantic cleanup (URLs, links, ~, &, symbols, units,
         emoji) — engine-agnostic, conservative.
      2. Numeric cleanup ($/%/ordinals/bare numbers) — required for Kokoro,
         harmless for Edge.
    Acronym spelling-out is gated to Kokoro only.
    """
    if not text:
        return text

    # Kill-switch: skip the structural/numeric cleanup entirely.
    # Custom rules still run below so users can opt in to only their rewrites.
    if not normalize_enabled:
        return apply_custom_replacements(text, custom_replacements)

    # 1. Decode HTML entities first so &amp; -> & before the &->"and" rule.
    import html as _html
    text = _html.unescape(text)

    # 2. Markdown links [text](url) -> text (URL would otherwise be read aloud).
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # 2b. Bracketed citation / reference markers — [12], [1, 2], [3-5] —
    #     from academic PDFs, footnotes and bibliography refs. No TTS
    #     engine should read "twelve" for a footnote superscript that
    #     PyPDF2 / PDF.js flattened into the prose. The bracketed group is
    #     digits + separators only, so "[text]" and arrays like
    #     "[foo, bar]" are untouched.
    text = re.sub(r"\[\s*\d+(?:\s*[,;–—-]\s*\d+)*\s*\]", "", text)
    #     Removing "[12]" can leave "in ." — pull punctuation back.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # 3. URLs: drop scheme + leading www, and any /path?query, leaving the host.
    text = re.sub(r"https?://", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\S)www\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([\w-]+(?:\.[\w-]+)+)(/[^\s]*)", r"\1", text)

    # 4. & -> "and" (entity already decoded above).
    text = re.sub(r"\s*&\s*", " and ", text)

    # 5. ~ : approximate before a number; home-dir before a slash; else leave.
    text = re.sub(r"~\s*(\d)", r"approximately \1", text)
    text = re.sub(r"~/", "home ", text)

    # 6. #N (not a hex color) -> "number N".
    text = re.sub(r"#(\d+)(?![0-9a-fA-F])", r"number \1", text)

    # 7. Unicode symbol -> word.
    for sym, word in _SYMBOL_WORDS.items():
        text = text.replace(sym, word)

    # 8. Glued number+unit -> "5 kilometers" (digit worded below).
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*(km|cm|mm|px|gb|mb|kb|tb|ms|kg|ml|hz|ghz|mhz)",
        lambda m: f"{m.group(1)}{_UNIT_WORDS[m.group(2).lower()]}",
        text,
        flags=re.IGNORECASE,
    )

    # 9. Acronyms: spell out only for Kokoro (Edge handles them).
    if engine == "kokoro":
        for acr, spelled in _KOKORO_ACRONYMS.items():
            text = re.sub(rf"\b{acr}\b", spelled, text)

    # 10. Drop emoji/pictographs.
    text = _EMOJI_RE.sub("", text)

    # 11. Numeric cleanup (existing behavior, kept verbatim).
    try:
        from num2words import num2words
    except ImportError:
        # Without num2words we can't word numbers; return the structural cleanup.
        return re.sub(r"\s+", " ", text).strip()

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
    # Match a number optionally followed by a plural 's' (for 'the 1990s') so
    # we can speak it as 'nineteen nineties'.
    text = re.sub(
        r"(?<![\w:./-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(s?\b)(?![\w:./%-])",
        _bare_number_to_words, text)

    # 12. Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # 13. User-defined rules run last so they can rewrite any of the above
    #     output (e.g. swap a specific word to a pronunciation hint).
    return apply_custom_replacements(text, custom_replacements)


def _is_abbrev_fragment(p: str) -> bool:
    return len(p) <= 6 and p.endswith('.') and not re.search(r"[^A-Za-z.]", p)


def _split_sentences(text: str) -> List[str]:
    # Split on terminal punctuation or paragraph breaks (blank lines).
    # Note: we intentionally do NOT split on single \n — the upstream
    # _extract_pdf collapses soft-wraps to spaces before we get here, so
    # single newlines that survive are real paragraph breaks and are
    # already handled by the \n{2,} alternative.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip()) if p.strip()]
    out, buf = [], ""
    for p in parts:
        if _is_abbrev_fragment(p):
            buf = (buf + " " + p).strip()
        else:
            out.append((buf + " " + p).strip())
            buf = ""
    if buf:
        out.append(buf)
    return out


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
        text = normalize_tts_text(
            text,
            engine=self.config.engine,
            normalize_enabled=self.config.normalize_enabled,
            custom_replacements=self.config.custom_replacements,
        )

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

    async def stream_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None
    ) -> AsyncGenerator[tuple, None]:
        """Async generator: yield (filename, audio_url) per sentence as each is generated.

        edge-tts: one segment (whole text — cloud synth is fast enough).
        kokoro: one segment per sentence, each cached by sentence hash, so
        re-reads of the same message stream instantly from cache. Generation
        bails between sentences once should_stop() is true (client paused).
        """
        text = normalize_tts_text(
            text,
            engine=self.config.engine,
            normalize_enabled=self.config.normalize_enabled,
            custom_replacements=self.config.custom_replacements,
        )
        voice = voice or self.config.voice
        rate = rate or self.config.rate
        engine = self.config.engine

        if engine == "edge-tts" and HAS_EDGE_TTS:
            filename = self._get_cache_filename(text, voice, rate, "mp3")
            filepath = os.path.join(self.config.output_dir, filename)
            if not os.path.exists(filepath):
                await self._generate_with_edge_tts(text, voice, rate, filepath)
            yield filename, f"/api/audio/{filename}"
            return

        if engine != "kokoro" or not _check_kokoro_available():
            return

        import soundfile as sf
        import numpy as np

        pipeline = self._get_kokoro_pipeline()
        if pipeline is None:
            return

        if voice not in KOKORO_VOICES:
            voice = "bf_emma" if self.config.kokoro_lang == "b" else "af_bella"

        speed = self.config.kokoro_speed
        loop = asyncio.get_running_loop()

        for sentence in _split_sentences(text):
            if should_stop and should_stop():
                return
            filename = self._get_cache_filename(sentence, voice, rate, "wav")
            filepath = os.path.join(self.config.output_dir, filename)
            if not os.path.exists(filepath):
                def _gen():
                    audio_parts = []
                    for _, _, audio in pipeline(sentence, voice=voice, speed=speed):
                        if should_stop and should_stop():
                            return None
                        audio_parts.append(audio)
                    if not audio_parts:
                        return None
                    return np.concatenate(audio_parts)
                try:
                    audio_data = await loop.run_in_executor(None, _gen)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self._kokoro_pipeline = None
                        pipeline = self._get_kokoro_pipeline()
                        if pipeline is None:
                            return
                        audio_data = await loop.run_in_executor(None, _gen)
                    else:
                        raise
                if audio_data is None:
                    if should_stop and should_stop():
                        return
                    continue
                if self.config.kokoro_volume != 1.0:
                    audio_data = audio_data * self.config.kokoro_volume
                sf.write(filepath, audio_data, 24000)
            yield filename, f"/api/audio/{filename}"

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
        "The rate was 0.1 per year.": "The rate was zero point one per year.",
        "All 3.5 stars!": "All three point five stars!",
        # Years: split reading, even without an indicator word.
        "The 1964 edition was revised.": "The nineteen sixty-four edition was revised.",
        "1964 was a turning point.": "nineteen sixty-four was a turning point.",
        "From 1955 to 1960 the map changed.": "From nineteen fifty-five to nineteen sixty the map changed.",
        "In the 1990s it grew.": "In the nineteen nineties it grew.",
        # Non-year 4-digit numbers stay cardinal (num2words uses "and").
        "See page 1207 for details.": "See page one thousand two hundred and seven for details.",
        "Line 1812 broke.": "Line one thousand eight hundred and twelve broke.",
        "1,812 people attended.": "one thousand eight hundred and twelve people attended.",
        # Bracketed citation markers dropped before TTS.
        "The effect was noted in [12].": "The effect was noted in.",
        "Doe et al. [1, 2] showed it.": "Doe et al. showed it.",
        "See [3-5] for details.": "See for details.",
        "12:30 PM": "12:30 PM",  # untouched: time
        "12/25/2024": "12/25/2024",  # untouched: date
        "10-20 items": "10-20 items",  # untouched: range
        "5km away": "five kilometers away",  # glued unit -> words
        "Visit https://www.google.com": "Visit google.com",
        "See [Google](https://google.com)": "See Google",
        "~30 minutes": "approximately thirty minutes",
        "AT&T": "AT and T",
        "A \u2192 B": "A to B",
        "#1 issue": "number one issue",
        "hi \U0001F44D there": "hi there",
    }
    for inp, expected in cases.items():
        got = normalize_tts_text(inp)
        assert got == expected, f"{inp!r} -> {got!r}, expected {expected!r}"
    # Engine-specific: Kokoro spells out acronyms; Edge (and default) does not.
    assert normalize_tts_text("call the API", engine="kokoro") == "call the A P I"
    assert normalize_tts_text("call the API", engine="edge-tts") == "call the API"
    assert normalize_tts_text("call the API") == "call the API"

    # User-defined rules: word/phrase swap, regex with backref, kill-switch,
    # and a malformed pattern that must be skipped (not raise).
    rules = [
        {"pattern": r"\bDr\.\s*", "replacement": "Doctor "},
        {"pattern": r"(\d+)\s*°\s*C", "flags": "i", "replacement": r"\1 degrees Celsius"},
        {"pattern": r"GPT", "replacement": "G P T"},
        {"pattern": r"[bad", "replacement": "x"},  # malformed — must be skipped
    ]
    assert normalize_tts_text("Dr. Smith says 20°C and GPT-4", custom_replacements=rules) == \
        "Doctor Smith says twenty degrees C and G P T-4"
    # Kill-switch leaves structural cleanup untouched; custom rules still apply.
    assert normalize_tts_text("$5 and Dr. Foo", normalize_enabled=False, custom_replacements=rules) == \
        "$5 and Doctor Foo"
    # No rules, kill-switch off -> untouched.
    assert normalize_tts_text("$5 and Dr. Foo", normalize_enabled=False) == "$5 and Dr. Foo"
    print("normalizer self-check: OK")

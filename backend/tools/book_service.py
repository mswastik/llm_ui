"""Book extraction for the read-aloud reader.

Two formats:
  - PDF: reuse DocumentProcessor._extract_pdf (PyPDF2) — already in the
    dependency tree, no new import.
  - EPUB: lazy-import ebooklib; if missing, raise a clear error at upload
    time so the user knows what to install. (Adding it as a hard dep would
    bloat every install for users who never read EPUBs.)

Both produce the same shape: a list of sentences with their source page,
which the stream endpoint walks sentence-by-sentence to drive TTS + the
reader's highlight.

We deliberately reuse TTSService._split_sentences (the same sentence
boundaries the audio will use) so the highlight always lands on exactly
what the TTS is reading — no off-by-one between text and audio.
"""
import re
from typing import Any, Dict, List, Tuple

# Reuse the PDF extractor we already have — no fork
from tools.rag_service import DocumentProcessor
# Reuse the sentence splitter the chat TTS uses — keeps text/audio aligned
from tools.tts_service import TTSService
# Reuse the module-level sentence splitter; calling it on TTSService itself
# would AttributeError (it's not a method).
from tools import tts_service as _tts_mod


def _extract_epub(path: str) -> List[str]:
    """Walk an EPUB's spine in order, return one string per 'page' (chapter).
    A real EPUB reader renders CSS — for read-aloud we only need the text, in
    reading order, with the book split into a reasonable number of segments.
    The spine gives us chapters; we use one chapter as one 'page' so the
    highlight-to-page mapping has useful granularity."""
    try:
        from ebooklib import epub
    except ImportError:
        raise ValueError(
            "EPUB support requires ebooklib — install it (pip install ebooklib)"
        )

    book = epub.read_epub(path)
    chapters: List[str] = []
    for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
        # body content; strip HTML tags to plain text. lxml would be cleaner
        # but adds a dep; regex is fine for text-only rendering.
        from lxml import html as _lhtml  # ebooklib pulls this in transitively
        try:
            tree = _lhtml.fromstring(item.get_content())
            text = " ".join(t.strip() for t in tree.itertext() if t.strip())
        except Exception:
            text = re.sub(r"<[^>]+>", " ", item.get_content().decode("utf-8", errors="ignore"))
        if text:
            chapters.append(text)
    if not chapters:
        raise ValueError("EPUB contains no readable text")
    return chapters


def _split_into_sentences(page_text: str) -> List[str]:
    """Drop the existing public visibility and call the chat-TTS splitter so
    the reader's highlight lines up with what the TTS actually reads out.
    Ponytail: one source of truth for sentence boundaries; if the chat TTS
    later changes its rule, the reader inherits the change for free."""
    return _tts_mod._split_sentences(page_text)


def _is_broken_chunk(s: str) -> bool:
    """PyPDF2's print-typeset extraction can produce sentences where every
    word is split into 1-4 char chunks separated by single spaces
    ('B ut of c our set he y know t ha tt he ir'). Such sentences sound
    terrible when TTSed. Drop them — the reader will jump to the next
    clean sentence and the user hears a brief silence instead of garbled
    audio.

    Heuristic: 8+ whitespace tokens, all ≤ 4 chars, no token ≥ 8 chars
    (so any real long word saves the sentence). Short real sentences
    ('I am a man', 'Hi there') are safe — they're under 8 tokens.
    """
    tokens = s.split()
    if len(tokens) < 8:
        return False
    if any(len(t) >= 8 for t in tokens):
        return False
    short = sum(1 for t in tokens if len(t) <= 4)
    return short / len(tokens) >= 0.8


def extract(path: str, file_type: str) -> Dict[str, Any]:
    """Extract sentences + page map from a PDF or EPUB.

    Returns:
      {
        "sentences": [{text, page, char_start}, ...],   # char_start is reserved for future use
        "page_map": [page_for_idx_0, page_for_idx_1, ...],
      }

    For EPUB, "page" is the chapter index (1-based), since the EPUB spec
    has no fixed page boundaries.
    """
    if file_type == "pdf":
        page_texts, _joined = DocumentProcessor._extract_pdf(path)
    elif file_type == "epub":
        page_texts = _extract_epub(path)
    else:
        raise ValueError(f"Unsupported book file type: {file_type}")

    sentences: List[Dict[str, Any]] = []
    page_map: List[int] = []

    for page_num, page_text in enumerate(page_texts, start=1):
        # Track per-page char offset so the highlight window in the future
        # can scroll the original PDF to the right line.
        char_offset = 0
        for sent in _split_into_sentences(page_text):
            # Drop sentences that PyPDF2 garbled into 1-4 char chunks —
            # they'd sound terrible when TTSed.
            if _is_broken_chunk(sent):
                continue
            # Find this sentence's start char in the page (best-effort, may
            # be approximate after TTS normalization drops whitespace)
            idx = page_text.find(sent[:40], char_offset)
            if idx < 0:
                idx = char_offset
            sentences.append({"text": sent, "page": page_num, "char_start": idx})
            page_map.append(page_num)
            char_offset = idx + len(sent)

    if not sentences:
        raise ValueError("No readable text could be extracted from this file")
    return {"sentences": sentences, "page_map": page_map}


def derive_title(filename: str) -> str:
    """Strip extension + clean up the uploaded filename for display."""
    name = re.sub(r"\.(pdf|epub)$", "", filename, flags=re.IGNORECASE)
    return name.replace("_", " ").strip() or "Untitled"


if __name__ == "__main__":
    # Self-check: exercise the splitter + title logic without needing a
    # real PDF on disk. The PDF extraction itself is exercised by the
    # upload endpoint on a real user upload.
    import sys

    sample = "Hello world. This is page one. It has two sentences."
    sents = _split_into_sentences(sample)
    assert sents == ["Hello world.", "This is page one.", "It has two sentences."], \
        f"unexpected split: {sents}"

    # Multi-page extraction shape: list of page texts → sentences with page map
    page_texts = [
        "First page sentence one. First page sentence two.",
        "Second page starts. Another here. And a third.",
    ]
    all_sentences = []
    page_map = []
    for p_num, pt in enumerate(page_texts, start=1):
        for s in _split_into_sentences(pt):
            all_sentences.append({"text": s, "page": p_num, "char_start": 0})
            page_map.append(p_num)
    assert len(all_sentences) == 5
    assert page_map == [1, 1, 2, 2, 2]
    assert all_sentences[0]["page"] == 1
    assert all_sentences[2]["page"] == 2

    # Title derivation
    assert derive_title("The_Hobbit.pdf") == "The Hobbit"
    assert derive_title("book.epub") == "book"
    assert derive_title("") == "Untitled"

    # Unknown file type rejected
    try:
        extract("/nonexistent.txt", "txt")
        assert False, "should have raised"
    except ValueError as e:
        assert "Unsupported" in str(e)

    # Broken-chunk detection: print-typeset PDF artifacts
    assert _is_broken_chunk("B ut of c our set he y know t ha tt he ir pa rtic ul ar") is True
    assert _is_broken_chunk("w a s t h e b e s t t i m e") is True
    assert _is_broken_chunk("This is a normal sentence that should not be dropped.") is False
    assert _is_broken_chunk("Hi I am a man") is False
    assert _is_broken_chunk("Maybe yes maybe no") is False
    assert _is_broken_chunk("The quick brown fox jumps over the lazy dog") is False

    print(f"OK: splitter={len(sents)} sentences, page_map={page_map}, title derivation correct, broken-chunk filter correct")

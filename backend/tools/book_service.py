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

# Superscript / footnote / citation markers that extraction flattens
# into the prose when it pulls text from print-typeset PDFs (no
# font-size info, so a superscript "1" is indistinguishable from a body
# number). We strip these markers so TTS doesn't read "one" or
# "asterisk" between sentences. Three locations:
#
#   LEADING: a footnote reference in the body text that was rendered
#   as a superscript BEFORE the word it qualifies ("*1 The next...").
#   The asterisk IS the footnote indicator in some book styles (common
#   in academic / reference books).
#
#   TRAILING: a marker at the end of a sentence ("text.1", "text[3]",
#   "text *"). More common in narrative books. The star may also sit
#   directly BEFORE the number ("text.*12") or AFTER it ("text.12*")
#   depending on how the PDF flattened the superscript.
#
#   MID-ROW: mangled spacing defeats the sentence splitter, so one
#   stored row can hold "sentence.24 Next sentence..." or a mid-sentence
#   footnote star ("amygdala * is"). These need non-anchored patterns.
#
#   Gating rule: a bare digit run is only a citation when it sits AFTER
#   the sentence's terminal punctuation (".24", ". 24", ".24*"), and the
#   period itself must not extend a decimal ("4.3 Overview" keeps its
#   "3"). Digits BEFORE the period ("There were 12.", "rated 5*.",
#   "CO2.") are genuine prose and must survive — same for a digit run
#   with no period at all ("disputed 24", "5* hotels"). Star-FIRST
#   markers ("*12") need no gating: prose never ends that way, while
#   footnotes often do. A lone "*" is removed mid-row only when it
#   cannot be math ("2 * 3") or a rating ("5*").
#
# Patterns are deliberately narrow: bare digits mid-sentence ("I ate 3
# apples", "page 1") are left alone since they could be the actual
# content. Only marker-shaped patterns at sentence boundaries (or after
# terminal punctuation) are stripped.
#
# SUP: ASCII digits + unicode superscript / subscript digits (²⁴, ₂₄).
# Subscripts are included ONLY in the after-period cluster — "CO2" /
# "H2O" (digits before the period) must never be touched.
_SUP_DIGITS = r"[\d\u00B9\u00B2\u00B3\u2070-\u2079\u2080-\u2089]+"
# A cluster digit run continues across spaces/commas ("1 0", "24,25" —
# flattened multi-digit footnotes). Used after terminal punctuation.
_SUP_SEQ = rf"{_SUP_DIGITS}(?:\s*[,;]?\s*{_SUP_DIGITS})*"
# What may start a new sentence after a mid-row citation (capital
# letter, quote or bracket). Lowercase after digits is prose
# ("Stop! 3 times", "Fig. 2 shows this") and must survive.
_SENT_START = r"""[A-Z\"'\(\[\u201c\u201d]"""
_LEADING_CITATION_RE = re.compile(
    rf"""
    ^\s* (?:
          \**\[\d+(?:[\s,;\-–]*\d+)*\]\**  # [1]  [1, 2]  [1-3]  [3 0]
        | \*+\s*{_SUP_SEQ}                   # *1  **2  * 12  *3 0
        | \*\s+(?=[A-Z])                    # *  followed by capital (footnote in body)
        | [†‡§¶]                            # unicode footnote / pilcrow
        | [\u00B9\u00B2\u00B3\u2070-\u2079]+  # unicode superscript digits
    )
    \s*
    """,
    re.VERBOSE,
)

# Mid-row footnote star with its number ("life." *3 0", "mother, *2 3
# and"). Star-first is always marker-shaped, so no end anchor is needed —
# but it must not be math ("2 * 3"): block when digits precede the star.
_MID_STAR_DIGITS_RE = re.compile(
    rf"(?<!\d)(?<!\d\s)\*+\s*{_SUP_SEQ}",
    re.VERBOSE,
)

# Mid-row lone footnote star ("amygdala * is", "; * when", "prosocial*—less").
# Same math/rating guard; must be followed by space, punctuation, quote
# or dash (so glued emphasis like "M*A*S*H" and code like "a*b" survive).
_MID_LONE_STAR_RE = re.compile(
    r"""(?<!\d)(?<!\d\s)\*+(?=[\s.,;:!?\"'()\[\]\u201c\u201d\u2014\u2013])""",
    re.VERBOSE,
)

_TRAILING_CITATION_RE = re.compile(
    rf"""
    (?:
          \s*(?:
               \**\[\d+(?:[\s,;\-–]*\d+)*\]\**  # [1]  [1, 2]  [1-3]  [3 0]
             | (?<!\d)(?<!\d\s)\*+              # lone stars — never off a
                                               # rating ("rated 5*") or math
             | [†‡§¶]                          # dagger / pilcrow / etc.
             | \^\d+                           # ^1
          )\s*$
        | (?<=[^\d][.!?])\s*\**\s*{_SUP_SEQ}\s*\**
          (?=\s+{_SENT_START}|\s*$)  # citation cluster AFTER terminal
                                     # punctuation: .24  . 24  .*12  .12*
                                     # .24,25  .²⁴ — to the end, or into a
                                     # capital-starting next sentence
                                     # (".2 Now", ".1 0 In"). The period
                                     # must not extend a decimal ("4.3").
        | (?<=\D[12]\d{{3}}[.!?])\s*\**\s*{_SUP_SEQ}\s*\**
          (?=\s+{_SENT_START}|\s*$)  # ...unless it extends a 4-digit year:
                                     # years never take decimals, so
                                     # "in 2008.1 The" is year + footnote.
    )
    """,
    re.VERBOSE,
)


def _strip_citations(sent: str) -> str:
    # Strip leading marker first (footnote refs in body text), then
    # mid-row stars (star+digits before lone stars, so "*2 3" goes as
    # one marker instead of leaving "2 3" behind), then trailing.
    sent = _LEADING_CITATION_RE.sub("", sent)
    sent = _MID_STAR_DIGITS_RE.sub("", sent)
    sent = _MID_LONE_STAR_RE.sub("", sent)
    sent = _TRAILING_CITATION_RE.sub("", sent)
    # Mid-row removal can leave double spaces ("amygdala  is") — collapse
    # (whitespace is insignificant to TTS and the highlight).
    sent = re.sub(r" {2,}", " ", sent)
    # Stripping a marker can leave "disputed ." — pull punctuation back.
    sent = re.sub(r"\s+([.!?,;:])", r"\1", sent)
    return sent.rstrip()


# A sentence that is ONLY digits / citation punctuation ("12", "14.",
# "[3]", "²⁴") is a flattened superscript reference or a page number —
# PyPDF2 puts superscripts on their own line, and the sentence splitter
# then yields the bare number as its own sentence. Never real prose, so
# it must not reach the TTS engine.
_CITATION_ONLY_RE = re.compile(r"[\d\s,.;:()\[\]{}*†‡§¶%#\^\u00B9\u00B2\u00B3\u2070-\u2079\u2080-\u2089]+")


def _is_citation_only(sent: str) -> bool:
    return _CITATION_ONLY_RE.fullmatch((sent or "").strip()) is not None


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
            # Strip trailing footnote / citation markers (e.g. "text.1",
            # "text[3]", "text *") so TTS doesn't read them out loud.
            sent = _strip_citations(sent)
            # Drop standalone marker / page-number sentences ("12",
            # "14.") — PyPDF2 flattens superscript refs onto their own
            # line, and the splitter yields the bare number alone.
            if not sent or _is_citation_only(sent):
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

    # Citation-marker stripping. PyPDF2 flattens superscript footnote
    # markers into the prose text, so we strip them at the boundaries
    # of each sentence before TTS reads them out.
    assert _strip_citations("The cat sat on the mat.") == "The cat sat on the mat."
    assert _strip_citations("The cat sat on the mat.[3]") == "The cat sat on the mat."
    assert _strip_citations("The cat sat on the mat.1") == "The cat sat on the mat."
    assert _strip_citations("He whispered hello!23") == "He whispered hello!"
    assert _strip_citations("End of paragraph *") == "End of paragraph"
    assert _strip_citations("Footnote marker here^1") == "Footnote marker here"
    assert _strip_citations("He saw 3 cats.") == "He saw 3 cats."  # bare number untouched
    assert _strip_citations("Already at end[1, 2]") == "Already at end"
    assert _strip_citations("Quoth the raven †") == "Quoth the raven"
    # Leading footnote markers (common in academic books: "*1 The next
    # quote..." becomes "The next quote..." after stripping the *1).
    assert _strip_citations("*1 W or ds pa ck pow er .") == "W or ds pa ck pow er."
    assert _strip_citations("[1] W or ds pa ck pow er .") == "W or ds pa ck pow er."
    assert _strip_citations("¹ T he ne xt quote .") == "T he ne xt quote."
    # Bare "* " at the start of a sentence is a footnote ref in this
    # book's style — TTS reading "asterisk" between sentences is
    # worse than the risk of stripping a real emphasis asterisk.
    assert _strip_citations("* T he a ut onom ic ne r vous s ys tem .") == "T he a ut onom ic ne r vous s ys tem."
    # Don't strip "I ate 3 apples" — bare mid-sentence digits stay.
    # (Spaces before terminal punctuation are collapsed as a side
    # effect of marker stripping — harmless for TTS.)
    assert _strip_citations("I ate 3 apples .") == "I ate 3 apples."
    assert _strip_citations("see page 1 .") == "see page 1."
    # Bare superscripts flattened to digits, no brackets: citations sit
    # AFTER the sentence's terminal punctuation ("text.12"), so only
    # that shape is stripped. Digits BEFORE the period or at the start of
    # a sentence are genuine prose ("There were 12.", "12 Reasons...")
    # and must survive.
    assert _strip_citations("The claim is disputed.12") == "The claim is disputed."
    # Star glued to the number ("text.*12", "text.12*", "text *12") —
    # star-first is always a marker; digit-star is gated on the period
    # (see below), so "rated 5*." survives while ".12*" is stripped.
    assert _strip_citations("The claim is disputed.*12") == "The claim is disputed."
    assert _strip_citations("The claim is disputed.12*") == "The claim is disputed."
    assert _strip_citations("The claim is disputed *12") == "The claim is disputed"
    assert _strip_citations("The claim is disputed.*[3]") == "The claim is disputed."
    # Period-gated cluster: extraction often leaves a space (". 24"),
    # groups (".24,25") or unicode superscripts (".²⁴") after the period.
    assert _strip_citations("The claim is disputed. 24") == "The claim is disputed."
    assert _strip_citations("The claim is disputed.24,25") == "The claim is disputed."
    assert _strip_citations("The claim is disputed.²⁴") == "The claim is disputed."
    assert _strip_citations("The claim is disputed. 12 *") == "The claim is disputed."
    assert _strip_citations("The claim is disputed.[3 0]") == "The claim is disputed."
    # Mid-row citations: mangled spacing defeats the splitter, so one row
    # can hold "sentence.24 Next..." or a mid-sentence footnote star.
    # The cluster reaches into a capital-starting next sentence (".2 Now",
    # ".1 0 In") but never into lowercase prose ("3. 4 apples" survives).
    assert _strip_citations("It is complicated.2 Now he left.") == "It is complicated. Now he left."
    assert _strip_citations("The link.1 0 In lab animals died.") == "The link. In lab animals died."
    assert _strip_citations("Wellesley College.8 And violence rose.") == "Wellesley College. And violence rose."
    assert _strip_citations("he amygdala * is the core structure") == "he amygdala is the core structure"
    assert _strip_citations("pay; * when it rises late") == "pay; when it rises late"
    assert _strip_citations("behaviors, *23 and she will stop") == "behaviors, and she will stop"
    assert _strip_citations("rate . *31 Embed this word") == "rate. Embed this word"
    assert _strip_citations("less prosocial*—less charitable") == "less prosocial—less charitable"
    assert _strip_citations('syndrome (PMS)*—the symptoms stay') == 'syndrome (PMS)—the symptoms stay'
    # ...keep genuine prose numbers untouched (years, counts, versions,
    # ratings, chemistry). Anything before the period — or with no
    # period at all — is prose, never a citation.
    assert _strip_citations("The rate rose in 1964 .") == "The rate rose in 1964."
    assert _strip_citations("There were 12 .") == "There were 12."
    assert _strip_citations("The claim is disputed 12.") == "The claim is disputed 12."
    assert _strip_citations("The claim is disputed12.") == "The claim is disputed12."
    assert _strip_citations("He rated it 5*.") == "He rated it 5*."
    assert _strip_citations("The claim is disputed 12*") == "The claim is disputed 12*"
    assert _strip_citations("The claim is disputed 24") == "The claim is disputed 24"
    assert _strip_citations("CO2 levels rose.") == "CO2 levels rose."
    assert _strip_citations("Drink H2O daily.") == "Drink H2O daily."
    assert _strip_citations("Version 2.0 is out.") == "Version 2.0 is out."
    assert _strip_citations("5* hotels are great.") == "5* hotels are great."
    assert _strip_citations("2 * 3 equals 6.") == "2 * 3 equals 6."
    assert _strip_citations("I ate 3. 4 apples today.") == "I ate 3. 4 apples today."
    assert _strip_citations("Stop! 3 times in a row.") == "Stop! 3 times in a row."
    assert _strip_citations("Fig. 2 shows this clearly.") == "Fig. 2 shows this clearly."
    assert _strip_citations("In 1964. The war began.") == "In 1964. The war began."
    assert _strip_citations("Section 4.3 Overview here.") == "Section 4.3 Overview here."
    assert _strip_citations("DDC 612.8-dc23 record kept.") == "DDC 612.8-dc23 record kept."
    assert _strip_citations("What?! 3") == "What?!"
    assert _strip_citations("12 The next quote follows.") == "12 The next quote follows."
    assert _strip_citations("12* The next quote follows.") == "12* The next quote follows."
    # Standalone marker sentences identified as citation-only.
    assert _is_citation_only("12") is True
    assert _is_citation_only("14.") is True
    assert _is_citation_only("*12") is True
    assert _is_citation_only("12*") is True
    assert _is_citation_only("²⁴") is True
    assert _is_citation_only("^1") is True
    assert _is_citation_only("12 years later") is False

    print(f"OK: splitter={len(sents)} sentences, page_map={page_map}, title derivation correct, broken-chunk filter correct, citation strip correct")

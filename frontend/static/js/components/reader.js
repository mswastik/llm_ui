/**
 * Reader overlay — fullscreen PDF reader with TTS read-aloud and
 * in-PDF sentence highlight.
 *
 * The PDF is rendered with PDF.js 3.11's generic multi-page viewer
 * (PDFViewer from pdfjs-dist/web/pdf_viewer.js) — the same machinery
 * Mozilla's reference viewer uses. That gives us continuous scroll,
 * lazy page rendering, zoom, page-navigation and outline for free.
 *
 * Sentence highlighting rides on the per-page text layers that
 * PDFViewer renders: each page gets a <div class="textLayer"> whose
 * children are absolutely-positioned <span>s — one per PDF text item —
 * giving us a precise handle. On every `textlayerrendered` event we
 * cache the page's spans + strings (from `source.textLayer.textDivs`
 * and `.textContentItemsStr`, which PDF.js keeps parallel to the text
 * items), then toggle `.pdf-highlight` on the spans that overlap the
 * current TTS sentence.
 *
 * Why not the browser's native <iframe> viewer? It has no way to draw
 * a highlight over a specific span of the rendered PDF. PDF.js' text
 * layer does.
 *
 * Why keep PDF.js objects off `this`? PDF.js classes use private class
 * fields (#d, #u, #listeners, ...). If Alpine's reactive Proxy ever
 * wraps one of those objects, any private-field access throws "Cannot
 * read private member from an object whose class did not declare it".
 * So every PDF.js object lives on a module-level singleton `_pdf` —
 * outside Alpine's reactive world entirely (see _raw below).
 *
 * Viewer lifecycle: the PDFViewer + EventBus are rebuilt per open into
 * a *fresh* DOM subtree (old subtree dropped). PDFViewer attaches a
 * ResizeObserver to its container in its constructor and never exposes
 * a dispose — rebuilding the container node each time is the only way
 * to fully tear one down without leaking observers across opens.
 *
 * Lifecycle:
 *   1. `open-reader` event with {bookId} → fetch the book + file
 *   2. PDFViewer renders pages lazily; `pagesPromise` resolves once the
 *      viewer has built its page list — only then are page jumps valid
 *   3. Server streams one NDJSON event per sentence {idx, page, sentence,
 *      seg, url} → play each <audio> URL, jump + highlight the page
 *   4. Progress is persisted per sentence on the server
 */
import { getTtsVolume, setTtsVolume } from '../utils.js'

const PDFJS_VERSION = '3.11.174'
const PDFJS_WORKER = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDFJS_VERSION}/build/pdf.worker.min.js`

// pdfjsLib is loaded as a UMD <script> in base.html; resolve once it's
// ready. Polling instead of an event because 3.11's UMD doesn't emit
// a "ready" signal.
function _loadPdfjs() {
  return new Promise((resolve, reject) => {
    if (window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER
      return resolve(window.pdfjsLib)
    }
    const start = Date.now()
    const t = setInterval(() => {
      if (window.pdfjsLib) {
        clearInterval(t)
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER
        resolve(window.pdfjsLib)
      } else if (Date.now() - start > 3000) {
        clearInterval(t)
        reject(new Error('PDF.js failed to load from CDN'))
      }
    }, 50)
  })
}

// pdfjsViewer is the companion bundle that ships the multi-page viewer
// classes (PDFViewer, PDFLinkService, EventBus, etc.). Same pattern as
// _loadPdfjs — poll for window.pdfjsViewer after we know pdfjsLib is
// there. (Loading order: pdf.min.js first, then pdf_viewer.js, in base.html.)
function _loadPdfViewer() {
  return new Promise((resolve, reject) => {
    if (window.pdfjsViewer) return resolve(window.pdfjsViewer)
    const start = Date.now()
    const t = setInterval(() => {
      if (window.pdfjsViewer) {
        clearInterval(t)
        resolve(window.pdfjsViewer)
      } else if (Date.now() - start > 3000) {
        clearInterval(t)
        reject(new Error('pdfjsViewer failed to load from CDN'))
      }
    }, 50)
  })
}

// Non-reactive storage for PDF.js objects. The app mounts exactly one
// reader overlay, so this is a module-level singleton rather than
// per-instance state. That choice is deliberate: Alpine.js hands
// component methods a different proxy object depending on how the
// method is reached (template handler, Alpine.$data, magic getters),
// and PDF.js objects must NEVER go through one of those proxies (their
// private class fields would throw). A singleton means every code path
// — TTS playback, event-bus callbacks, template buttons — reads and
// writes the same raw viewer/doc regardless of what `this` it sees.
//
// Stored:
//   { doc, viewer, linkService, eventBus,
//     pageNum -> { texts, textDivs }   (built by textlayerrendered)
//     pendingHl, lastHl, outline }
const _pdf = {
  doc: null,
  viewer: null,
  linkService: null,
  eventBus: null,
  // PDFFindController (PDF search). Same non-reactive rule as the other
  // PDF.js objects.
  findCtrl: null,
  // Web Audio volume-boost graph for the TTS <audio> element (only built
  // when the user's volume differs from 1×). Kept raw — never proxied.
  audioCtx: null,
  gainNode: null,
  // pageNum -> { texts, textDivs } — captured at textlayerrendered.
  // texts/textDivs are parallel: PDF.js pushes one string and one
  // span per text item with a defined str.
  tlCache: new Map(),
  // { pageNum, sentence } queued until that page's text layer renders
  pendingHl: null,
  // { pageNum, sentence } — the most recent highlight request, so a
  // re-render of the same page re-applies it automatically.
  lastHl: null,
  // Raw outline items from doc.getOutline() plus the flattened list with
  // refs (kept non-reactive — PDF.js objects must not be proxied). The
  // reactive mirror (titles/depth only) lives on outlineItems.
  outline: [],
  outlineFlat: [],
}

function _raw() {
  return _pdf
}

// Client-side sentence splitter. Mirrors backend `tts_service._split_sentences`
// so client-extracted text produces the same segments the server would have
// given clean text. Why duplicate? Some PDFs are mangled by PyPDF2 (spaced
// out letters, broken per-line layouts) but render fine in PDF.js' own
// getTextContent — we want the client to take over extraction in those cases.
function _splitSentencesClient(text) {
  const parts = (text || '').trim().split(/(?<=[.!?])\s+|\n{2,}/).map((p) => p.trim()).filter(Boolean)
  const out = []
  for (const p of parts) {
    // Abbreviation heuristic: M., U.S., etc. — glue to next fragment.
    if (out.length && /^[A-Z]\.$/.test(p)) {
      out[out.length - 1] = (out[out.length - 1] + ' ' + p).trim()
    } else {
      out.push(p)
    }
  }
  return out
}

// Client-side mirror of backend book_service._strip_citations: strips
// footnote / citation reference markers that PyPDF2 or PDF.js flatten
// into the prose so TTS never reads "twelve" / "asterisk" for a
// reference. Keep the patterns in sync with the Python version: bracket
// refs (with optional flanking stars), star+number combos in either
// order ("*12" / "12*"), asterisk/dagger symbols, unicode superscripts,
// and digit markers glued AFTER the sentence's terminal punctuation
// ("text.12", "text.*12", "text.12*"). Digits BEFORE the period or at
// the start of a sentence are genuine prose ("There were 12.",
// "12 Reasons...") and must NOT be stripped.
function _stripCitationsClient(s) {
  let t = (s || '').trim()
  // Leading marker (footnote ref at the start of a sentence).
  t = t.replace(
    /^\s*(?:\**\[\d+(?:[-,–]\s*\d+)*\]\**|\*+\s*\d+|\d+\*+|\*\s+(?=[A-Z])|[†‡§¶]|[\u00B9\u00B2\u00B3\u2070-\u2079]+)\s*/,
    ''
  )
  // Trailing marker (end of sentence).
  t = t.replace(
    /\s*(?:\**\[\d+(?:[-,–]\s*\d+)*\]\**|\*+\s*\d+|\d+\s*\*+|\*+|[†‡§¶]|\^\d+|(?<=[.!?])\**\d+\**)\s*$/,
    ''
  )
  // Stripping can leave "disputed ." — pull punctuation back.
  t = t.replace(/\s+([.!?,;:])/g, '$1')
  return t.trim()
}

// A sentence that is only digits / citation punctuation ("12", "14.") is
// a flattened superscript reference or page number — never prose to read.
function _isCitationOnlyClient(s) {
  return /^[\d\s,.;:()\[\]{}*†‡§¶%#]+$/.test((s || '').trim())
}

// Extract every page's text via PDF.js (clean per-page text) and re-ship
// the resulting sentences + page_map to the server. The server's PyPDF2
// fallback produces badly-mangled text for some print-typeset PDFs;
// this replaces those rows with the same sentences the user sees in the
// reader. Returns the new total sentence count, or null on failure.
// Heuristic: classify a text item as superscript (footnote / citation
// marker) so we can drop it from the TTS input. We can't tell with
// absolute certainty, but superscript text in print-typeset books is
// almost always:
//   * very short (a number, a single letter, a star/asterisk) — under 5
//     visible chars
//   * rendered in a font height clearly smaller than the surrounding
//     body text on the same line
//
// So: build a per-page "body height" (the mode of the non-tiny text
// items' font sizes) and treat anything under ~70% of that AND short
// enough to be a reference marker as a superscript to drop. This avoids
// the false positive where the body of a book happens to contain a
// short number ("1.", "2.") — we only suppress when the FONT is also
// small, which only footnote markers and TOC page numbers share.
const _SUPERSCRIPT_MAX_CHARS = 4
const _SUPERSCRIPT_HEIGHT_RATIO = 0.7
// A text run counts as a "paragraph heading" (chapter title, section
// header) when its font height is clearly above the page's body line
// height. 1.15× is conservative enough to skip body-text emphasis
// (slightly larger bold spans) while still catching the typical
// 1.3–2.0× jump print-typeset PDFs use for headings.
const _HEADING_HEIGHT_RATIO = 1.15
// Headings are short and rarely carry terminal punctuation — anything
// ending in `.`/`?`/`!`/`:`/`;` is almost certainly body text. We cap
// the length at 120 chars to avoid catching the rare oversized lead
// paragraph on chapter-opening pages.
const _HEADING_MAX_CHARS = 120
// In print-typeset PDFs, body text is laid out on lines spaced at
// ~1 line height apart. When a real paragraph break exists in the
// source — heading, section break, blank line between paragraphs —
// the next line is dropped by MORE than the normal line spacing.
// We detect that as: |y(prev) - y(curr)| > 1.5× the typical line
// height. 1.0× is normal wrap; 1.5×+ is a paragraph break. This is
// the secondary signal (the primary being the heading's larger
// font) — it handles regular paragraph breaks with no heading
// involved.
const _PARAGRAPH_Y_GAP_MULT = 1.5

function _bodyLineHeight(items) {
  // Median line height for this page. We use the font height of the
  // text runs as a proxy (line height ≈ font height in print-typeset
  // PDFs at single spacing).
  const heights = []
  for (const it of items) {
    const s = (it.str || '').trim()
    if (s.length <= _SUPERSCRIPT_MAX_CHARS) continue
    const h = it.height || 0
    if (h > 0) heights.push(Math.round(h * 10) / 10)
  }
  if (!heights.length) return 12
  heights.sort((a, b) => a - b)
  return heights[Math.floor(heights.length / 2)]
}

// Body font height: the smallest common font on the page. Body
// text is usually the SMALLEST regular text — so when there's a
// tie in the font-height histogram (a page with one heading + a
// few body lines), we pick the smaller of the tied heights.
// Headings are at 1.3-2× body, so the body mode is the smallest
// common height.
function _bodyFontHeight(items) {
  const counts = new Map()
  for (const it of items) {
    const s = (it.str || '').trim()
    if (s.length <= _SUPERSCRIPT_MAX_CHARS) continue
    const h = Math.round((it.height || 0) * 10) / 10
    if (h <= 0) continue
    counts.set(h, (counts.get(h) || 0) + 1)
  }
  let best = 0, bestN = -1
  for (const [h, n] of counts) {
    if (n > bestN || (n === bestN && h < best)) {
      best = h
      bestN = n
    }
  }
  return best || 10
}

// Does a page contain superscript-style text runs — short content (≤4
// chars) rendered in a font height clearly below the page's body line
// height? In print-typeset PDFs that combination is almost exclusively a
// footnote / citation reference marker (footnote numbers, asterisk refs,
// TOC page leaders). This is the reliable detector: PDF.js gives us the
// per-run font metrics that PyPDF2 throws away.
async function _pageHasSuperscripts(page) {
  try {
    const tc = await page.getTextContent()
    const items = tc.items || []
    if (!items.length) return false
    const lineH = _bodyLineHeight(items)
    for (const it of items) {
      const s = (it.str || '').trim()
      const h = it.height || 0
      if (s.length > 0 && s.length <= _SUPERSCRIPT_MAX_CHARS
          && h > 0 && h < lineH * _SUPERSCRIPT_HEIGHT_RATIO) {
        return true
      }
    }
  } catch { /* unrenderable page — skip */ }
  return false
}

// Sample a handful of pages (start of the book + the page the user is
// resuming on) for superscript markers. Books with citations almost always
// have them on the early pages; the resume page covers books whose refs
// only appear later. When found, the reader forces the clean re-extract so
// the markers never reach the TTS engine.
async function _docHasSuperscripts(rawDoc, resumePage) {
  const numPages = rawDoc.numPages || 0
  const seen = new Set()
  const n = Math.min(numPages, 12)
  for (let p = 1; p <= n; p++) seen.add(p)
  if (resumePage > 1 && resumePage <= numPages) seen.add(resumePage)
  for (const p of seen) {
    try {
      const page = await rawDoc.getPage(p)
      if (await _pageHasSuperscripts(page)) return true
    } catch { /* keep scanning */ }
  }
  return false
}

async function _reextractAndShip(rawDoc, bookId) {
  try {
    const allSents = []
    const allPages = []
    const numPages = rawDoc.numPages
    // Cap to first 200 pages for the initial re-extract. Books longer
    // than that get a 200-page "starter" set the user can read; the
    // TTS will run for the first ~200 pages of clean sentences, and
    // when the user scrolls past page 200, they can re-open to
    // re-extract more (we add a guard to skip if sentences already
    // look healthy for the user's progress).
    const maxPages = Math.min(numPages, 200)
    for (let p = 1; p <= maxPages; p++) {
      const page = await rawDoc.getPage(p)
      const tc = await page.getTextContent()
      const items = tc.items || []
      const lineH = _bodyLineHeight(items)
      const bodyH = _bodyFontHeight(items)
      // Drop superscript runs (footnote / citation markers): they're
      // short AND have a font height well below the body line height
      // for this page. Concatenating a space in place keeps the
      // original text spacing so the highlight still lines up.
      //
      // Paragraph breaks come from three signals:
      //   1. y-gap: |Δy| between consecutive text items > 1.5× the
      //      body line height. Catches real paragraph breaks in
      //      the PDF layout (blank line between paragraphs, a
      //      heading with body well below it).
      //   2. inline heading: a heading is followed by body on the
      //      SAME line (no hasEOL between them) — common in chapter
      //      openings where the title runs into the first
      //      paragraph. We force a paragraph break so the heading
      //      stands alone as a sentence.
      //   3. heading with no inline follow-up: when a heading line
      //      ends (hasEOL), we force a paragraph break after it
      //      regardless of the y-gap, so the heading is always its
      //      own sentence even when the body text is on the very
      //      next line (small y-gap, no visible blank line in the
      //      PDF). A heading is a text run with font height
      //      > 1.15× the body height (print-typeset PDFs use a
      //      1.3-2× jump for headings), short (≤120 chars), and no
      //      terminal punctuation (so body sentences ending in `.`
      //      aren't misclassified).
      const yOf = (it) => Array.isArray(it.transform) ? it.transform[5] : 0
      const lines = []
      let buf = ''
      let prevY = null
      let lastWasHeading = false
      for (const it of items) {
        const s = it.str || ''
        const trimmed = s.trim()
        const h = it.height || 0
        const isSuper = trimmed.length > 0
          && trimmed.length <= _SUPERSCRIPT_MAX_CHARS
          && h > 0
          && h < lineH * _SUPERSCRIPT_HEIGHT_RATIO
        const isHeading = !isSuper
          && trimmed.length > 0
          && trimmed.length <= _HEADING_MAX_CHARS
          && h > 0
          && h > bodyH * _HEADING_HEIGHT_RATIO
          && !/[.?!\-:;]$/.test(trimmed)
        if (isSuper) {
          // Replace the marker with a single space so word boundaries
          // stay sensible ("text1next" would otherwise fuse).
          if (buf && !buf.endsWith(' ')) buf += ' '
        } else {
          const currY = yOf(it)
          if (prevY !== null && buf) {
            const yDelta = Math.abs(currY - prevY)
            if (yDelta > _PARAGRAPH_Y_GAP_MULT * lineH) {
              // The previous line is at least ~1.5× line height away
              // from the current one — that's a real paragraph
              // break, not just a line wrap. Promote the existing
              // line break to a paragraph break (\n\n when joined).
              lines.push(buf)
              lines.push('')
              buf = ''
            }
          }
          // Inline heading: a heading is on the same visual line as
          // the following body text (no hasEOL between them) — common
          // in chapter openings where the title runs into the first
          // paragraph. Force a paragraph break so the heading stands
          // alone as a sentence.
          if (lastWasHeading && !isHeading && buf.trim()) {
            lines.push(buf.trim())
            lines.push('')
            buf = ''
          }
          buf += s
        }
        if (it.hasEOL) {
          lines.push(buf)
          buf = ''
          prevY = yOf(it)
          lastWasHeading = isHeading
          // If THIS line was a heading, force a paragraph break
          // (blank line) after it so the splitter can split the
          // heading from any body text below — even if the body
          // text is on the very next line (small y-gap, no visible
          // blank line in the PDF). The heading is always its own
          // sentence; TTS reads "Chapter Three" then a brief pause,
          // then the body paragraph.
          if (isHeading) {
            lines.push('')
          }
        } else {
          buf += ' '
          // Don't update prevY on a continuation — the y of the
          // continued line is the same as the first item's y.
          lastWasHeading = isHeading
        }
      }
      if (buf) lines.push(buf)
      const pageText = lines.join('\n').replace(/[ \t]+/g, ' ').trim()
      if (!pageText) continue
      const sents = _splitSentencesClient(pageText)
      let charStart = 0
      for (const s of sents) {
        let text = _stripCitationsClient(s)
        if (text.length < 2 || _isCitationOnlyClient(text)) {
          charStart += s.length + 1
          continue
        }
        allSents.push({ text, page: p, char_start: charStart })
        allPages.push(p)
        charStart += s.length + 1
      }
    }
    if (!allSents.length) return null
    const r = await fetch(`/api/books/${bookId}/sentences`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentences: allSents, page_map: allPages }),
    })
    if (!r.ok) return null
    const j = await r.json().catch(() => null)
    return j?.total_sentences ?? allSents.length
  } catch {
    return null
  }
}

// Resolve a PDF outline item's destination to a 1-based PDF page number.
// Outline `dest` may be a string (named destination) or an explicit
// destination array whose first element is a page ref ({num, gen}) or an
// integer index. Returns null when there is no resolvable destination.
async function _resolveOutlinePage(doc, ref) {
  try {
    let dest = ref && ref.dest
    if (dest == null) return null
    if (typeof dest === 'string') dest = await doc.getDestination(dest)
    if (!Array.isArray(dest) || dest.length < 1) return null
    const destRef = dest[0]
    if (destRef !== null && typeof destRef === 'object' && destRef.num !== undefined) {
      const pageIndex = await doc.getPageIndex(destRef)
      return pageIndex + 1
    }
    if (Number.isInteger(destRef)) return destRef + 1
    return null
  } catch {
    return null
  }
}

// Fuzzy match a sentence against the page's text-layer strings and
// return the indices of the spans that cover it. `texts` is the array
// of per-span strings PDF.js produced (parallel to the textDivs).
//
// We match a *prefix* of the sentence (length thresholds 30→15→8)
// because the TTS sentence text and the PDF text often differ at the
// end — TTS may read "Hello world." where the PDF has "Hello world!"
// (or a hyphenated break). A single space is inserted between spans
// before matching so words that merely wrap across lines (where the
// PDF has no space character) still line up with the sentence text.
function _findMatchTexts(texts, sentence) {
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim()
  const target = norm(sentence)
  if (!target) return null
  const pageText = texts.map((t) => t || '').join(' ').replace(/\s+/g, ' ').toLowerCase()
  const targetLower = target.toLowerCase()
  let startIdx = -1
  for (const n of [Math.min(30, target.length), Math.min(15, target.length), Math.min(8, target.length)]) {
    if (n < 3) continue
    startIdx = pageText.indexOf(targetLower.slice(0, n))
    if (startIdx >= 0) break
  }
  if (startIdx < 0) return null
  const endIdx = startIdx + target.length
  // Map the page-text char index back to which spans overlap it.
  let cursor = 0
  const matched = []
  for (let i = 0; i < texts.length; i++) {
    const s = texts[i] || ''
    const itemStart = cursor
    const itemEnd = cursor + s.length
    if (itemEnd > startIdx && itemStart < endIdx) {
      matched.push(i)
    }
    cursor = itemEnd + 1
    if (cursor > endIdx) break
  }
  return matched.length ? matched : null
}

export function reader() {
  return {
    // ─── Reactive state ────────────────────────────────────────────
    open: false,
    book: null,
    bookId: null,
    sentences: [],
    currentIdx: 0,
    isLoading: false,
    error: '',
    // Reading view for url/text saves with an article snapshot:
    // 'article' (formatted HTML, pictures) or 'tts' (sentence list).
    readerView: 'article',

    // TTS playback state
    audio: null,
    segments: [],
    segCursor: 0,
    streamDone: false,
    pendingPlay: false,
    isPlaying: false,
    _abort: null,
    // True while playback is paused mid-sentence — play() must then replay
    // the full current sentence instead of resuming mid-word.
    _paused: false,

    // PDF state — only safe primitives are reactive; PDF.js objects
    // live in the per-instance Map (see _raw).
    pdfReady: false,
    pdfNumPages: 0,
    currentPdfPage: 1,
    pageMap: [],
    outlineOpen: false,
    outlineItems: [],
    outlineEmpty: true,

    // PDF zoom (percentage shown in the toolbar; mirrors the viewer's
    // currentScale). 100 = 100%.
    zoomPct: 100,

    // PDF search (find) state
    findOpen: false,
    findQuery: '',
    findCurrent: 0,
    findTotal: 0,
    findPending: false,
    findNotFound: false,
    _findDebounce: null,

    // TTS playback volume boost (0.2×–2×); applied via a Web Audio gain
    // node so it can exceed the <audio> element's 0–1 volume range.
    volume: 1,

    // Non-reactive scratch (page-persist throttle)
    _pageTimer: null,
    _pendingPage: null,

    init() {
      window.addEventListener('open-reader', (e) => this.start(e.detail?.bookId))
      // Bare `@keydown.window` in Alpine was not reliably firing for
      // ArrowLeft/Right on this element, and the template-level Space
      // binding was lost in a refactor (Space fell through to the
      // browser and scrolled the page). All reader keys live here in
      // one real DOM listener instead.
      this._arrowHandler = (ev) => this.onReaderKey(ev)
      window.addEventListener('keydown', this._arrowHandler, true)
    },

    // ─── Lifecycle ─────────────────────────────────────────────────
    async start(bookId) {
      if (!bookId) return
      this.bookId = bookId
      this.open = true
      this.isLoading = true
      this.error = ''
      this.segments = []
      this.segCursor = 0
      this.streamDone = false
      this.pendingPlay = false
      this.isPlaying = false
      // Playback boost from Settings → TTS (localStorage-backed).
      this.volume = getTtsVolume()
      this._resetFind()

      try {
        const r = await fetch(`/api/books/${bookId}?include_text=true`)
        if (!r.ok) throw new Error('Failed to load book: ' + r.status)
        const book = await r.json()
        this.book = book
        this.sentences = book.sentences || []
        this.pageMap = book.page_map || []
        // Web/paste saves with an article snapshot open formatted;
        // everything else (and legacy rows without a snapshot) uses TTS.
        this.readerView = book.has_article ? 'article' : 'tts'
        const savedIdx = Math.max(0, Math.min(
          book.current_sentence_idx || 0,
          Math.max(0, this.sentences.length - 1)
        ))
        this.currentPdfPage = Math.max(1, book.current_page || 1)
        // Header/progress should reflect where reading will resume (saved
        // sentence, or the top of the page shown in the viewer if the user
        // is further ahead). play() recomputes this at press time too.
        this.currentIdx = this._pickStartIdx(savedIdx)
        this._ensureAudio()

        if (book.file_type === 'pdf') {
          await this._loadPdf()
        } else {
          // EPUB: no native viewer; we use a simple text-only pane.
          this.pdfReady = false
        }
      } catch (e) {
        this.error = e.message || 'Failed to load book'
        this.isLoading = false
        return
      }
      this.isLoading = false
    },

    close() {
      this.stop()
      this._flushPersist()
      const raw = _raw(this)
      // Tell the find controller to drop its highlights before the viewer
      // is torn down.
      if (raw.eventBus) {
        try { raw.eventBus.dispatch('findbarclose', { source: raw }) } catch { /* noop */ }
      }
      if (raw.audioCtx) {
        try { raw.audioCtx.close().catch(() => {}) } catch { /* noop */ }
        raw.audioCtx = null
        raw.gainNode = null
      }
      raw.findCtrl = null
      if (raw.viewer) {
        // setDocument(null) tears down the page views + listeners that
        // PDFViewer attached to the eventBus.
        try { raw.viewer.setDocument(null) } catch { /* noop */ }
        try { raw.viewer.cleanup() } catch { /* noop */ }
        raw.viewer = null
      }
      if (raw.eventBus) {
        try { raw.eventBus.dispose() } catch { /* noop */ }
        raw.eventBus = null
      }
      raw.linkService = null
      raw.tlCache.clear()
      raw.pendingHl = null
      raw.lastHl = null
      raw.outline = []
      raw.outlineFlat = []
      this._resetFind()
      if (raw.doc) {
        try { raw.doc.destroy() } catch { /* noop */ }
        raw.doc = null
      }
      // Drop the whole viewer DOM subtree so the PDFViewer's internal
      // ResizeObserver has nothing left to observe and can be GC'd.
      const host = this.$refs && this.$refs.pdfHost
      if (host) host.textContent = ''
      this.audio = null
      this.volume = 1
      this.pdfReady = false
      this.pdfNumPages = 0
      this.pageMap = []
      this.outlineOpen = false
      this.outlineItems = []
      this.outlineEmpty = true
      this.zoomPct = 100
      this.open = false
      this.book = null
      this.bookId = null
      this.readerView = 'article'
      this.sentences = []
      this.segments = []
      this.segCursor = 0
      this.currentIdx = 0
      this.currentPdfPage = 1
    },

    // ─── TTS controls ──────────────────────────────────────────────
    toggle() {
      if (this.isPlaying) this.pause()
      else this.play()
    },

    async play() {
      if (this.isPlaying) return
      if (!this.sentences.length) return
      this.error = ''
      // Apply the volume boost now (this call runs inside the user's click
      // gesture, which is the only time AudioContext.resume() is allowed).
      this._ensurePlaybackGraph()
      const wasPaused = this._paused
      this._paused = false
      if (this.segments.length === 0) {
        // Start from the reading position: saved sentence progress when it
        // is at/after the page on screen, otherwise from the top of the
        // page the user is actually looking at (so opening a book at page
        // 100 doesn't restart the TTS at page 1).
        await this._startStream(this._pickStartIdx(this.currentIdx))
        return
      }
      if (wasPaused && this.pendingPlay) {
        // Pause happened in the gap between two sentences — the next
        // segment starts on its own when it arrives (same as normal
        // flow). Don't replay the sentence that already finished.
        this.isPlaying = true
        return
      }
      if (wasPaused) {
        // pause() rewound the audio to the start of the segment; replay
        // the FULL current sentence rather than resuming mid-word.
        try { this.audio.currentTime = 0 } catch { /* noop */ }
      }
      this.audio.play().catch(() => { /* user gesture will fix */ })
      this.isPlaying = true
    },

    // Pick the sentence index TTS should start from. `savedIdx` is the
    // persisted read-along position (current_sentence_idx).
    //
    // Default behaviour: resume exactly at the checkpoint. But the PDF
    // viewer also tracks which page is on screen (currentPdfPage): if the
    // user is looking at a *different* page than the checkpoint's page —
    // they scrolled, used page nav, or jumped via the outline since the
    // checkpoint was saved — TTS starts at the top of the page they are
    // actually on instead of silently pulling them back to the old page.
    _pickStartIdx(savedIdx) {
      const pageMap = this.pageMap || []
      const saved = Math.max(0, Math.min(savedIdx || 0, Math.max(0, pageMap.length - 1)))
      if (!pageMap.length || this.book?.file_type !== 'pdf') return saved
      const viewPage = Math.max(1, this.currentPdfPage || 1)
      const savedPage = pageMap[saved]
      if (savedPage && savedPage === viewPage) return saved
      // Viewer is on another page → find the first sentence on it.
      const at = pageMap.indexOf(viewPage)
      if (at >= 0) return at
      // Page has no extracted sentences (image/blank) → nearest content
      // page after it.
      for (let p = viewPage + 1; p <= this.pdfNumPages; p++) {
        const i = pageMap.indexOf(p)
        if (i >= 0) return i
      }
      return saved
    },

    // Jump the viewer back to the page of the sentence currently being
    // spoken and re-apply its highlight. Shown while TTS plays and the
    // user has scrolled away from the read-along position.
    _goToSpokenPage() {
      const pageMap = this.pageMap || []
      if (!pageMap.length) return
      const idx = Math.max(0, Math.min(this.currentIdx || 0, pageMap.length - 1))
      const page = pageMap[idx]
      if (!page) return
      this.goToPage(page)
      const sent = (this.sentences && this.sentences[idx] && this.sentences[idx].text) || ''
      if (sent) this._highlightSentence(page, sent)
    },

    pause() {
      // Remember that we were cut mid-sentence, and rewind the audio to
      // the start of the segment so the next play() replays the FULL
      // sentence instead of resuming mid-word. (If we paused in the gap
      // between sentences — audio already ended — there's nothing to
      // rewind and pendingPlay stays set.)
      this._paused = true
      if (this.audio && !this.audio.paused) {
        this.audio.pause()
        try { this.audio.currentTime = 0 } catch { /* noop */ }
      }
      this.isPlaying = false
    },

    // Start TTS from the first sentence of the currently displayed page.
    // Unlike the plain play button this ALWAYS starts at the top of the
    // page the user is looking at, tearing down whatever was in flight.
    playFromPage() {
      if (!this.sentences.length) return
      let idx = -1
      const pageMap = this.pageMap || []
      if (this.book?.file_type === 'pdf' && pageMap.length && this.pdfNumPages) {
        const page = Math.max(1, this.currentPdfPage || 1)
        idx = pageMap.indexOf(page)
        if (idx < 0) {
          // Page has no extracted sentences (image/blank) → first
          // content page after it.
          for (let p = page + 1; p <= this.pdfNumPages; p++) {
            const i = pageMap.indexOf(p)
            if (i >= 0) { idx = i; break }
          }
        }
      }
      if (idx < 0) idx = this._pickStartIdx(this.currentIdx)
      this.stop()
      this._paused = false
      this.currentIdx = idx
      this._ensureAudio()
      this._ensurePlaybackGraph()
      this._startStream(idx)
    },

    // Jump to the previous or next sentence and resume TTS from there.
    // Repeated presses walk through the book one sentence at a time.
    // The new index is clamped to the sentence range; pressing prev at
    // the very first sentence stops at the top (so the user can tell
    // they've hit the start), pressing next at the end does the same.
    async skipSentence(direction) {
      if (!this.sentences.length) return
      // Anchor on the last *spoken* sentence, not the viewer's page —
      // that's what the user's mental model is ("where TTS is / was").
      // If nothing has played yet, fall back to the resume index.
      const base = Math.max(0, Math.min(this.currentIdx || 0, this.sentences.length - 1))
      const target = base + (direction > 0 ? 1 : -1)
      if (target < 0 || target >= this.sentences.length) return
      // Tear down whatever is mid-flight; the new stream re-uses the
      // existing <audio> element and the cached TTS audio on the server
      // (re-reads of the same sentence are instant).
      this.stop()
      this.currentIdx = target
      // For PDF: jump the viewer to the target page so the user lands
      // on the right spot even before the first audio segment arrives.
      const targetPage = this.pageMap[target]
      if (this.book?.file_type === 'pdf' && targetPage) {
        this.goToPage(targetPage)
      }
      this._ensureAudio()
      this._ensurePlaybackGraph()
      await this._startStream(target)
    },

    prevSentence() { this.skipSentence(-1) },
    nextSentence() { this.skipSentence(1) },

    stop() {
      if (this._abort) {
        try { this._abort.abort() } catch { /* noop */ }
        this._abort = null
      }
      if (this.audio) {
        this.audio.pause()
        this.audio.removeAttribute('src')
        this.audio.load()
      }
      this.isPlaying = false
    },

    _ensureAudio() {
      if (this.audio) return
      const a = new Audio()
      a.preload = 'auto'
      a.addEventListener('ended', () => this._onSegmentEnd())
      a.addEventListener('error', () => this._onAudioError())
      this.audio = a
    },

    // ─── Volume boost (Web Audio gain) ─────────────────────────────
    // The <audio> element's own volume is capped at 1.0, so boosting past
    // 100% requires routing it through an AudioContext gain node. The graph
    // is only built when the user's volume differs from 1× (default), so
    // normal playback is untouched and never depends on the context.
    _ensurePlaybackGraph() {
      const raw = _raw(this)
      const a = this.audio
      if (!a) return
      const vol = Math.max(0.2, Math.min(2, Number(this.volume) || 1))
      if (Math.abs(vol - 1) > 0.001 && !raw.gainNode) {
        try {
          const Ctor = window.AudioContext || window.webkitAudioContext
          if (!Ctor) {
            // No Web Audio: fall back to the element's (≤1) volume.
            a.volume = Math.min(1, vol)
            return
          }
          const ctx = new Ctor()
          const src = ctx.createMediaElementSource(a)
          const gain = ctx.createGain()
          src.connect(gain)
          gain.connect(ctx.destination)
          raw.audioCtx = ctx
          raw.gainNode = gain
        } catch {
          raw.gainNode = null
          return
        }
      }
      if (raw.audioCtx && raw.audioCtx.state === 'suspended') {
        raw.audioCtx.resume().catch(() => { /* noop */ })
      }
      if (raw.gainNode) raw.gainNode.gain.value = vol
    },

    // Called from the header volume slider while the reader is open.
    // Persists the choice and, if a gain graph already exists (mid-play),
    // updates it live.
    onVolumeInput() {
      this.volume = Math.max(0.2, Math.min(2, Number(this.volume) || 1))
      setTtsVolume(this.volume)
      const raw = _raw(this)
      if (raw.gainNode) raw.gainNode.gain.value = this.volume
      if (raw.audioCtx && raw.audioCtx.state === 'suspended') {
        raw.audioCtx.resume().catch(() => { /* noop */ })
      }
    },

    // ─── PDF find (search) ─────────────────────────────────────────
    // Backed by pdf.js' own PDFFindController, wired into the stock
    // PDFViewer exactly like the reference viewer: matches are painted on
    // the per-page text layers and the active match is scrolled into view.
    // The controller extracts page text itself, so results cover the whole
    // document (not just rendered pages).
    openFind() {
      const raw = _raw(this)
      if (!this.open || this.book?.file_type !== 'pdf') return
      if (!raw.viewer || !raw.viewer.pdfDocument) return
      this.findOpen = true
      this.$nextTick(() => {
        const el = this.$refs?.findInput
        if (el) { el.focus(); el.select() }
      })
    },

    closeFind() {
      if (!this.findOpen) return
      this.findOpen = false
      clearTimeout(this._findDebounce)
      this._findDebounce = null
      this._resetFind()
      const raw = _raw(this)
      if (raw.eventBus) {
        try { raw.eventBus.dispatch('findbarclose', { source: this }) } catch { /* noop */ }
      }
    },

    _resetFind() {
      this.findOpen = false
      this.findQuery = ''
      this.findCurrent = 0
      this.findTotal = 0
      this.findPending = false
      this.findNotFound = false
    },

    // Live typing → debounce, then a fresh search. Empty query clears.
    findOnInput() {
      clearTimeout(this._findDebounce)
      this.findNotFound = false
      const q = (this.findQuery || '').trim()
      if (!q) {
        // No query: drop any active search + highlights.
        const raw = _raw(this)
        if (raw.eventBus) {
          try { raw.eventBus.dispatch('findbarclose', { source: this }) } catch { /* noop */ }
        }
        this._resetFind()
        this.findOpen = true // keep the bar open for the next keystroke
        return
      }
      this._findDebounce = setTimeout(() => this._dispatchFind({ type: '' }), 220)
    },

    findNext() { this._findStep(false) },
    findPrev() { this._findStep(true) },

    _findStep(previous) {
      if (!(this.findQuery || '').trim()) return
      clearTimeout(this._findDebounce)
      this._findDebounce = null
      this._dispatchFind({ type: 'again', previous })
    },

    // Mirrors the reference viewer's find bar event payloads.
    _dispatchFind({ type, previous = false }) {
      const raw = _raw(this)
      if (!raw.eventBus) return
      raw.eventBus.dispatch('find', {
        source: this,
        type,
        query: (this.findQuery || '').trim(),
        phraseSearch: true,
        caseSensitive: false,
        entireWord: false,
        highlightAll: true,
        findPrevious: previous,
        matchDiacritics: false,
      })
    },

    // Esc inside the reader: close the find bar first, then the reader.
    handleEscape() {
      if (this.findOpen && this.book?.file_type === 'pdf') this.closeFind()
      else this.close()
    },

    // Window-level key router. Space toggles play/pause (preventDefault
    // so the page doesn't scroll); arrows step prev/next sentence in
    // TTS. Alpine's @keydown.arrowleft.window / .arrowright matching
    // was not firing reliably for ArrowLeft/Right in this app, so we
    // filter here instead. Escape and Ctrl/Cmd+F stay as template
    // listeners in reader_overlay.html.
    onReaderKey(ev) {
      if (!this.open) return
      const key = ev?.key
      const tag = ev?.target?.tagName
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      // Space → play/pause. Gated on focus so typing a space in the
      // page-number input or the find query still works.
      if (key === ' ' || key === 'Spacebar') {
        if (typing) return
        ev.preventDefault()
        this.toggle()
        return
      }
      // Arrows → prev/next sentence. Gated on focus so typing in the
      // page-number input or the find query still works. The find
      // bar takes the arrow keys when it's open.
      if (key === 'ArrowLeft' || key === 'ArrowRight') {
        if (this.findOpen) return
        if (typing) return
        ev.preventDefault()
        this.skipSentence(key === 'ArrowLeft' ? -1 : 1)
      }
    },

    // ─── Stream → segment queue → audio ────────────────────────────
    async _startStream(fromIdx) {
      this._abort = new AbortController()
      // Reset the queue so a re-entry (e.g. skipSentence) doesn't play
      // segments from the previous stream — the OLD segments array
      // would still be at segCursor=0 and the OLD [0] would be played
      // for the new fetch's first NDJSON line.
      this.segments = []
      this.segCursor = 0
      this.streamDone = false
      this.pendingPlay = false
      this._paused = false
      this.isPlaying = true
      try {
        const res = await fetch(
          `/api/books/${this.bookId}/stream?from_idx=${fromIdx}`,
          { signal: this._abort.signal }
        )
        if (!res.ok) throw new Error('Stream failed: ' + res.status)
        const rdr = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        for (;;) {
          const { done, value } = await rdr.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          let nl
          while ((nl = buf.indexOf('\n')) !== -1) {
            const line = buf.slice(0, nl).trim()
            buf = buf.slice(nl + 1)
            if (!line) continue
            let evt
            try { evt = JSON.parse(line) } catch { continue }
            if (evt.error) { this.error = evt.error; break }
            if (evt.done) { this.streamDone = true; break }
            if (evt.url) {
              this.segments.push({
                url: evt.url,
                idx: evt.idx,
                page: evt.page,
                sentence: evt.sentence,
              })
              if (this.segCursor === 0 && this.segments.length === 1) {
                this._playCurrent()
              }
            }
          }
        }
        this.streamDone = true
        if (this.pendingPlay) {
          this.pendingPlay = false
          this._playCurrent()
        }
      } catch (e) {
        if (e?.name !== 'AbortError') {
          this.error = e.message || 'Stream error'
        }
        this.isPlaying = false
      }
    },

    async _playCurrent() {
      if (this.segCursor >= this.segments.length) {
        if (this.streamDone) {
          this.isPlaying = false
          return
        }
        this.pendingPlay = true
        return
      }
      const seg = this.segments[this.segCursor]
      this.currentIdx = seg.idx
      // For PDF: jump the viewer to the right page (triggers lazy
      // render) and highlight the current sentence.
      if (this.book?.file_type === 'pdf' && seg.page) {
        this._highlightSentence(seg.page, seg.sentence)
      }
      this.audio.src = seg.url
      this.audio.play().catch(() => { /* user gesture will fix */ })
      this.isPlaying = true
    },

    _onSegmentEnd() {
      this.segCursor += 1
      if (this.segCursor < this.segments.length) {
        this._playCurrent()
      } else if (this.streamDone) {
        this.isPlaying = false
      } else {
        this.pendingPlay = true
      }
    },

    _onAudioError() {
      // One bad segment shouldn't kill the whole stream.
      this.segCursor += 1
      if (this.segCursor < this.segments.length) this._playCurrent()
      else this.isPlaying = false
    },

    // ─── PDF.js loading ────────────────────────────────────────────
    async _loadPdf() {
      const lib = await _loadPdfjs()
      const viewerLib = await _loadPdfViewer()
      const r = await fetch(`/api/books/${this.bookId}/file`)
      if (!r.ok) throw new Error('Failed to fetch file: ' + r.status)
      const buf = await r.arrayBuffer()
      const rawDoc = await lib.getDocument({
        data: buf,
        disableWorker: true,
        useWorkerFetch: false,
      }).promise
      const raw = _raw(this)
      raw.doc = rawDoc
      this.pdfNumPages = rawDoc.numPages

      // Build a FRESH container subtree for this open. PDFViewer pins a
      // ResizeObserver to its container in its constructor and never
      // lets go — dropping the whole subtree on close is the clean way
      // to let a viewer be GC'd. The host div is a stable mount point.
      const host = this.$refs?.pdfHost
      if (!host) throw new Error('PDF viewer host missing')
      host.textContent = ''
      const container = document.createElement('div')
      container.className = 'pdfjs-container'
      const viewerEl = document.createElement('div')
      viewerEl.className = 'pdfViewer'
      container.append(viewerEl)
      host.append(container)

      // Wire the EventBus + PDFLinkService + PDFViewer exactly the way
      // Mozilla's reference viewer does. PDFViewer lazy-renders pages on
      // scroll and gives us scroll, page nav, link annotations, etc.
      const eventBus = new viewerLib.EventBus()
      const linkService = new viewerLib.PDFLinkService({ eventBus })
      // PDF search. Progressive counts keep the UI's "n / m" live while a
      // long document is being scanned.
      const findController = new viewerLib.PDFFindController({
        linkService,
        eventBus,
        updateMatchesCountOnProgress: true,
      })
      const pdfViewer = new viewerLib.PDFViewer({
        container,
        viewer: viewerEl,
        linkService,
        eventBus,
        findController,
        // Text layer for highlighting + selection.
        textLayerMode: 1 /* TextLayerMode.ENABLE */,
        // Annotation layer ON so link annotations (TOC entries,
        // footnote / bibliography cross-refs, URL links) are
        // clickable in the rendered PDF — same UX as a normal PDF
        // viewer. PDF.js draws them as invisible-but-clickable
        // link rectangles over the text.
        annotationMode: 1 /* AnnotationMode.ENABLE */,
      })
      // Store BEFORE setDocument — setDocument fires pagechanging
      // synchronously for the initial page, and we want our handlers
      // attached for that.
      raw.viewer = pdfViewer
      raw.linkService = linkService
      raw.eventBus = eventBus
      raw.findCtrl = findController
      // PDFLinkService needs a back-reference to the viewer to resolve
      // destinations (used by our outline sidebar). The reference viewer
      // always calls setViewer before setDocument.
      linkService.setViewer(pdfViewer)

      // Capture per-page text spans as each page's text layer renders.
      // In 3.11 the event payload is { source: PDFPageView, pageNumber,
      // numTextDivs, error } — the strings + spans live on
      // source.textLayer (textContentItemsStr / textDivs, parallel to
      // the PDF text items).
      eventBus.on('textlayerrendered', (e) => this._onTextLayerRendered(e))

      // Track the current page as the user scrolls. PDFViewer keeps
      // currentPageNumber in sync with the most-visible page; we mirror
      // it to reactive state so the header counter + progress persist
      // follow along without manual scroll listeners.
      eventBus.on('pagechanging', (e) => {
        this.currentPdfPage = e.pageNumber
        this._persistPage(e.pageNumber)
      })

      // Mirror zoom changes onto reactive state for the toolbar % readout.
      // Fires whenever the scale changes — our zoom buttons, or any future
      // wheel-zoom. (initial fit-to-width is mirrored right after setup.)
      eventBus.on('scalechanging', (e) => {
        const s = e && e.scale
        if (typeof s === 'number' && s > 0) {
          this.zoomPct = Math.round(s * 100)
        }
      })

      // Mirror find state onto reactive data for the find bar.
      eventBus.on('updatefindmatchescount', (e) => {
        const m = e.matchesCount || {}
        this.findCurrent = m.current || 0
        this.findTotal = m.total || 0
      })
      eventBus.on('updatefindcontrolstate', (e) => {
        this.findPending = e.state === 3 /* FindState.PENDING */
        this.findNotFound = e.state === 1 /* FindState.NOT_FOUND */
        // While PENDING the count is still partial — keep the last stable
        // one until the scan settles.
        if (e.state !== 3) {
          const m = e.matchesCount || {}
          this.findCurrent = m.current || 0
          this.findTotal = m.total || 0
        }
      })

      linkService.setDocument(rawDoc)
      pdfViewer.setDocument(rawDoc)

      // Outline / bookmarks — fetch once, flatten with refs kept off the
      // reactive surface (raw), and mirror the titles/depth onto reactive
      // outlineItems so the sidebar renders when it appears.
      try {
        raw.outline = (await rawDoc.getOutline()) || []
      } catch { raw.outline = [] }
      raw.outlineFlat = []
      const flat = []
      const walk = (items, depth) => {
        for (const it of items) {
          flat.push({ title: it.title || '(untitled)', depth, ref: it })
          if (it.items && it.items.length) walk(it.items, depth + 1)
        }
      }
      walk(raw.outline, 0)
      raw.outlineFlat = flat
      this.outlineItems = flat.map(({ title, depth }) => ({ title, depth }))
      // Mirror the emptiness onto a plain reactive flag so the sidebar's
      // "no outline" message reliably hides once items exist.
      this.outlineEmpty = flat.length === 0
      // Resolve each entry's destination to a page number (async) and
      // patch it in — rows appear immediately with titles, then page
      // numbers fill in next to them.
      flat.forEach((entry, i) => {
        _resolveOutlinePage(rawDoc, entry.ref).then((pageNum) => {
          if (pageNum == null || !this.bookId || i >= this.outlineItems.length) return
          this.outlineItems.splice(i, 1, { ...this.outlineItems[i], page: pageNum })
        }).catch(() => { /* leave without a page number */ })
      })

      const resumePage = Math.max(1, Math.min(this.currentPdfPage || 1, rawDoc.numPages))

      // PDFViewer builds its page list asynchronously: right after
      // setDocument() pagesCount is still 0 and setting
      // currentPageNumber is silently ignored. Wait for pagesPromise
      // before any jump/zoom so navigation actually lands.
      try { await pdfViewer.pagesPromise } catch { /* init may still render */ }

      // Fit pages to the container width (the reference viewer default).
      pdfViewer.currentScaleValue = 'auto'
      // Mirror the actual rendered fit-to-width scale into the toolbar
      // readout (scalechanging may not fire for the initial 'auto' set).
      this.zoomPct = Math.round((pdfViewer.currentScale || 1) * 100)

      this.pdfReady = true
      if (resumePage !== pdfViewer.currentPageNumber) {
        pdfViewer.currentPageNumber = resumePage
      }

      // Re-extract sentences client-side if the server's PyPDF2 fallback
      // produced too few (mangled) or too many (per-line) sentences.
      const pages = this.pdfNumPages
      const cachedSents = this.sentences.length
      const looksBad =
        cachedSents < Math.max(50, pages / 2) || cachedSents > pages * 10
      // Do the CACHED sentences still carry citation-marker-like text
      // (standalone digits, or text that citation-stripping would change)?
      // A book uploaded before the strip patterns existed has those
      // markers baked into its cache — re-extracting is the only way to
      // get rid of them for TTS. Once the client re-extracts and the
      // clean list is POSTed back, this turns false and later opens skip
      // the expensive re-extract.
      const hasMarkers = (this.sentences || []).some((s) => {
        const t = (s && s.text) || ''
        return !!t && (_isCitationOnlyClient(t) || _stripCitationsClient(t) !== t.trim())
      })
      // ...or when the PDF contains superscript citation markers. The font
      // scan is cheap (12 pages + the resume page) and catches reference
      // books whose PyPDF2 extraction looks healthy but still carries
      // footnote numbers that TTS would read out loud. Gated on hasMarkers
      // so a clean cache (already re-extracted) doesn't re-run it.
      // The superscript scan samples up to a dozen pages via getTextContent
      // — only run it when there's an actual reason to re-extract (mangled
      // count, or the cache still carries marker-like text). A clean cache
      // (already re-extracted once) must not re-scan on every open.
      let hasSuperscripts = false
      if (looksBad || hasMarkers) {
        try {
          hasSuperscripts = await _docHasSuperscripts(rawDoc, resumePage)
        } catch { /* fall through to the looksBad gate */ }
      }
      if (looksBad || hasMarkers || hasSuperscripts) {
        const total = await _reextractAndShip(rawDoc, this.bookId)
        if (total) {
          try {
            const br = await fetch(`/api/books/${this.bookId}?include_text=true`)
            if (br.ok) {
              const b = await br.json()
              this.sentences = b.sentences || []
              this.pageMap = b.page_map || []
              this.book = b
              // Re-point resume state at the freshly extracted sentence
              // list (indices shifted after re-extraction).
              this.currentPdfPage = Math.max(1, b.current_page || this.currentPdfPage || 1)
              this.currentIdx = this._pickStartIdx(b.current_sentence_idx || 0)
            }
          } catch { /* keep going with old sentences */ }
        }
      }
    },

    // Called from the viewer's `textlayerrendered` event. Caches the
    // page's text spans for highlighting and applies any queued highlight.
    _onTextLayerRendered(e) {
      const raw = _raw(this)
      const pageView = e.source
      if (!pageView || !pageView.textLayer) return
      const texts = pageView.textLayer.textContentItemsStr || []
      const textDivs = pageView.textLayer.textDivs || []
      if (!texts.length || !textDivs.length) return
      raw.tlCache.set(e.pageNumber, { texts, textDivs })
      // Apply a queued highlight (or re-apply the active one) once the
      // layer is in the DOM — this is what makes a jump-then-highlight
      // land on a page that was still rendering.
      const pending = raw.pendingHl
      raw.pendingHl = null
      if (pending && pending.pageNum === e.pageNumber) {
        this._applyHighlight(pending.pageNum, pending.sentence)
      } else if (raw.lastHl && raw.lastHl.pageNum === e.pageNumber) {
        this._applyHighlight(raw.lastHl.pageNum, raw.lastHl.sentence)
      }
    },

    // Highlight `sentence` on `pageNum`, jumping the viewer there first
    // if it is off-screen (the jump triggers lazy-render; the render
    // event applies the highlight when the text layer lands).
    _highlightSentence(pageNum, sentence) {
      const raw = _raw(this)
      if (!raw.viewer || !pageNum || !sentence) return
      raw.lastHl = { pageNum, sentence }
      if (raw.viewer.currentPageNumber !== pageNum) {
        raw.viewer.currentPageNumber = pageNum
      }
      this._applyHighlight(pageNum, sentence)
    },

    // Apply a highlight directly on a page that's already rendered.
    // Clears the previous highlight across all cached pages first, then
    // toggles .pdf-highlight on the matching <span>s.
    _applyHighlight(pageNum, sentence) {
      const raw = _raw(this)
      if (!sentence) return
      // Clear highlights on every cached page (only one is active at a
      // time, but old pages may still hold lingering spans).
      for (const entry of raw.tlCache.values()) {
        for (const span of entry.textDivs) {
          if (span && span.classList) span.classList.remove('pdf-highlight')
        }
      }
      const entry = raw.tlCache.get(pageNum)
      if (!entry) {
        // Page not yet rendered; queue and bail — textlayerrendered
        // applies it.
        raw.pendingHl = { pageNum, sentence }
        return
      }
      const idxs = _findMatchTexts(entry.texts, sentence)
      if (!idxs) return
      for (const i of idxs) {
        const span = entry.textDivs[i]
        if (span && span.classList) span.classList.add('pdf-highlight')
      }
    },

    // Jump to a page number. Used by nav buttons, outline clicks, and
    // the keyboard handler. Safe no-op until the viewer is ready.
    goToPage(n) {
      const raw = _raw(this)
      if (!raw.viewer || !raw.viewer.pdfDocument || !raw.viewer.pagesCount) return
      const max = raw.viewer.pagesCount
      const clamped = Math.max(1, Math.min(max, Math.round(Number(n) || 1)))
      if (raw.viewer.currentPageNumber !== clamped) {
        raw.viewer.currentPageNumber = clamped
      }
    },

    // ─── PDF zoom ──────────────────────────────────────────────────
    // PDFViewer's currentScale is a plain multiplier, so zooming is just
    // assigning a number; the viewer re-renders + re-scales the text
    // layers (which re-fires textlayerrendered, re-applying the active
    // sentence highlight). Safe no-ops until the viewer is ready.
    _setZoom(scale) {
      const raw = _raw(this)
      if (!raw.viewer || !raw.viewer.pdfDocument) return
      raw.viewer.currentScale = scale
      this.zoomPct = Math.round(scale * 100)
    },

    zoomIn() {
      const raw = _raw(this)
      const base = raw.viewer ? raw.viewer.currentScale || 1 : 1
      this._setZoom(Math.min(5, +(base + 0.25).toFixed(2)))
    },

    zoomOut() {
      const raw = _raw(this)
      const base = raw.viewer ? raw.viewer.currentScale || 1 : 1
      this._setZoom(Math.max(0.25, +(base - 0.25).toFixed(2)))
    },

    // Reset to a literal 100% (1:1 pixels).
    zoomReset() {
      this._setZoom(1)
    },

    // Fit the page to the container width (the reader's default view).
    // Can't be represented as a plain number, hence the value string;
    // assigning 'auto' re-fits even after the user zoomed in.
    zoomFit() {
      const raw = _raw(this)
      if (!raw.viewer || !raw.viewer.pdfDocument) return
      raw.viewer.currentScaleValue = 'auto'
      this.zoomPct = Math.round((raw.viewer.currentScale || 1) * 100)
    },

    // Persist the current PDF page to the server, throttled. Fire-and-
    // forget; never blocks TTS playback. The user can close the reader
    // mid-read and resume from the last page.
    _persistPage(pageNum) {
      if (!this.bookId) return
      this._pendingPage = pageNum
      if (this._pageTimer) return
      this._pageTimer = setTimeout(() => {
        this._pageTimer = null
        this._flushPersist()
      }, 500)
    },

    _flushPersist() {
      if (this._pageTimer) {
        clearTimeout(this._pageTimer)
        this._pageTimer = null
      }
      const pageNum = this._pendingPage
      this._pendingPage = null
      if (!this.bookId || !pageNum) return
      try {
        fetch(`/api/books/${this.bookId}/progress`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ current_page: pageNum }),
        }).catch(() => { /* offline is fine */ })
      } catch { /* noop */ }
    },

    // Click handler for outline items. outlineItems[i] is the reactive
    // (title/depth) mirror; the raw PDF.js item with its destination ref
    // lives in _raw(this).outlineFlat[i].ref, indexed the same way.
    async _gotoOutlineItemAt(i) {
      const raw = _raw(this)
      const entry = raw.outlineFlat && raw.outlineFlat[i]
      const ref = entry && entry.ref
      if (!raw.linkService || !raw.linkService.pdfDocument || !ref || ref.dest == null) return
      try {
        // Outline `dest` may be a string (named destination) or an
        // already-resolved destination array — PDFLinkService.goToDestination
        // handles both.
        await raw.linkService.goToDestination(ref.dest)
      } catch { /* destination that cannot resolve is a no-op */ }
    },

    // ─── View helpers ──────────────────────────────────────────────
    progressPct() {
      if (!this.sentences.length) return 0
      return Math.min(100, Math.round((this.currentIdx / this.sentences.length) * 100))
    },

    // Formatted article view (url/text saves with an HTML snapshot).
    hasArticle() { return !!(this.book && this.book.has_article) },
    articleUrl() { return this.bookId ? `/api/books/${this.bookId}/article` : '' },
    // Blocked-site entry: no extractable sentences, just the source link.
    isLinkCard() {
      return !!(this.book && !this.sentences.length &&
        (this.book.file_type === 'url' || this.book.file_type === 'text') &&
        this.book.source_url)
    },

    // Sentence list helpers (EPUB only — PDF uses the in-PDF text layer)
    sentenceText(i) {
      const s = this.sentences[i]
      if (!s) return ''
      return (s.text || '').replace(/\s+/g, ' ').trim()
    },
    isCurrent(i) { return i === this.currentIdx && this.isPlaying },
    isPast(i) { return i < this.currentIdx },
  }
}

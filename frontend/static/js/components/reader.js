/**
 * Reader overlay — fullscreen PDF reader with TTS read-aloud and
 * in-PDF sentence highlight.
 *
 * The PDF is rendered with PDF.js 3.11 (UMD, classic worker) using
 * its standard multi-page text-layer pattern: each page is a
 * <div class="pdf-page"> containing a <canvas> for the visual and a
 * <div class="textLayer"> for selectable text. The text layer is a
 * series of absolutely-positioned <span>s — one per PDF text item —
 * which gives us a free, precise handle for highlighting. We just add
 * a CSS class to the spans that overlap the current TTS sentence.
 *
 * Why not the browser's native <iframe> viewer? It gives scroll/zoom/
 * outline/etc. for free, but the browser doesn't expose a way to draw
 * a highlight on a specific span in the rendered PDF. PDF.js' text
 * layer does, at the cost of writing the viewer ourselves.
 *
 * Why a WeakMap for the PDFDocumentProxy? PDF.js classes use private
 * fields (#d, #u, etc.). When Alpine's reactive Proxy wraps the proxy
 * object, those private-field accesses raise "Cannot read private
 * member #X from an object whose class did not declare it" — because
 * the Proxy's class is not PDFDocumentProxy. The standard fix (per
 * mozilla/pdf.js#20478) is to keep the raw proxy off `this` on a
 * module-level WeakMap keyed by the component instance.
 *
 * Lifecycle:
 *   1. `open-reader` event with {bookId} → fetch the book + file
 *   2. PDF.js renders pages on demand (current ± 2 kept warm)
 *   3. Server streams one NDJSON event per sentence {idx, page, sentence,
 *      seg, url} → play each <audio> URL, highlight the matching spans
 *   4. Progress is persisted per sentence on the server
 */
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

// Module-level WeakMap for non-reactive storage of PDF.js objects.
// Keyed by the Alpine component instance (`this` in the factory).
const _rawPdf = new WeakMap()
function _raw(self) {
  if (!_rawPdf.has(self)) _rawPdf.set(self, { doc: null, cache: new Map() })
  return _rawPdf.get(self)
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

// Extract every page's text via PDF.js (clean per-page text) and re-ship
// the resulting sentences + page_map to the server. The server's PyPDF2
// fallback produces badly-mangled text for some print-typeset PDFs;
// this replaces those rows with the same sentences the user sees in the
// reader. Returns the new total sentence count, or null on failure.
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
      // PDF.js emits one item per text run. hasEOL is the only reliable
      // newline signal — concat runs with single space, push at hasEOL.
      const lines = []
      let buf = ''
      for (const it of tc.items) {
        buf += (it.str || '')
        if (it.hasEOL) {
          lines.push(buf)
          buf = ''
        } else {
          buf += ' '
        }
      }
      if (buf) lines.push(buf)
      const pageText = lines.join('\n').replace(/[ \t]+/g, ' ').trim()
      if (!pageText) continue
      const sents = _splitSentencesClient(pageText)
      let charStart = 0
      for (const s of sents) {
        if (s.length < 2) { charStart += s.length + 1; continue }
        allSents.push({ text: s, page: p, char_start: charStart })
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

// Fuzzy match a sentence against concatenated text items. Returns the
// [itemIndexStart, itemIndexEnd] pair that covers the longest matching
// prefix of the sentence (we match a prefix because TTS text and PDF
// text often differ at the end — TTS may have "Hello world." while
// PDF.js has "Hello world!"). Length thresholds fall back 30→15→8.
function _findMatchItems(items, sentence) {
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim()
  const target = norm(sentence)
  if (!target) return null
  const pageText = items.map((i) => i.str).join(' ').replace(/\s+/g, ' ').toLowerCase()
  const targetLower = target.toLowerCase()
  let startIdx = -1
  for (const n of [Math.min(30, target.length), Math.min(15, target.length), Math.min(8, target.length)]) {
    if (n < 3) continue
    startIdx = pageText.indexOf(targetLower.slice(0, n))
    if (startIdx >= 0) break
  }
  if (startIdx < 0) return null
  const endIdx = startIdx + target.length
  // Map the page-text char index back to which text items overlap.
  let cursor = 0
  const matched = []
  for (let i = 0; i < items.length; i++) {
    const s = items[i].str
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

    // TTS playback state
    audio: null,
    segments: [],
    segCursor: 0,
    streamDone: false,
    pendingPlay: false,
    isPlaying: false,
    _abort: null,

    // PDF state — only safe primitives are reactive; PDF.js objects
    // live in the WeakMap.
    pdfReady: false,
    pdfNumPages: 0,
    currentPdfPage: 1,
    pageMap: [],

    init() {
      window.addEventListener('open-reader', (e) => this.start(e.detail?.bookId))
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

      try {
        const r = await fetch(`/api/books/${bookId}?include_text=true`)
        if (!r.ok) throw new Error('Failed to load book: ' + r.status)
        const book = await r.json()
        this.book = book
        this.sentences = book.sentences || []
        this.pageMap = book.page_map || []
        this.currentIdx = Math.max(0, Math.min(
          book.current_sentence_idx || 0,
          Math.max(0, this.sentences.length - 1)
        ))
        this.currentPdfPage = Math.max(1, book.current_page || 1)
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
      this._clearPdfCache()
      const raw = _raw(this)
      if (raw.doc) {
        try { raw.doc.destroy() } catch { /* noop */ }
        raw.doc = null
      }
      this.audio = null
      this.pdfReady = false
      this.pdfNumPages = 0
      this.pageMap = []
      this.open = false
      this.book = null
      this.bookId = null
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
      if (this.segments.length === 0) {
        await this._startStream(this.currentIdx)
      } else {
        this.audio.play().catch(() => { /* user gesture will fix */ })
        this.isPlaying = true
      }
    },

    pause() {
      if (this.audio && !this.audio.paused) this.audio.pause()
      this.isPlaying = false
    },

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

    // ─── Stream → segment queue → audio ────────────────────────────
    async _startStream(fromIdx) {
      this._abort = new AbortController()
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
      // If the user paused between segments, currentIdx may be ahead
      // of segCursor; just keep them in sync.
      // For PDF: ensure the page is rendered, scroll to it, highlight.
      if (this.book?.file_type === 'pdf' && seg.page) {
        await this._showPage(seg.page)
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
      this.pdfReady = true
      // Render the resumed page so the viewer isn't blank.
      await this._renderPage(this.currentPdfPage)
      await this._showPage(this.currentPdfPage)
      // Re-extract sentences client-side if the server's PyPDF2 fallback
      // produced too few (mangled) or too many (per-line) sentences. The
      // client has the same PDF and can run PDF.js' getTextContent per
      // page, which gives clean per-line text that the splitter turns
      // into real sentences. The re-extracted set ships to the server so
      // /api/books/{id}/stream uses the clean version from then on.
      const pages = this.pdfNumPages
      const cachedSents = this.sentences.length
      const looksBad =
        cachedSents < Math.max(50, pages / 2) || cachedSents > pages * 10
      if (looksBad) {
        const total = await _reextractAndShip(rawDoc, this.bookId)
        if (total) {
          // Re-fetch the book so we get the new sentences.
          try {
            const br = await fetch(`/api/books/${this.bookId}?include_text=true`)
            if (br.ok) {
              const b = await br.json()
              this.sentences = b.sentences || []
              this.pageMap = b.page_map || []
              this.book = b
            }
          } catch { /* keep going with old sentences */ }
        }
      }
    },

    // Render a single page (canvas + text layer) and cache it. Returns
    // the entry from the cache.
    async _renderPage(pageNum) {
      const raw = _raw(this)
      if (!raw.doc) return null
      const cache = raw.cache
      if (cache.has(pageNum)) return cache.get(pageNum)

      const page = await raw.doc.getPage(pageNum)
      const viewport = page.getViewport({ scale: 1.4 })

      // Canvas for the visual
      const canvas = document.createElement('canvas')
      canvas.width = viewport.width
      canvas.height = viewport.height
      canvas.className = 'block'
      const ctx = canvas.getContext('2d')
      await page.render({ canvasContext: ctx, viewport }).promise

      // Text layer: PDF.js builds absolutely-positioned <span>s that
      // match the visual. We use the same scale as the canvas so the
      // text aligns pixel-perfect. PDF.js requires --scale-factor set
      // to the same value as viewport.scale; without it the text layer
      // logs an error and the spans end up with NaN positions.
      const textLayerDiv = document.createElement('div')
      textLayerDiv.className = 'textLayer'
      textLayerDiv.style.width = viewport.width + 'px'
      textLayerDiv.style.height = viewport.height + 'px'
      textLayerDiv.style.setProperty('--scale-factor', String(viewport.scale))

      const textContent = await page.getTextContent()
      // textDivs collects the per-item <span>s PDF.js creates, in item
      // order. We index into it to find which spans overlap a sentence.
      const textDivs = []
      await window.pdfjsLib.renderTextLayer({
        textContentSource: textContent,
        container: textLayerDiv,
        viewport,
        textDivs,
      }).promise

      // Container <div> positions the canvas + text layer on top of
      // each other. Both use the same dimensions and scale.
      const container = document.createElement('div')
      container.className = 'pdf-page relative mx-auto shadow-md bg-white mb-4'
      container.style.width = viewport.width + 'px'
      container.style.height = viewport.height + 'px'
      container.dataset.page = String(pageNum)
      container.appendChild(canvas)
      container.appendChild(textLayerDiv)

      const items = textContent.items
        .filter((it) => it.str !== undefined)
        .map((it) => ({
          str: it.str,
          // e, f in transform[4], [5] are x, y in PDF user space
          x: it.transform[4],
          y: it.transform[5],
          width: it.width || 0,
          height: it.height || Math.abs(it.transform[3]) || 12,
        }))
      const entry = { container, canvas, textLayerDiv, textDivs, items, viewport }
      cache.set(pageNum, entry)
      if (cache.size > 10) {
        const oldest = cache.keys().next().value
        cache.delete(oldest)
      }
      return entry
    },

    // Ensure `pageNum` is rendered and visible in the viewer. Pre-renders
    // neighbors so a TTS-driven page flip is instant.
    async _showPage(pageNum) {
      const raw = _raw(this)
      if (!raw.doc) return
      if (pageNum < 1 || pageNum > this.pdfNumPages) return
      this.currentPdfPage = pageNum
      const neighbors = [pageNum - 1, pageNum, pageNum + 1].filter(
        (n) => n >= 1 && n <= this.pdfNumPages
      )
      await Promise.all(neighbors.map((n) => this._renderPage(n)))
      // Add the new page's container if not already in the viewer.
      const viewer = this.$refs.pdfViewer
      if (!viewer) return
      for (const n of neighbors) {
        const entry = raw.cache.get(n)
        if (!entry) continue
        if (!entry.container.parentNode) {
          viewer.appendChild(entry.container)
        }
      }
      // Scroll the page into view. We smooth-scroll rather than jump.
      const entry = raw.cache.get(pageNum)
      if (entry) {
        entry.container.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    },

    // Remove every cached page's container from the DOM and clear the
    // cache. Used on close() and on big enough state changes.
    _clearPdfCache() {
      const raw = _raw(this)
      for (const entry of raw.cache.values()) {
        if (entry.container.parentNode) {
          entry.container.parentNode.removeChild(entry.container)
        }
      }
      raw.cache.clear()
    },

    // Find the spans that overlap `sentence` and toggle the highlight
    // class. The previous highlight (on any page) is cleared first.
    _highlightSentence(pageNum, sentence) {
      const raw = _raw(this)
      // Clear highlights on every cached page (only one is active at a
      // time, but old pages may still have lingering spans).
      for (const entry of raw.cache.values()) {
        for (const span of entry.textDivs) {
          if (span && span.classList) span.classList.remove('pdf-highlight')
        }
      }
      const entry = raw.cache.get(pageNum)
      if (!entry || !sentence) return
      const idxs = _findMatchItems(entry.items, sentence)
      if (!idxs) return
      for (const i of idxs) {
        const span = entry.textDivs[i]
        if (span && span.classList) span.classList.add('pdf-highlight')
      }
    },

    // ─── View helpers ──────────────────────────────────────────────
    progressPct() {
      if (!this.sentences.length) return 0
      return Math.min(100, Math.round((this.currentIdx / this.sentences.length) * 100))
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
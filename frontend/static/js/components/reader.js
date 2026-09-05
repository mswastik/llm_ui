/**
 * Reader overlay — fullscreen book viewer with TTS read-aloud.
 *
 * The PDF/EPUB is displayed in a native <iframe> using the browser's
 * built-in viewer, so the user gets a real PDF reader: scrolling pages,
 * page numbers, zoom, search, outline (when the PDF has one), text
 * selection, thumbnails sidebar. We don't try to render PDF.js ourselves
 * — that path kept running into Alpine/Proxy issues with PDF.js private
 * fields, and the browser's native viewer already does the job.
 *
 * TTS runs separately: the server streams one NDJSON event per
 * sentence ({idx, page, sentence, seg, url}); the shared <audio> element
 * plays the segment URL, and a "now reading" banner above the iframe
 * shows the sentence the TTS is currently speaking. The user follows
 * along by eye in the iframe; for an in-PDF highlight overlay, future
 * iteration can postMessage the page into the iframe after TTS advances
 * (Chrome's PDF viewer accepts #page=N, so this is cheap to add later).
 *
 * Lifecycle:
 *   1. User opens a book from the library → `open-reader` event with {bookId}
 *   2. We fetch the book (sentences + page_map) and open the iframe
 *   3. Each NDJSON event pushes an audio URL into a segment queue
 *   4. The <audio> element plays segments in order; the banner updates
 *      per segment
 *   5. Progress is persisted server-side per sentence, so a disconnect
 *      mid-read resumes from `current_sentence_idx` on reopen
 */
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
    audio: null,             // shared <audio> element
    segments: [],            // queue of {url, idx, page, sentence}
    segCursor: 0,
    streamDone: false,
    pendingPlay: false,
    isPlaying: false,
    _abort: null,

    // Sentence banner: the text currently being spoken, plus its page
    // (so the user can scroll the iframe to follow along).
    currentSentence: '',
    currentSentencePage: 0,
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
      this.currentSentence = ''
      this.currentSentencePage = 0

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
        this._ensureAudio()
      } catch (e) {
        this.error = e.message || 'Failed to load book'
        this.isLoading = false
        return
      }
      this.isLoading = false
    },

    close() {
      this.stop()
      this.audio = null
      this.open = false
      this.book = null
      this.bookId = null
      this.sentences = []
      this.pageMap = []
      this.segments = []
      this.segCursor = 0
      this.currentIdx = 0
      this.currentSentence = ''
      this.currentSentencePage = 0
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
      // If the queue is empty, kick off the stream from the current sentence.
      if (this.segments.length === 0) {
        await this._startStream(this.currentIdx)
      } else {
        // Resume: just play the audio. If we paused mid-segment, the
        // current <audio>.src is already set, so this restarts it.
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

    _playCurrent() {
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
      this.currentSentence = seg.sentence
      this.currentSentencePage = seg.page
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
      // One bad segment shouldn't kill the whole stream — skip and continue.
      this.segCursor += 1
      if (this.segCursor < this.segments.length) this._playCurrent()
      else this.isPlaying = false
    },

    // ─── Iframe URL ────────────────────────────────────────────────
    // Open the file in the browser's native viewer. #page=N is honored
    // by both Chrome's and Firefox's built-in PDF viewers; if the PDF
    // doesn't support it (or the browser ignores it), the user still
    // gets a normal view and can scroll/zoom themselves.
    fileSrc() {
      if (!this.book) return ''
      const base = `/api/books/${this.book.id}/file`
      if (this.book.file_type === 'pdf' && this.book.current_page > 1) {
        return `${base}#page=${this.book.current_page}`
      }
      return base
    },

    // ─── View helpers ──────────────────────────────────────────────
    progressPct() {
      if (!this.sentences.length) return 0
      return Math.min(100, Math.round((this.currentIdx / this.sentences.length) * 100))
    },

    // Current sentence's text in display form (collapsed whitespace).
    currentSentenceText() {
      return (this.currentSentence || '').replace(/\s+/g, ' ').trim()
    },
  }
}

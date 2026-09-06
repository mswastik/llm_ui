/**
 * Library panel — books (PDF/EPUB), saved web pages, and pasted notes for
 * the read-aloud reader.
 *
 * Main-pane view (not a modal): owns search / type-filter / sort so it
 * scales to hundreds of web saves. Click hands off to the reader overlay
 * via the `open-reader` event; the reader streams /api/books/{id}/stream.
 */
import { api } from '../utils.js'

export const library = () => ({
  uploading: false,
  savingUrl: false,
  savingText: false,
  error: '',
  showAdd: false,
  // Toolbar state
  search: '',
  filter: 'all', // all | pdf | epub | url | text
  sort: 'newest', // newest | oldest | title | progress
  // Add-form state
  urlInput: '',
  pasteTitle: '',
  pasteText: '',
  // Hub picker: section-front URL returned article links to choose from
  hubTitle: '',
  hubLinks: [],
  hubSaving: '',

  async init() {
    await this.refresh()
  },

  async refresh() {
    try {
      const data = await api.get('/api/books')
      this.$store.ui.books = data.books || []
    } catch (e) {
      this.error = e.message
    }
  },

  // ─── List: filter + sort ──────────────────────────────────
  get filteredBooks() {
    let list = [...(this.$store.ui.books || [])]
    if (this.filter !== 'all') {
      list = list.filter((b) => (b.file_type || '') === this.filter)
    }
    if (this.search.trim()) {
      const q = this.search.trim().toLowerCase()
      list = list.filter((b) =>
        (b.title || '').toLowerCase().includes(q) ||
        (b.domain || '').toLowerCase().includes(q) ||
        (b.source_url || '').toLowerCase().includes(q),
      )
    }
    const by = {
      newest: (a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0),
      oldest: (a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0),
      title: (a, b) => (a.title || '').localeCompare(b.title || ''),
      progress: (a, b) => this.progressPct(b) - this.progressPct(a),
    }
    return list.sort(by[this.sort] || by.newest)
  },

  // ─── Origin metadata ──────────────────────────────────────
  sourceIcon(book) {
    const t = book.file_type || ''
    if (t === 'url') return 'ph ph-globe'
    if (t === 'text') return 'ph ph-note-pencil'
    if (t === 'epub') return 'ph ph-book'
    return 'ph ph-file-pdf'
  },

  sourceLabel(book) {
    const t = book.file_type || ''
    if (t === 'url') {
      const base = book.domain || book.source_url || 'Web page'
      return book.total_sentences ? base : base + ' · Link'
    }
    if (t === 'text') return 'Pasted note'
    return (t || 'file').toUpperCase()
  },

  // ─── Mutations ────────────────────────────────────────────
  async uploadFile(file) {
    if (!file) return
    this.uploading = true
    this.error = ''
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch('/api/books/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      this.$store.ui.books.unshift(data.book)
      this.$store.ui.showToast(`"${data.book.title}" added to library`, 'success')
    } catch (e) {
      this.$store.ui.showToast('Upload error: ' + e.message, 'error')
    } finally {
      this.uploading = false
    }
  },

  async saveFromUrl(url) {
    url = (url !== undefined ? url : this.urlInput || '').trim()
    if (!url || this.savingUrl) return
    this.savingUrl = true
    this.error = ''
    try {
      const data = await api.post('/api/books/from-url', { url })
      if (data.status === 'hub') {
        // Section front: offer the article links instead of saving thin text
        this.hubTitle = data.title || url
        this.hubLinks = data.links || []
        this.hubSaving = ''
        this.$store.ui.showToast(`Found ${this.hubLinks.length} articles — pick one to save`, 'info')
        return
      }
      if (data.status === 'link_card') {
        this.$store.ui.books.unshift(data.book)
        this.$store.ui.showToast(data.detail || `"${data.book.title}" saved as link`, 'info')
        this.urlInput = ''
        this.hubLinks = []
        return
      }
      if (!data.deduped) this.$store.ui.books.unshift(data.book)
      this.$store.ui.showToast(
        data.deduped ? `"${data.book.title}" is already in your library` : `"${data.book.title}" saved`,
        'success',
      )
      this.urlInput = ''
      this.hubLinks = []
    } catch (e) {
      this.error = e.message
      this.$store.ui.showToast('Save failed: ' + e.message, 'error')
    } finally {
      this.savingUrl = false
    }
  },

  async saveHubLink(link) {
    if (!link || this.hubSaving) return
    this.hubSaving = link.url
    try {
      const data = await api.post('/api/books/from-url', { url: link.url })
      if (data.status === 'hub') {
        this.$store.ui.showToast('That link is another hub — open it in a browser instead', 'info')
        return
      }
      if (data.status === 'link_card') {
        this.$store.ui.books.unshift(data.book)
      } else if (!data.deduped) {
        this.$store.ui.books.unshift(data.book)
      }
      this.hubLinks = this.hubLinks.filter((l) => l.url !== link.url)
      this.$store.ui.showToast(`"${(data.book || {}).title || link.title}" saved`, 'success')
      if (!this.hubLinks.length) this.hubTitle = ''
    } catch (e) {
      this.$store.ui.showToast('Save failed: ' + e.message, 'error')
    } finally {
      this.hubSaving = ''
    }
  },

  async saveFromText() {
    const text = (this.pasteText || '').trim()
    if (text.length < 20 || this.savingText) return
    this.savingText = true
    this.error = ''
    try {
      const data = await api.post('/api/books/from-text', { title: this.pasteTitle.trim(), text })
      this.$store.ui.books.unshift(data.book)
      this.$store.ui.showToast(`"${data.book.title}" saved`, 'success')
      this.pasteTitle = ''
      this.pasteText = ''
    } catch (e) {
      this.error = e.message
      this.$store.ui.showToast('Save failed: ' + e.message, 'error')
    } finally {
      this.savingText = false
    }
  },

  async deleteBook(book) {
    if (!confirm(`Delete "${book.title}"? This removes the file too.`)) return
    try {
      await api.delete(`/api/books/${book.id}`)
      this.$store.ui.books = this.$store.ui.books.filter((b) => b.id !== book.id)
      this.$store.ui.showToast('Removed from library', 'success')
    } catch (e) {
      this.$store.ui.showToast('Delete failed: ' + e.message, 'error')
    }
  },

  async reextractBook(book) {
    if (!confirm(`Re-extract "${book.title}"? Resets reading progress.`)) return
    try {
      const r = await api.post(`/api/books/${book.id}/reextract`)
      this.$store.ui.showToast(`Re-extracted: ${r.total_sentences} sentences`, 'success')
      await this.refresh()
    } catch (e) {
      this.$store.ui.showToast('Re-extract failed: ' + e.message, 'error')
    }
  },

  openBook(book) {
    // Hand off to the reader overlay. The reader owns its own state and
    // calls /api/books/{id}/stream — it does not need any data from here.
    this.$dispatch('open-reader', { bookId: book.id })
  },

  // Where the user is in the item, as a percentage. Used in the library
  // card so resuming is visually obvious.
  progressPct(book) {
    if (!book.total_sentences) return 0
    return Math.min(100, Math.round((book.current_sentence_idx / book.total_sentences) * 100))
  },

  progressLabel(book) {
    if (!book.total_sentences) return 'New'
    const pct = this.progressPct(book)
    if (pct === 0) return 'New'
    if (pct >= 99) return 'Finished'
    return `${pct}% · sentence ${book.current_sentence_idx}/${book.total_sentences}`
  },
})

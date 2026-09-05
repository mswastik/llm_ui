/**
 * Library Modal — upload/list PDF & EPUB books for the read-aloud reader.
 *
 * Thin wrapper around the modal factory: it holds the book list + an
 * upload handler, and dispatches `open-reader` with the book id when the
 * user clicks a book. The reader overlay (components/reader.js) listens
 * for that event and takes over the fullscreen view.
 */
import { api } from '../utils.js'

export const library = () => ({
  uploading: false,
  error: '',

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

  async deleteBook(book) {
    if (!confirm(`Delete "${book.title}"? This removes the file too.`)) return
    try {
      await api.delete(`/api/books/${book.id}`)
      this.$store.ui.books = this.$store.ui.books.filter((b) => b.id !== book.id)
      this.$store.ui.showToast('Book removed', 'success')
    } catch (e) {
      this.$store.ui.showToast('Delete failed: ' + e.message, 'error')
    }
  },

  openBook(book) {
    // Hand off to the reader overlay. The reader owns its own state and
    // calls /api/books/{id}/stream — it does not need any data from here.
    this.$dispatch('open-reader', { bookId: book.id })
  },

  // Where the user is in the book, as a percentage. Used in the library
  // card so resuming a book is visually obvious.
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

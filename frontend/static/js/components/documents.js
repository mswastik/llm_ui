/**
 * Documents Component - Knowledge base / RAG document management
 */
import { formatters, api } from '../utils.js'

console.log('[DOCUMENTS] Module loading...')

// Define component factory as global function
window.documents = () => {
  console.log('[DOCUMENTS] Component factory called')
  return {
    // Local state
    show: false,
    list: [],

    // Initialization
    async init() {
      console.log('[DOCUMENTS] init() called')
      console.log('[DOCUMENTS] $store available:', !!this.$store)
      
      // Sync from store
      this.show = this.$store.documents.show
      this.list = this.$store.documents.list
      
      console.log('[DOCUMENTS] Initial state:', { show: this.show, list: this.list?.length })
      
      await this.loadDocuments()
    },

    // Documents
    async loadDocuments() {
      console.log('[DOCUMENTS] loadDocuments() called')
      try {
        const data = await api.get('/api/documents')
        this.list = data.documents
        this.$store.documents.list = data.documents
        console.log('[DOCUMENTS] Loaded documents:', this.list.length)
      } catch (error) {
        console.error('[DOCUMENTS] Error loading documents:', error)
      }
    },

    async uploadDocument(file) {
      console.log('[DOCUMENTS] uploadDocument() called with:', file?.name)
      if (!file) return
      
      const formData = new FormData()
      formData.append('file', file)
      
      try {
        const response = await fetch('/api/documents/upload', { 
          method: 'POST', 
          body: formData 
        })
        
        if (response.ok) {
          const data = await response.json()
          this.list.unshift(data.document)
          this.$store.documents.list = this.list
          this.$store.chat.showToast('Document uploaded!', 'success')
        } else {
          const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
          this.$store.chat.showToast(`Error: ${error.detail}`, 'error')
        }
      } catch (error) {
        console.error('[DOCUMENTS] Error uploading:', error)
        this.$store.chat.showToast('Upload failed', 'error')
      }
    },

    async deleteDocument(documentId) {
      console.log('[DOCUMENTS] deleteDocument() called with:', documentId)
      if (!confirm('Delete this document?')) return
      try {
        await api.delete(`/api/documents/${documentId}`)
        this.list = this.list.filter(d => d.id !== documentId)
        this.$store.documents.list = this.list
      } catch (error) {
        console.error('[DOCUMENTS] Error deleting document:', error)
        this.$store.chat.showToast('Failed to delete document', 'error')
      }
    },

    // Utilities
    formatFileSize: (bytes) => formatters.formatFileSize(bytes),
    formatDate: (isoString) => formatters.formatDate(isoString)
  }
}

console.log('[DOCUMENTS] window.documents defined:', typeof window.documents)

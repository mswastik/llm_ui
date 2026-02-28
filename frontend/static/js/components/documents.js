/**
 * Documents Component - Knowledge base / RAG document management
 */
import { formatters, api } from '../utils.js'


// Define component factory as global function
window.documents = () => {
  
  const component = {
    // Local state
    list: [],
    
    // show is a getter/setter that syncs with store
    get show() {
      return this.$store?.documents?.show || false
    },
    set show(val) {
      this.$store.documents.show = val
    },

    // Initialization
    async init() {
      
      // Sync from store
      this.list = this.$store.documents.list
      
      
      await this.loadDocuments()
    },

    // Documents
    async loadDocuments() {
      try {
        const data = await api.get('/api/documents')
        this.list = data.documents
        this.$store.documents.list = data.documents
      } catch (error) {
        console.error('[DOCUMENTS] Error loading documents:', error)
      }
    },

    async uploadDocument(file) {
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
  
  return component
}


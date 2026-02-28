/**
 * Sidebar Component - Conversation management
 */
import { api } from '../utils.js'
import { formatters } from '../utils.js'

// Define component
document.addEventListener('alpine:init', () => {
  Alpine.data('sidebar', () => ({
    // Local state
    conversations: [],
    currentConversationId: null,
    sidebarCollapsed: false,
    sidebarWidth: 280,
    isResizing: false,
    currentConversationTitle: 'New Chat',
    minWidth: 200,
    maxWidth: 500,

    // Initialization
    async init() {
      try {
        // Load saved width from localStorage
        const savedWidth = localStorage.getItem('sidebarWidth')
        if (savedWidth) {
          this.sidebarWidth = parseInt(savedWidth, 10)
        }

        // Sync from store
        this.conversations = this.$store.chat.conversations || []
        this.currentConversationId = this.$store.chat.currentConversationId
        this.sidebarCollapsed = this.$store.chat.sidebarCollapsed
        this.currentConversationTitle = this.$store.chat.currentConversationTitle

        // Add global mouse listeners for resize
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e))
        document.addEventListener('mouseup', () => this.handleMouseUp())

        await this.loadConversations()
      } catch (error) {
        console.error('[sidebar] init error:', error)
      }
    },

    // Resize handlers
    startResize(e) {
      this.isResizing = true
      e.preventDefault()
      e.stopPropagation()
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    },

    handleMouseMove(e) {
      if (!this.isResizing) return

      const newWidth = e.clientX
      if (newWidth >= this.minWidth && newWidth <= this.maxWidth) {
        this.sidebarWidth = newWidth
        this.sidebarCollapsed = false
        localStorage.setItem('sidebarWidth', newWidth)
      }
    },

    handleMouseUp() {
      if (this.isResizing) {
        this.isResizing = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
    },

    // Conversations
    async loadConversations() {
      try {
        const data = await api.get('/api/conversations')
        this.conversations = data.conversations || []
        this.$store.chat.conversations = this.conversations
      } catch (error) {
        console.error('[sidebar] Error loading conversations:', error)
        this.conversations = []
      }
    },

    async createNewConversation() {
      try {
        const data = await api.post('/api/conversations', { title: 'New Chat' })
        this.$store.chat.addConversation(data.conversation)
        this.conversations = this.$store.chat.conversations
        await this.loadConversation(data.conversation.id)
      } catch (error) {
        console.error('[sidebar] Error creating conversation:', error)
        this.$store.chat.showToast('Failed to create conversation', 'error')
      }
    },

    async loadConversation(conversationId) {
      // Close any active SSE streams
      if (window.sseService) window.sseService.close()

      this.$store.chat.isLoading = false
      this.$store.chat.toolStatus.active = false

      try {
        const data = await api.get(`/api/conversations/${conversationId}`)
        this.$store.chat.currentConversationId = conversationId
        this.$store.chat.currentConversationTitle = data.conversation.title
        this.$store.chat.messages = data.messages

        // Update local state
        this.currentConversationId = conversationId
        this.currentConversationTitle = data.conversation.title

        this.$nextTick(() => {
          const container = document.getElementById('messages-container')
          if (container) container.scrollTop = container.scrollHeight
        })
      } catch (error) {
        console.error('[sidebar] Error loading conversation:', error)
        this.$store.chat.showToast('Failed to load conversation', 'error')
      }
    },

    async deleteConversation(conversationId, event) {
      event?.stopPropagation()
      if (!confirm('Delete this conversation?')) return

      try {
        await api.delete(`/api/conversations/${conversationId}`)
        this.$store.chat.removeConversation(conversationId)
        this.conversations = this.$store.chat.conversations

        if (conversationId === this.$store.chat.currentConversationId) {
          if (this.$store.chat.conversations.length > 0) {
            await this.loadConversation(this.$store.chat.conversations[0].id)
          } else {
            await this.createNewConversation()
          }
        }
      } catch (error) {
        console.error('[sidebar] Error deleting conversation:', error)
        this.$store.chat.showToast('Failed to delete conversation', 'error')
      }
    },

    async updateConversationTitle(conversationId, event) {
      const newTitle = event.target.value.trim()
      if (!newTitle) return

      try {
        await api.put(`/api/conversations/${conversationId}`, { title: newTitle })
        this.$store.chat.updateConversation(conversationId, { title: newTitle })

        if (conversationId === this.$store.chat.currentConversationId) {
          this.$store.chat.currentConversationTitle = newTitle
          this.currentConversationTitle = newTitle
        }
      } catch (error) {
        console.error('[sidebar] Error updating title:', error)
        this.$store.chat.showToast('Failed to update title', 'error')
      }
    },

    async deleteMessage(messageId, event) {
      event?.stopPropagation()
      if (!confirm('Delete this message?')) return
      try {
        await api.delete(`/api/messages/${messageId}`)
        this.$store.chat.removeMessage(messageId)
      } catch (error) {
        console.error('[sidebar] Error deleting message:', error)
        this.$store.chat.showToast('Failed to delete message', 'error')
      }
    },

    // UI helpers
    toggleSidebar() {
      this.sidebarCollapsed = !this.sidebarCollapsed
      this.$store.chat.sidebarCollapsed = this.sidebarCollapsed
    },

    formatDate: (isoString) => formatters.formatDate(isoString)
  }))
})

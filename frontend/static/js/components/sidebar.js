/**
 * Sidebar Component - Conversation management
 */
import { api } from '../utils.js'
import { formatters } from '../utils.js'

console.log('[SIDEBAR] Module loading...')

// Define component factory
window.sidebar = () => {
  console.log('[SIDEBAR] Component factory called')
  return {
    // Local state
    conversations: [],
    currentConversationId: null,
    sidebarCollapsed: false,
    currentConversationTitle: 'New Chat',

    // Initialization
    async init() {
      console.log('[SIDEBAR] init() called')
      console.log('[SIDEBAR] $store available:', !!this.$store)
      console.log('[SIDEBAR] $store.chat available:', !!this.$store?.chat)
      
      // Sync from store
      this.conversations = this.$store.chat.conversations
      this.currentConversationId = this.$store.chat.currentConversationId
      this.sidebarCollapsed = this.$store.chat.sidebarCollapsed
      this.currentConversationTitle = this.$store.chat.currentConversationTitle
      
      console.log('[SIDEBAR] Initial state:', {
        conversations: this.conversations.length,
        currentConversationId: this.currentConversationId
      })
      
      await this.loadConversations()
    },

    // Conversations
    async loadConversations() {
      console.log('[SIDEBAR] loadConversations() called')
      try {
        const data = await api.get('/api/conversations')
        console.log('[SIDEBAR] Loaded conversations:', data.conversations.length)
        this.conversations = data.conversations
        this.$store.chat.conversations = data.conversations
      } catch (error) {
        console.error('[SIDEBAR] Error loading conversations:', error)
      }
    },

    async createNewConversation() {
      console.log('[SIDEBAR] createNewConversation() called')
      try {
        const data = await api.post('/api/conversations', { title: 'New Chat' })
        this.$store.chat.addConversation(data.conversation)
        this.conversations = this.$store.chat.conversations
        await this.loadConversation(data.conversation.id)
      } catch (error) {
        console.error('[SIDEBAR] Error creating conversation:', error)
        this.$store.chat.showToast('Failed to create conversation', 'error')
      }
    },

    async loadConversation(conversationId) {
      console.log('[SIDEBAR] loadConversation() called with id:', conversationId)
      // Close any active SSE streams
      if (window.sseService) window.sseService.close()
      
      this.$store.chat.isLoading = false
      this.$store.chat.toolStatus.active = false
      
      try {
        const data = await api.get(`/api/conversations/${conversationId}`)
        console.log('[SIDEBAR] Loaded conversation:', data.conversation.title)
        console.log('[SIDEBAR] Messages count:', data.messages?.length || 0)
        console.log('[SIDEBAR] First message:', data.messages?.[0])
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
        console.error('[SIDEBAR] Error loading conversation:', error)
        this.$store.chat.showToast('Failed to load conversation', 'error')
      }
    },

    async deleteConversation(conversationId, event) {
      event?.stopPropagation()
      console.log('[SIDEBAR] deleteConversation() called with id:', conversationId)
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
        console.error('[SIDEBAR] Error deleting conversation:', error)
        this.$store.chat.showToast('Failed to delete conversation', 'error')
      }
    },

    async updateConversationTitle(conversationId, event) {
      const newTitle = event.target.value.trim()
      if (!newTitle) return
      
      console.log('[SIDEBAR] updateConversationTitle() called with:', newTitle)
      try {
        await api.put(`/api/conversations/${conversationId}`, { title: newTitle })
        this.$store.chat.updateConversation(conversationId, { title: newTitle })
        
        if (conversationId === this.$store.chat.currentConversationId) {
          this.$store.chat.currentConversationTitle = newTitle
          this.currentConversationTitle = newTitle
        }
      } catch (error) {
        console.error('[SIDEBAR] Error updating title:', error)
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
        console.error('[SIDEBAR] Error deleting message:', error)
        this.$store.chat.showToast('Failed to delete message', 'error')
      }
    },

    // UI helpers
    toggleSidebar() {
      console.log('[SIDEBAR] toggleSidebar() called')
      this.sidebarCollapsed = !this.sidebarCollapsed
      this.$store.chat.sidebarCollapsed = this.sidebarCollapsed
    },

    formatDate: (isoString) => formatters.formatDate(isoString)
  }
}

console.log('[SIDEBAR] window.sidebar defined:', typeof window.sidebar)

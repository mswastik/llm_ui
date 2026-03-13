/**
 * Sidebar Component - Conversation management
 */
import { api } from '../utils.js'
import { formatters } from '../utils.js'

// Define component factory function
const sidebarComponent = () => ({
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
    // Check if there's active streaming for THIS conversation
    const isActiveStreamingForThisConversation = 
      this.$store.chat.activeStreaming?.isStreaming && 
      this.$store.chat.activeStreaming?.conversationId === conversationId

    // If actively streaming to this conversation, restore messages from storage
    if (isActiveStreamingForThisConversation) {
      console.log('[sidebar] Active streaming detected for THIS conversation, restoring stored messages')
      this.$store.chat.isLoading = true
      this.$store.chat.currentConversationId = conversationId
      this.$store.chat.currentConversationTitle = this.$store.chat.activeStreaming.conversationTitle || 'Streaming...'
      // Restore messages from storage
      this.$store.chat.messages = [...this.$store.chat.activeStreaming.messages]
      this.currentConversationId = conversationId
      this.currentConversationTitle = this.$store.chat.currentConversationTitle
      return
    }

    // No active streaming for this conversation, load from DB
    console.log('[sidebar] Loading conversation from DB:', conversationId)
    this.$store.chat.isLoading = false

    try {
      const data = await api.get(`/api/conversations/${conversationId}`)
      this.$store.chat.currentConversationId = conversationId
      this.$store.chat.currentConversationTitle = data.conversation.title

      // Normalize messages - ensure blocks are properly assigned
      const normalizedMessages = data.messages.map(msg => {
        // First, check if blocks exist in metadata from API
        const blocksFromApi = msg.blocks || (msg.metadata && msg.metadata.blocks) || null
        
        console.log('[sidebar] Message:', msg.id, 'blocks from API:', blocksFromApi)
        
        // If blocks exist from API, use them directly
        if (blocksFromApi && Array.isArray(blocksFromApi) && blocksFromApi.length > 0) {
          console.log('[sidebar] Using blocks directly:', blocksFromApi.map(b => b.type))
          return {
            ...msg,
            blocks: blocksFromApi,
            tool_calls: msg.tool_calls || [] // Keep tool_calls for backward compatibility
          }
        }
        
        console.log('[sidebar] No blocks found, message has:', {
          has_tool_calls: !!msg.tool_calls,
          has_thinking: !!msg.thinking,
          tool_calls_count: msg.tool_calls?.length || 0
        })
        
        // Otherwise, normalize from old format (backward compatibility during transition)
        const actualToolCalls = (msg.tool_calls || []).filter(tc =>
          tc.type === 'tool_call' || tc.name === 'search_web' || tc.name === 'query_documents' || tc.type === 'thinking'
        )

        const normalizedToolCalls = actualToolCalls.map((toolCall) => {
          let searchSteps = toolCall.search_steps || []
          let searchTerms = toolCall.search_terms || []
          let reasoning = toolCall.reasoning
          let coverageScore = toolCall.coverage_score
          let sources = toolCall.sources || []
          let name = toolCall.name || toolCall.type || 'tool'

          if (toolCall.result) {
            sources = toolCall.result.sources || sources
            if (toolCall.result.search_steps) {
              searchSteps = toolCall.result.search_steps
            }
            if (toolCall.result.search_terms_used) {
              searchTerms = toolCall.result.search_terms_used
            }
          }

          if (toolCall.progress_history && toolCall.progress_history.length > 0) {
            for (let i = toolCall.progress_history.length - 1; i >= 0; i--) {
              const progress = toolCall.progress_history[i]
              if (progress.data) {
                if (progress.data.search_steps && progress.data.search_steps.length > 0) {
                  searchSteps = progress.data.search_steps
                }
                if (progress.data.search_terms && progress.data.search_terms.length > 0) {
                  searchTerms = progress.data.search_terms
                }
                if (progress.data.reasoning && !reasoning) {
                  reasoning = progress.data.reasoning
                }
                if (progress.data.coverage_score !== undefined && coverageScore === undefined) {
                  coverageScore = progress.data.coverage_score
                }
                if (progress.result?.sources && progress.result.sources.length > 0) {
                  sources = progress.result.sources
                }
              }
            }
          }

          return {
            ...toolCall,
            name: name,
            search_steps: searchSteps,
            search_terms: searchTerms,
            reasoning: reasoning,
            coverage_score: coverageScore,
            sources: sources
          }
        })

        return {
          ...msg,
          tool_calls: normalizedToolCalls
        }
      })

      this.$store.chat.messages = normalizedMessages
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
    if (this.sidebarCollapsed===true) {
      this.sidebarWidth = 60;
      localStorage.setItem('sidebarWidth', 60);
    }
    else {
      this.sidebarWidth = 220;
      localStorage.setItem('sidebarWidth', 220);
    }
  },

  formatDate: (isoString) => formatters.formatDate(isoString)
})

// Register with Alpine.js - handle both sync and async loading
if (window.Alpine) {
  window.Alpine.data('sidebar', sidebarComponent)
} else {
  document.addEventListener('alpine:init', () => {
    window.Alpine.data('sidebar', sidebarComponent)
  })
}

// Export for potential external use
export { sidebarComponent }

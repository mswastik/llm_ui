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

        console.log('[loadConversation] Raw messages from API:', JSON.stringify(data.messages, null, 2))

        // Normalize messages - ensure tool_calls is always an array and extract search_steps from progress_history
        const normalizedMessages = data.messages.map(msg => {
          console.log('[loadConversation] Processing message:', msg.role, 'tool_calls:', msg.tool_calls)

          // Filter to only actual tool calls (not content or thinking blocks)
          const actualToolCalls = (msg.tool_calls || []).filter(tc => 
            tc.type === 'tool_call' || tc.name === 'search_web' || tc.name === 'query_documents' || tc.type === 'thinking'
          )
          
          console.log('[loadConversation] Filtered tool calls:', actualToolCalls.length, 'from', msg.tool_calls?.length)

          const normalizedToolCalls = actualToolCalls.map((toolCall, idx) => {
            console.log('[loadConversation] toolCall[' + idx + ']:', JSON.stringify(toolCall, null, 2))

            // Handle different data structures
            // Structure 1: Direct properties (new format)
            let searchSteps = toolCall.search_steps || []
            let searchTerms = toolCall.search_terms || []
            let reasoning = toolCall.reasoning
            let coverageScore = toolCall.coverage_score
            let sources = toolCall.sources || []
            let name = toolCall.name || toolCall.type || 'tool'

            // Check if this is the result-based format
            if (toolCall.result) {
              sources = toolCall.result.sources || sources
              if (toolCall.result.search_steps) {
                searchSteps = toolCall.result.search_steps
              }
              if (toolCall.result.search_terms_used) {
                searchTerms = toolCall.result.search_terms_used
              }
            }

            console.log('[loadConversation] Processing toolCall:', name, 'has progress_history:', !!toolCall.progress_history)

            // Extract data from progress_history for legacy messages
            if (toolCall.progress_history && toolCall.progress_history.length > 0) {
              // Find the LAST progress event that contains search_steps (most complete data)
              for (let i = toolCall.progress_history.length - 1; i >= 0; i--) {
                const progress = toolCall.progress_history[i]
                console.log('[loadConversation] Checking progress event:', progress.type, progress.tool, 'data:', progress.data)

                if (progress.data) {
                  // Get search_steps from this progress event
                  if (progress.data.search_steps && progress.data.search_steps.length > 0) {
                    searchSteps = progress.data.search_steps
                    console.log('[loadConversation] Found search_steps:', searchSteps.length, 'steps')
                  }
                  // Get search_terms
                  if (progress.data.search_terms && progress.data.search_terms.length > 0) {
                    searchTerms = progress.data.search_terms
                  }
                  // Get reasoning
                  if (progress.data.reasoning && !reasoning) {
                    reasoning = progress.data.reasoning
                  }
                  // Get coverage_score
                  if (progress.data.coverage_score !== undefined && coverageScore === undefined) {
                    coverageScore = progress.data.coverage_score
                  }
                  // Get sources from result
                  if (progress.result?.sources && progress.result.sources.length > 0) {
                    sources = progress.result.sources
                  }
                }
              }
            }
            
            console.log('[loadConversation] Normalized toolCall:', {
              name: name,
              type: toolCall.type,
              search_steps: searchSteps?.length,
              search_terms: searchTerms?.length,
              has_reasoning: !!reasoning
            })
            
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
        
        console.log('[loadConversation] Normalized messages:', normalizedMessages)
        this.$store.chat.messages = normalizedMessages

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

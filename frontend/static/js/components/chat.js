/**
 * Chat Component - Core chat functionality
 */
import { sseService } from '../services/sse.js'
import { formatters, markdownUtils, helpers, api } from '../utils.js'
import { ttsService } from '../services/tts.js'

// Define component factory as global function
window.chat = () => {
  // Create reactive component
  const component = {
    // Local state
    isLoading: false,
    toolStatus: { active: false, tool: '', status: '', progress: null, data: null },
    selectedModel: '',
    selectedAgentId: null,
    selectedDocumentIds: 'all',
    enableWebSearch: false,
    enableRAG: false,
    editingMessageId: null,
    editContent: '',
    inputMessage: '',
    availableModels: [],
    availableAgents: [],
    documents: [],
    expandedToolCallBlocks: {},
    expandedThinkingBlocks: {},
    expandedThinking: {},
    expandedSources: {},
    currentConversationTitle: 'New Chat',

    // Messages is a getter that always reads from store
    get messages() {
      return this.$store?.chat?.messages || []
    },
    set messages(val) {
      this.$store.chat.messages = val
    },

    // Initialization
    async init() {
      // Sync from store
      this.isLoading = this.$store.chat.isLoading
      this.toolStatus = { ...this.$store.chat.toolStatus }
      this.selectedModel = this.$store.chat.selectedModel
      this.selectedAgentId = this.$store.chat.selectedAgentId
      this.enableWebSearch = this.$store.chat.enableWebSearch
      this.enableRAG = this.$store.chat.enableRAG
      this.availableModels = this.$store.chat.availableModels
      this.availableAgents = this.$store.chat.availableAgents
      this.currentConversationTitle = this.$store.chat.currentConversationTitle
      this.documents = this.$store.documents.list || []

      await Promise.all([this.loadModels(), this.loadAgents(), this.loadDocuments()])
      this.$store.chat.loadSavedModel()
      this.selectedModel = this.$store.chat.selectedModel
      this.$store.chat.loadSavedAgent()
      this.selectedAgentId = this.$store.chat.selectedAgentId
      this.applyAgentConfig()

      await ttsService.checkAvailability()
      this.$store.tts.available = ttsService.ttsAvailable

      // Check if there's an active stream and start polling for updates
      this.checkAndReattachActiveStream()
    },

    // Check for active streaming and start polling for updates
    async checkAndReattachActiveStream() {
      const activeStreaming = this.$store.chat.activeStreaming
      if (!activeStreaming.isStreaming) return

      // Start polling for this conversation's messages
      console.log('[Chat] Active stream detected for conversation:', activeStreaming.conversationId)
      this.startStreamingPolling(activeStreaming.requestId, activeStreaming.conversationId, activeStreaming.msgIndex)
    },

    startStreamingPolling(requestId, conversationId, msgIndex) {
      // Clear any existing polling
      this.stopStreamingPolling()

      // Poll for updated messages while streaming
      const pollInterval = setInterval(async () => {
        const activeStreaming = this.$store.chat.activeStreaming
        if (!activeStreaming.isStreaming || activeStreaming.conversationId !== conversationId) {
          clearInterval(pollInterval)
          this.streamingPollInterval = null
          return
        }

        try {
          // Fetch latest messages to get updated tool results and content
          const data = await api.get(`/api/conversations/${conversationId}`)
          const latestMessages = data.messages || []

          if (latestMessages.length > msgIndex) {
            const latestMsg = latestMessages[msgIndex]
            const currentMsg = this.$store.chat.messages[msgIndex]

            if (latestMsg && currentMsg) {
              let hasChanges = false

              // Update content if changed
              if (latestMsg.content && latestMsg.content !== currentMsg.content) {
                this.$store.chat.messages[msgIndex].content = latestMsg.content
                hasChanges = true
              }

              // Update tool_calls if changed (more complete data from DB)
              if (latestMsg.tool_calls && latestMsg.tool_calls.length > 0) {
                // Merge tool calls - prefer latest data for existing calls
                const currentToolCalls = currentMsg.tool_calls || []
                const mergedToolCalls = []

                for (let i = 0; i < Math.max(currentToolCalls.length, latestMsg.tool_calls.length); i++) {
                  const current = currentToolCalls[i]
                  const latest = latestMsg.tool_calls[i]

                  if (latest && !current) {
                    // New tool call from DB
                    mergedToolCalls.push(latest)
                  } else if (current && !latest) {
                    // Tool call exists in current but not DB (still streaming)
                    mergedToolCalls.push(current)
                  } else if (current && latest) {
                    // Merge: prefer latest result data, keep current progress
                    mergedToolCalls.push({
                      ...current,
                      ...latest,
                      // Keep progress_history from current if latest doesn't have it
                      progress_history: latest.progress_history || current.progress_history,
                      // Keep result from latest (more complete)
                      result: latest.result || current.result
                    })
                  }
                }

                if (JSON.stringify(mergedToolCalls) !== JSON.stringify(currentToolCalls)) {
                  this.$store.chat.messages[msgIndex].tool_calls = mergedToolCalls
                  hasChanges = true
                }
              }

              // Update thinking if changed
              if (latestMsg.thinking && latestMsg.thinking !== currentMsg.thinking) {
                this.$store.chat.messages[msgIndex].thinking = latestMsg.thinking
                hasChanges = true
              }

              if (hasChanges) {
                this.messages = [...this.$store.chat.messages]
                console.log('[Chat] Messages updated from polling')
              }
            }
          }
        } catch (error) {
          console.error('[Chat] Polling error:', error)
        }
      }, 1000) // Poll every second

      // Store interval ID for cleanup
      this.streamingPollInterval = pollInterval
      console.log('[Chat] Started polling for stream updates')
    },

    stopStreamingPolling() {
      if (this.streamingPollInterval) {
        clearInterval(this.streamingPollInterval)
        this.streamingPollInterval = null
        console.log('[Chat] Stopped polling for stream updates')
      }
    },

    // Models
    async loadModels() {
      try {
        const data = await api.get('/api/models')
        this.availableModels = data.models || []
        this.$store.chat.availableModels = this.availableModels
      } catch (error) {
        console.error('Error loading models:', error)
      }
    },

    updateSelectedModel() {
      this.$store.chat.setModel(this.selectedModel)
    },

    // Agents
    async loadAgents() {
      try {
        const data = await api.get('/api/agents')
        this.availableAgents = data.agents || []
        this.$store.chat.availableAgents = this.availableAgents
      } catch (error) {
        console.error('Error loading agents:', error)
      }
    },

    updateSelectedAgent() {
      this.$store.chat.setAgent(this.selectedAgentId)
      this.applyAgentConfig()
    },

    applyAgentConfig() {
      const agent = this.$store.chat.currentAgentConfig
      if (!agent) return

      // Apply agent configuration to current chat settings
      if (agent.model && this.availableModels.some(m => m.id === agent.model)) {
        this.selectedModel = agent.model
        this.$store.chat.selectedModel = agent.model
      }
      this.enableWebSearch = !!agent.enable_web_search
      this.enableRAG = !!agent.enable_rag
      this.$store.chat.enableWebSearch = this.enableWebSearch
      this.$store.chat.enableRAG = this.enableRAG
    },

    // Documents
    async loadDocuments() {
      try {
        const data = await api.get('/api/documents')
        this.documents = data.documents || []
        this.$store.documents.list = this.documents
      } catch (error) {
        console.error('Error loading documents:', error)
      }
    },

    async updateConversationTitle(conversationId, event) {
      const newTitle = event.target.value.trim()
      if (!newTitle) return
      
      try {
        await api.put(`/api/conversations/${conversationId}`, { title: newTitle })
        this.$store.chat.updateConversation(conversationId, { title: newTitle })
        this.$store.chat.currentConversationTitle = newTitle
        this.currentConversationTitle = newTitle
      } catch (error) {
        console.error('Error updating title:', error)
        this.$store.chat.showToast('Failed to update title', 'error')
      }
    },

    // Sending messages
    async sendMessage(inputMessage) {
      if (!inputMessage?.trim() || this.isLoading) return
      
      const messageText = inputMessage.trim()
      this.inputMessage = ''
      this.isLoading = true
      this.$store.chat.isLoading = true
      
      const userMessage = {
        id: helpers.generateId(),
        role: 'user',
        content: messageText,
        created_at: new Date().toISOString()
      }
      this.$store.chat.addMessage(userMessage)
      this.messages = this.$store.chat.messages
      
      try {
        const data = await api.post(`/api/conversations/${this.$store.chat.currentConversationId}/messages`, {
          message: messageText,
          enable_web_search: this.enableWebSearch,
          enable_rag: this.enableRAG
        })
        await this.streamResponse(data.request_id)
      } catch (error) {
        console.error('Error sending message:', error)
        this.isLoading = false
        this.$store.chat.isLoading = false
        this.$store.chat.showToast('Failed to send message', 'error')
      }
    },

    async streamResponse(requestId) {
      const options = {
        enableWebSearch: this.enableWebSearch,
        enableRag: this.enableRAG,
        model: this.selectedModel
      }

      // Await the stream method to get handlers
      const handlers = await sseService.stream(requestId, this.$store.chat.currentConversationId, options)

      const assistantMessage = {
        id: helpers.generateId() + 1,
        role: 'assistant',
        content: '',
        thinking: '',
        tool_calls: [],
        created_at: new Date().toISOString()
      }
      this.$store.chat.addMessage(assistantMessage)
      this.messages = this.$store.chat.messages
      const msgIndex = this.messages.length - 1

      // Store streaming state with current messages
      this.$store.chat.startStreaming(requestId, this.$store.chat.currentConversationId, msgIndex, this.$store.chat.currentConversationTitle, [...this.$store.chat.messages])

      handlers.onData((data) => {
        this.processStreamEvent(data, msgIndex)
      })

      handlers.onError((error) => {
        console.error('Stream error:', error)
        this.$store.chat.stopStreaming()
        this.$store.chat.messages[msgIndex].content += `\n\n❌ Error: ${error.message}`
        this.$store.chat.showToast(`Stream error: ${error.message}`, 'error')
        this.messages = [...this.$store.chat.messages]
      })

      handlers.onComplete(() => {
        console.log('Stream completed')
        this.$store.chat.stopStreaming()
      })
    },

    // Core streaming logic
    processStreamEvent(data, msgIndex) {
      console.log('[Chat] processStreamEvent:', data.type, data)

      // Check if we're currently on the streaming conversation
      const activeStreaming = this.$store.chat.activeStreaming
      if (activeStreaming.isStreaming &&
          activeStreaming.conversationId !== this.$store.chat.currentConversationId) {
        // We're not on the streaming conversation, skip processing
        // Events will be picked up via polling when we return
        console.log('[Chat] Not on streaming conversation, skipping event')
        return
      }

      // Check if message exists at this index
      if (!this.$store.chat.messages[msgIndex]) {
        console.warn('[Chat] Message at index', msgIndex, 'not found, skipping event')
        return
      }

      const msg = this.$store.chat.messages[msgIndex]
      
      // Initialize blocks array if not present (for sequential rendering)
      if (!msg.blocks) {
        msg.blocks = []
      }

      switch (data.type) {
        case 'content':
          // Check if last block is a content block - if so, append to it
          const lastBlock = msg.blocks.length > 0 ? msg.blocks[msg.blocks.length - 1] : null
          if (lastBlock && lastBlock.type === 'content') {
            lastBlock.content += data.content
          } else {
            // Create new content block
            msg.blocks.push({ type: 'content', content: data.content })
          }
          console.log('[Chat] Content updated, blocks:', msg.blocks.length)
          break
        case 'thinking':
          // Check if last block is a thinking block - if so, append to it
          const lastThinkingBlock = msg.blocks.length > 0 ? msg.blocks[msg.blocks.length - 1] : null
          if (lastThinkingBlock && lastThinkingBlock.type === 'thinking') {
            lastThinkingBlock.content += data.content
          } else {
            // Create new thinking block
            msg.blocks.push({ type: 'thinking', content: data.content })
          }
          console.log('[Chat] Thinking updated, blocks:', msg.blocks.length)
          break
        case 'tool_call_start':
          this.$store.chat.toolStatus.active = true
          this.$store.chat.toolStatus.tool = data.tool
          this.$store.chat.toolStatus.status = 'Starting...'
          this.toolStatus = { ...this.$store.chat.toolStatus }

          const newToolCall = {
            type: 'tool_call',
            name: data.tool,
            arguments: data.args,
            status: 'starting',
            progress: 0,
            result: null,
            progress_history: [{ status: 'starting', progress: 0 }]
          }
          msg.blocks.push(newToolCall)
          console.log('[Chat] Tool call started:', data.tool)
          break
        case 'tool_progress':
          this.$store.chat.toolStatus.status = data.status
          this.$store.chat.toolStatus.progress = data.progress || null
          this.toolStatus = { ...this.$store.chat.toolStatus }

          // Find the last incomplete tool call block
          const currentToolCall = msg.blocks.findLast(bc =>
            bc.type === 'tool_call' &&
            bc.status !== 'completed' &&
            bc.status !== 'error'
          )

          if (currentToolCall) {
            currentToolCall.status = data.status
            currentToolCall.progress = data.progress || 0

            // Store search steps if available
            if (data.data) {
              if (data.data.search_steps) {
                currentToolCall.search_steps = data.data.search_steps
              }
              if (data.data.search_terms) {
                currentToolCall.search_terms = data.data.search_terms
              }
              if (data.data.reasoning) {
                currentToolCall.reasoning = data.data.reasoning
              }
              if (data.data.coverage_score) {
                currentToolCall.coverage_score = data.data.coverage_score
              }
            }

            if (data.result) {
              currentToolCall.result = data.result
              currentToolCall.status = 'completed'
              // Store sources from result for citation display
              if (data.result.sources && data.result.sources.length > 0) {
                currentToolCall.sources = data.result.sources
              }
            }
            console.log('[Chat] Tool progress:', data.status, data.progress)
          }
          if (data.result) {
            this.$store.chat.toolStatus.active = false
            this.toolStatus = { ...this.$store.chat.toolStatus }
          }
          break
        case 'error':
          this.$store.chat.toolStatus.active = false
          this.toolStatus = { ...this.$store.chat.toolStatus }
          this.isLoading = false
          this.$store.chat.isLoading = false
          // Add error as a content block
          msg.blocks.push({ type: 'content', content: `\n\n❌ Error: ${data.error}` })
          this.$store.chat.showToast(`Error: ${data.error}`, 'error')
          console.log('[Chat] Error:', data.error)
          break
        case 'title_update':
          this.$store.chat.currentConversationTitle = data.title
          const convIndex = this.$store.chat.conversations.findIndex(c => c.id === this.$store.chat.currentConversationId)
          if (convIndex !== -1) {
            this.$store.chat.conversations[convIndex].title = data.title
          }
          console.log('[Chat] Title updated:', data.title)
          break
        case 'done':
          sseService.close()
          this.isLoading = false
          this.$store.chat.isLoading = false
          this.$store.chat.toolStatus.active = false
          this.toolStatus = { ...this.$store.chat.toolStatus }
          console.log('[Chat] Stream done')
          break
      }

      // Force reactivity by creating a new array reference
      this.messages = [...this.$store.chat.messages]
      console.log('[Chat] Messages array updated, length:', this.messages.length)
      // Update stored messages for streaming
      this.$store.chat.updateStreamingMessages(this.$store.chat.messages)
      // Scroll to bottom after update
      this.$nextTick(() => {
        const container = document.getElementById('messages-container')
        //if (container) {
        //  container.scrollTop = container.scrollHeight
        //}
      })
    },

    // Message actions
    async deleteMessage(messageId, event) {
      event?.stopPropagation()
      if (!confirm('Delete this message?')) return
      try {
        await api.delete(`/api/messages/${messageId}`)
        this.$store.chat.removeMessage(messageId)
        this.messages = this.$store.chat.messages
      } catch (error) {
        console.error('Error deleting message:', error)
        this.$store.chat.showToast('Failed to delete message', 'error')
      }
    },

    startEditMessage(messageId, content) {
      this.editingMessageId = messageId
      this.editContent = content
    },

    cancelEdit() {
      this.editingMessageId = null
      this.editContent = ''
    },

    async saveEdit(messageId) {
      if (!this.editContent.trim()) {
        this.cancelEdit()
        return
      }
      
      const msg = this.$store.chat.messages.find(m => m.id === this.editingMessageId)
      if (!msg) { this.cancelEdit(); return }
      
      if (msg.role === 'user') {
        if (this.editContent.trim() !== msg.content.trim()) {
          await this.forkConversation(this.editingMessageId, this.editContent.trim())
        } else {
          this.cancelEdit()
        }
        return
      }
      
      try {
        const data = await api.put(`/api/messages/${this.editingMessageId}`, { 
          content: this.editContent 
        })
        msg.content = data.message.content
      } catch (error) {
        console.error('Error updating message:', error)
        this.$store.chat.showToast('Failed to update message', 'error')
      }
      this.cancelEdit()
    },

    async forkConversation(originalMessageId, newContent) {
      try {
        const data = await api.post('/api/conversations', { 
          title: 'Forked: ' + newContent.substring(0, 30) + '...' 
        })
        const newConversationId = data.conversation.id
        this.$store.chat.addConversation(data.conversation)
        
        await api.post(`/api/conversations/${newConversationId}/messages`, { message: newContent })
        const streamData = await api.post(`/api/conversations/${newConversationId}/messages`, { message: newContent })
        
        this.$store.chat.currentConversationId = newConversationId
        this.$store.chat.currentConversationTitle = data.conversation.title
        this.$store.chat.clearMessages()
        this.$store.chat.addMessage({ 
          id: helpers.generateId(), 
          role: 'user', 
          content: newContent, 
          created_at: new Date().toISOString() 
        })
        this.messages = this.$store.chat.messages
        
        await this.streamResponse(streamData.request_id)
      } catch (error) {
        console.error('Error forking:', error)
        this.$store.chat.showToast('Failed to fork conversation', 'error')
      }
      this.cancelEdit()
    },

    async regenerateResponse(messageId) {
      if (this.isLoading) return
      
      this.isLoading = true
      this.$store.chat.isLoading = true
      try {
        const data = await api.post(`/api/conversations/${this.$store.chat.currentConversationId}/regenerate`, { 
          message_id: messageId 
        })
        
        const msgIndex = this.messages.findIndex(m => m.id === messageId)
        if (msgIndex !== -1) {
          this.$store.chat.messages = this.$store.chat.messages.slice(0, msgIndex + 1)
          this.messages = this.$store.chat.messages
        }
        
        const handlers = sseService.stream(data.request_id, this.$store.chat.currentConversationId, { 
          model: this.selectedModel 
        })
        
        const assistantMessage = { 
          id: helpers.generateId() + 1, 
          role: 'assistant', 
          content: '', 
          thinking: '', 
          tool_calls: [], 
          created_at: new Date().toISOString() 
        }
        this.$store.chat.addMessage(assistantMessage)
        this.messages = this.$store.chat.messages
        handlers.onData((d) => this.processStreamEvent(d, this.messages.length - 1))
      } catch (error) {
        console.error('Error regenerating:', error)
        this.isLoading = false
        this.$store.chat.isLoading = false
        this.$store.chat.showToast('Failed to regenerate', 'error')
      }
    },

    cancelRequest() {
      sseService.close()
      this.isLoading = false
      this.$store.chat.isLoading = false
      this.$store.chat.toolStatus.active = false
      this.toolStatus = { ...this.$store.chat.toolStatus }
      
      const lastMessage = this.$store.chat.messages[this.$store.chat.messages.length - 1]
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.content.trim() === '') {
        lastMessage.content = '⚠️ Request cancelled.'
      }
      this.$store.chat.showToast('Request cancelled', 'info')
    },

    // TTS
    async speakMessage(message) {
      const success = await ttsService.speak(message, (error) => 
        this.$store.chat.showToast(error, 'error')
      )
      if (success) {
        this.$store.tts.currentAudio = ttsService.currentAudio
        this.$store.tts.currentAudioMessageId = ttsService.currentAudioMessageId
        this.$store.tts.isPlaying = ttsService.isPlaying
      }
    },

    stopAudio() {
      ttsService.stop()
      this.$store.tts.cleanup()
    },

    // Utilities
    renderMarkdown: (text) => markdownUtils.render(text),
    renderMarkdownWithCitations: (text, sources) => markdownUtils.renderWithCitations(text, sources),
    getMessageSources: (message) => markdownUtils.getMessageSources(message),
    getMessageSourcesFromBlocks: (blocks) => markdownUtils.getMessageSourcesFromBlocks(blocks),
    formatDate: (isoString) => formatters.formatDate(isoString),
    
    async copyMessage(message) {
      const type = message.role === 'user' ? 'Query' : 'Response'
      const success = await helpers.copyToClipboard(message.content)
      this.$store.chat.showToast(success ? `${type} copied!` : 'Copy failed', success ? 'success' : 'error')
    },

    // Tool call display helpers
    shouldShowToolCalls(message) {
      return (message?.tool_calls && message.tool_calls.length > 0 && 
              this.editingMessageId !== message.id)
    },

    toggleToolCallBlock(messageId, blockIndex) {
      const key = helpers.createExpansionKey(messageId, blockIndex)
      helpers.toggleExpansion(this.expandedToolCallBlocks, key)
    },

    isToolCallBlockExpanded(messageId, blockIndex) {
      return helpers.isExpanded(this.expandedToolCallBlocks, 
                                helpers.createExpansionKey(messageId, blockIndex))
    },

    toggleThinkingBlock(messageId, blockIndex) {
      const key = helpers.createExpansionKey(messageId, blockIndex)
      helpers.toggleExpansion(this.expandedThinkingBlocks, key)
    },

    isThinkingBlockExpanded(messageId, blockIndex) {
      return helpers.isExpanded(this.expandedThinkingBlocks, 
                                helpers.createExpansionKey(messageId, blockIndex))
    },

    toggleThinking(messageId) {
      helpers.toggleExpansion(this.expandedThinking, messageId)
    },

    isThinkingExpanded(messageId) {
      return helpers.isExpanded(this.expandedThinking, messageId)
    },

    toggleSources(messageId) {
      helpers.toggleExpansion(this.expandedSources, messageId)
    },

    isSourcesExpanded(messageId) {
      return helpers.isExpanded(this.expandedSources, messageId)
    }
  }

  return component
}

/**
 * Chat Component — Messages, streaming, tools, TTS
 */
import { sseService } from '../services/sse.js'
import { ttsService } from '../services/tts.js'
import { formatters, markdownUtils, helpers, api } from '../utils.js'

const chatComponent = () => ({
  // Local state synced from store
  isLoading: false,
  toolStatus: { active: false, tool: '', status: '', progress: null },
  selectedModel: '',
  selectedAgentId: null,
  selectedDocumentIds: [],
  editingMessageId: null,
  editContent: '',
  inputMessage: '',
  availableModels: [],
  availableAgents: [],
  documents: [],
  currentConversationTitle: 'New Chat',

  // Expanded state for collapsible blocks
  expandedBlocks: {},

  // ─── Getters (reactive to store) ──────────────────────
  get messages() {
    return this.$store.chat.messages || []
  },
  set messages(val) {
    this.$store.chat.messages = val
  },

  get isRAGActive() {
    return this.selectedDocumentIds.length > 0
  },

  // ─── Init ─────────────────────────────────────────────
  async init() {
    this.syncFromStore()
    await Promise.all([this.loadModels(), this.loadAgents(), this.loadDocuments()])
    this.$store.chat.loadSavedModel()
    this.selectedModel = this.$store.chat.selectedModel
    this.$store.chat.loadSavedAgent()
    this.selectedAgentId = this.$store.chat.selectedAgentId
    this.selectedDocumentIds = [...(this.$store.chat.selectedDocumentIds || [])]
    await ttsService.checkAvailability()
    this.checkActiveStream()

    window.addEventListener('sync-agent', (e) => {
      this.selectedAgentId = e.detail.agentId
    })
  },

  syncFromStore() {
    this.isLoading = this.$store.chat.isLoading
    this.toolStatus = { ...this.$store.chat.toolStatus }
    this.selectedModel = this.$store.chat.selectedModel
    this.selectedAgentId = this.$store.chat.selectedAgentId
    this.selectedDocumentIds = [...(this.$store.chat.selectedDocumentIds || [])]
    this.availableModels = this.$store.chat.availableModels
    this.availableAgents = this.$store.chat.availableAgents
    this.currentConversationTitle = this.$store.chat.currentConversationTitle
    this.documents = this.$store.ui.documents || []
  },

  // ─── Data Loading ─────────────────────────────────────
  async loadModels() {
    try {
      const data = await api.get('/api/models')
      this.availableModels = data.models || []
      this.$store.chat.availableModels = this.availableModels
    } catch (e) { console.error('[chat] Models:', e) }
  },

  async loadAgents() {
    try {
      const data = await api.get('/api/agents')
      this.availableAgents = data.agents || []
      this.$store.chat.availableAgents = this.availableAgents
    } catch (e) { console.error('[chat] Agents:', e) }
  },

  async loadDocuments() {
    try {
      const data = await api.get('/api/documents')
      this.documents = data.documents || []
      this.$store.ui.documents = this.documents
    } catch (e) { console.error('[chat] Documents:', e) }
  },

  // ─── Active Stream Check ──────────────────────────────
  checkActiveStream() {
    const active = this.$store.chat.activeStreaming
    if (!active.isStreaming) return
    // Polling is no longer needed — SSE handles real-time updates
    // But we restore messages if we navigated away during streaming
    if (active.conversationId !== this.$store.chat.currentConversationId) {
      console.log('[chat] Active stream on another conversation, will update via SSE')
    }
  },

  // ─── Send Message ─────────────────────────────────────
  async sendMessage() {
    const text = this.inputMessage?.trim()
    if (!text || this.isLoading) return

    this.inputMessage = ''
    this.isLoading = true
    this.$store.chat.isLoading = true

    // Auto-create conversation if none selected
    let conversationId = this.$store.chat.currentConversationId
    if (!conversationId) {
      try {
        const agentId = this.selectedAgentId || null
        const conv = await api.post('/api/conversations', { title: 'New Chat', agent_id: agentId })
        conversationId = conv.conversation.id
        this.$store.chat.currentConversationId = conversationId
        this.$store.chat.currentConversationTitle = conv.conversation.title
        this.$store.chat.addConversation(conv.conversation)
        this.$store.chat.clearMessages()
      } catch (e) {
        this.isLoading = false
        this.$store.chat.isLoading = false
        this.$store.ui.showToast('Failed to create conversation', 'error')
        return
      }
    }

    // Add user message
    const userMsg = {
      id: helpers.generateId(),
      role: 'user',
      content: text,
      created_at: new Date().toISOString()
    }
    this.$store.chat.addMessage(userMsg)

    try {
      const data = await api.post(
        `/api/conversations/${conversationId}/messages`,
        { message: text, enable_rag: this.isRAGActive, document_ids: this.selectedDocumentIds.includes('all') ? null : this.selectedDocumentIds }
      )
      await this.streamResponse(data.request_id)
    } catch (e) {
      console.error('[chat] Send error:', e)
      this.isLoading = false
      this.$store.chat.isLoading = false
      this.$store.ui.showToast('Failed to send message', 'error')
    }
  },

  // ─── Stream Response ──────────────────────────────────
  async streamResponse(requestId) {
    const assistantMsg = {
      id: helpers.generateId(),
      role: 'assistant',
      content: '',
      blocks: [],
      created_at: new Date().toISOString()
    }
    this.$store.chat.addMessage(assistantMsg)
    const msgIndex = this.messages.length - 1

    this.$store.chat.startStreaming(
      requestId,
      this.$store.chat.currentConversationId,
      msgIndex,
      this.$store.chat.currentConversationTitle,
      [...this.$store.chat.messages]
    )

    const handlers = sseService.stream(requestId, this.$store.chat.currentConversationId, {
      enableRag: this.isRAGActive,
      documentIds: this.selectedDocumentIds.includes('all') ? null : this.selectedDocumentIds,
      model: this.selectedModel
    })

    handlers.onData((data) => this.processEvent(data, msgIndex))

    handlers.onError((error) => {
      console.error('[chat] Stream error:', error)
      this.$store.chat.stopStreaming()
      const msg = this.messages[msgIndex]
      if (msg) msg.content += `\n\n❌ Error: ${error.message}`
      this.$store.ui.showToast(`Stream error: ${error.message}`, 'error')
    })

    handlers.onComplete(() => {
      console.log('[chat] Stream complete')
      this.$store.chat.stopStreaming()
    })
  },

  // ─── Event Processing ─────────────────────────────────
  processEvent(data, msgIndex) {
    // Skip if on different conversation
    const active = this.$store.chat.activeStreaming
    if (active.isStreaming && active.conversationId !== this.$store.chat.currentConversationId) return

    const msg = this.messages[msgIndex]
    if (!msg) return

    // Ensure blocks array exists
    if (!msg.blocks) msg.blocks = []

    switch (data.type) {
      case 'content':
        this.appendBlock(msg, 'content', data.content)
        break

      case 'thinking':
        this.appendBlock(msg, 'thinking', data.content)
        break

      case 'tool_call_start':
        this.$store.chat.toolStatus = {
          active: true, tool: data.tool, status: 'Starting...', progress: null
        }
        msg.blocks.push({
          type: 'tool_call', name: data.tool, arguments: data.args || {},
          status: 'starting', progress: 0, result: null, sources: [], progress_history: []
        })
        break

      case 'tool_progress':
        this.$store.chat.toolStatus = {
          active: true, tool: data.tool || this.$store.chat.toolStatus.tool,
          status: data.status, progress: data.progress || null
        }
        const toolBlock = [...msg.blocks].reverse().find(b =>
          b.type === 'tool_call' && b.status !== 'completed' && b.status !== 'error'
        )
        if (toolBlock) {
          toolBlock.status = data.status
          toolBlock.progress = data.progress || 0
          if (data.data) {
            Object.assign(toolBlock, {
              search_steps: data.data.search_steps || [],
              search_terms: data.data.search_terms || [],
              reasoning: data.data.reasoning || null,
              coverage_score: data.data.coverage_score
            })
          }
          if (data.result) {
            toolBlock.result = data.result
            toolBlock.status = 'completed'
            if (data.result.sources?.length) toolBlock.sources = data.result.sources
          }
        }
        if (data.result) {
          this.$store.chat.toolStatus.active = false
        }
        break

      case 'error':
        this.$store.chat.toolStatus.active = false
        this.isLoading = false
        this.$store.chat.isLoading = false
        msg.blocks.push({ type: 'content', content: `\n\n❌ Error: ${data.error}` })
        this.$store.ui.showToast(`Error: ${data.error}`, 'error')
        break

      case 'title_update':
        this.$store.chat.currentConversationTitle = data.title
        const ci = this.$store.chat.conversations.findIndex(c => c.id === this.$store.chat.currentConversationId)
        if (ci !== -1) this.$store.chat.conversations[ci].title = data.title
        break

      case 'done':
        this.pendingTitleTimeout = setTimeout(() => sseService.close(), 5000)
        this.isLoading = false
        this.$store.chat.isLoading = false
        this.$store.chat.toolStatus.active = false
        break
    }

    // Force reactivity
    this.messages = [...this.$store.chat.messages]

    // Scroll to bottom
    //this.$nextTick(() => {
    //  const el = document.getElementById('messages-container')
    //  if (el) el.scrollTop = el.scrollHeight
    //})
  },

  appendBlock(msg, type, content) {
    const last = msg.blocks[msg.blocks.length - 1]
    if (last && last.type === type) {
      last.content += content
    } else {
      msg.blocks.push({ type, content })
    }
  },

  // ─── Message Actions ──────────────────────────────────
  async deleteMessage(id, e) {
    e?.stopPropagation()
    if (!confirm('Delete this message?')) return
    try {
      await api.delete(`/api/messages/${id}`)
      this.$store.chat.removeMessage(id)
    } catch (e) {
      this.$store.ui.showToast('Failed to delete', 'error')
    }
  },

  startEdit(id, content) {
    this.editingMessageId = id
    this.editContent = content
  },

  cancelEdit() {
    this.editingMessageId = null
    this.editContent = ''
  },

  async saveEdit(id) {
    if (!this.editContent.trim()) { this.cancelEdit(); return }
    const msg = this.messages.find(m => m.id === this.editingMessageId)
    if (!msg) { this.cancelEdit(); return }

    if (msg.role === 'user') {
      // Fork conversation for user message edits
      if (this.editContent.trim() !== msg.content.trim()) {
        await this.forkConversation(this.editingMessageId, this.editContent.trim())
      }
      this.cancelEdit()
      return
    }

    try {
      const data = await api.put(`/api/messages/${this.editingMessageId}`, {
        content: this.editContent
      })
      msg.content = data.message.content
    } catch (e) {
      this.$store.ui.showToast('Failed to update', 'error')
    }
    this.cancelEdit()
  },

  async forkConversation(originalId, newContent) {
    try {
      const currentConv = this.$store.chat.conversations.find(c => c.id === this.$store.chat.currentConversationId)
      const agentId = currentConv?.agent_id || this.selectedAgentId || null
      const data = await api.post('/api/conversations', {
        title: 'Forked: ' + newContent.substring(0, 30) + '...',
        agent_id: agentId
      })
      const newId = data.conversation.id
      this.$store.chat.addConversation(data.conversation)
      this.$store.chat.currentConversationId = newId
      this.$store.chat.currentConversationTitle = data.conversation.title
      this.$store.chat.clearMessages()
      this.$store.chat.addMessage({
        id: helpers.generateId(), role: 'user', content: newContent,
        created_at: new Date().toISOString()
      })
      this.messages = this.$store.chat.messages
      const streamData = await api.post(`/api/conversations/${newId}/messages`, { message: newContent })
      await this.streamResponse(streamData.request_id)
    } catch (e) {
      this.$store.ui.showToast('Failed to fork', 'error')
    }
    this.cancelEdit()
  },

  async regenerate(id) {
    if (this.isLoading) return
    this.isLoading = true
    this.$store.chat.isLoading = true

    try {
      const data = await api.post(
        `/api/conversations/${this.$store.chat.currentConversationId}/regenerate`,
        { message_id: id }
      )

      // Remove assistant message being regenerated
      const idx = this.messages.findIndex(m => m.id === id)
      if (idx !== -1) {
        this.$store.chat.messages = this.$store.chat.messages.slice(0, idx + 1)
        this.messages = [...this.$store.chat.messages]
      }

      const newMsg = {
        id: helpers.generateId(), role: 'assistant', content: '',
        blocks: [], created_at: new Date().toISOString()
      }
      this.$store.chat.addMessage(newMsg)
      const newIdx = this.messages.length - 1

      const handlers = sseService.stream(data.request_id, this.$store.chat.currentConversationId, {
        model: this.selectedModel
      })
      handlers.onData((d) => this.processEvent(d, newIdx))
    } catch (e) {
      console.error('[chat] Regenerate:', e)
      this.isLoading = false
      this.$store.chat.isLoading = false
      this.$store.ui.showToast('Failed to regenerate', 'error')
    }
  },

  cancelRequest() {
    sseService.close()
    this.isLoading = false
    this.$store.chat.isLoading = false
    this.$store.chat.toolStatus.active = false
    const last = this.messages[this.messages.length - 1]
    if (last?.role === 'assistant' && !last.content?.trim()) {
      last.content = '⚠️ Request cancelled.'
    }
    this.$store.ui.showToast('Request cancelled', 'info')
  },

  // ─── Take Note ────────────────────────────────────────
  async takeNote(msg) {
    const text = msg.content?.trim()
    if (!text) return

    // Get highlighted text or use first 200 chars as source
    const selected = window.getSelection()?.toString()?.trim()
    const sourceText = selected || text.substring(0, 500)

    // Prompt for note content
    const noteContent = prompt('Enter your note:', selected ? `» ${selected}` : '')
    if (!noteContent?.trim()) return

    try {
      const res = await api.post('/api/notes', {
        conversation_id: this.$store.chat.currentConversationId,
        message_id: msg.id,
        content: noteContent.trim(),
        source_text: sourceText.substring(0, 1000)
      })
      if (res.note) {
        // Add to UI store notes
        if (!this.$store.ui.notes) this.$store.ui.notes = []
        this.$store.ui.notes.unshift(res.note)
        this.$store.ui.showToast('Note saved!', 'success')
      }
    } catch (e) {
      console.error('[chat] Note error:', e)
      this.$store.ui.showToast('Failed to save note', 'error')
    }
  },

  // ─── TTS ──────────────────────────────────────────────
  async speakMessage(msg) {
    const loading = this.$store.ui
    loading.toast = { show: true, message: 'Generating speech...', type: 'info' }
    setTimeout(() => { loading.toast = { show: false, message: '', type: 'success' } }, 1500)

    const success = await ttsService.speak(msg,
      (err) => this.$store.ui.showToast(err, 'error')
    )
    if (success) {
      this.$store.ui.currentAudio = ttsService.currentAudio
      this.$store.ui.currentAudioMessageId = ttsService.currentAudioMessageId
      this.$store.ui.isPlaying = ttsService.isPlaying
    }
  },

  stopAudio() {
    ttsService.stop()
    this.$store.ui.currentAudio = null
    this.$store.ui.currentAudioMessageId = null
    this.$store.ui.isPlaying = false
  },

  // ─── Toggle Helpers ───────────────────────────────────
  toggleBlock(key) {
    this.expandedBlocks[key] = !this.expandedBlocks[key]
  },

  isExpanded(key) {
    return !!this.expandedBlocks[key]
  },

  // ─── Formatters ───────────────────────────────────────
  renderMarkdown: (t) => markdownUtils.render(t),
  renderWithCitations: (t, s) => markdownUtils.renderWithCitations(t, s),
  extractSources: (blocks) => markdownUtils.extractSources(blocks),
  formatToolResult: (r) => helpers.formatToolResult(r),
  parseEscapes: (t) => helpers.parseEscapes(t),
  formatDate: (s) => formatters.formatDate(s),

  async copyMessage(msg) {
    const success = await helpers.copyToClipboard(msg.content)
    this.$store.ui.showToast(success ? 'Copied!' : 'Copy failed', success ? 'success' : 'error')
  },

  // ─── UI Handlers ──────────────────────────────────────
  updateSelectedModel() {
    this.$store.chat.setModel(this.selectedModel)
  },

  async updateSelectedAgent() {
    this.$store.chat.setAgent(this.selectedAgentId)
    this.$store.chat.applyAgentConfig()
    this.selectedModel = this.$store.chat.selectedModel
    this.selectedDocumentIds = [...(this.$store.chat.selectedDocumentIds || [])]

    // Update the current conversation's agent_id if one is active
    const convId = this.$store.chat.currentConversationId
    if (convId) {
      try {
        const res = await api.put(`/api/conversations/${convId}/agent`, {
          agent_id: this.selectedAgentId || null
        })
        // Update the conversation's agent_id in the store for filter to work
        const conv = this.$store.chat.conversations.find(c => c.id === convId)
        if (conv) {
          conv.agent_id = res.agent_id
        }
      } catch (e) {
        console.error('[chat] Failed to update conversation agent:', e)
      }
    }
  },

  toggleDocument(docId) {
    if (docId === 'all') {
      if (this.selectedDocumentIds.includes('all')) {
        this.selectedDocumentIds = []
      } else {
        this.selectedDocumentIds = ['all']
      }
    } else {
      const idx = this.selectedDocumentIds.indexOf(docId)
      if (idx >= 0) {
        this.selectedDocumentIds.splice(idx, 1)
      } else {
        this.selectedDocumentIds.push(docId)
      }
      const allIdx = this.selectedDocumentIds.indexOf('all')
      if (allIdx >= 0 && this.selectedDocumentIds.length > 1) {
        this.selectedDocumentIds.splice(allIdx, 1)
      }
    }
    this.$store.chat.selectedDocumentIds = [...this.selectedDocumentIds]
  },

  isDocumentSelected(docId) {
    if (docId === 'all') return this.selectedDocumentIds.includes('all')
    return this.selectedDocumentIds.includes('all') || this.selectedDocumentIds.includes(docId)
  },

  updateTitle(e) {
    const title = e.target.value.trim()
    if (!title) return
    api.put(`/api/conversations/${this.$store.chat.currentConversationId}`, { title })
      .then(() => {
        this.$store.chat.currentConversationTitle = title
        this.currentConversationTitle = title
        this.$store.chat.updateConversation(this.$store.chat.currentConversationId, { title })
      })
      .catch(() => this.$store.ui.showToast('Failed to update title', 'error'))
  }
})

// Export factory for registration in main.js
export { chatComponent }

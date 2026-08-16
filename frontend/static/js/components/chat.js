/**
 * Chat Component — Messages, streaming, tools, TTS
 */
import { sseService } from '../services/sse.js'
import { ttsService } from '../services/tts.js?v=45'
import { sttService } from '../services/stt.js'
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
  availableProviders: [],
  selectedProviderId: '',
  availableAgents: [],
  documents: [],

  // MCP server management (composer dropdown)
  mcpServers: [],
  mcpView: 'list', // 'list' | 'form'
  mcpForm: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', headers: '{}', timeout: 60, originalName: null },
  currentConversationTitle: 'New Chat',

  // Input focus state for expanded input sizing
  isFocused: false,

  // Attached files before sending
  attachedFiles: [],
  isUploading: false,

  // Drag-and-drop state
  dragOver: false,
  dragCounter: 0,

  // Expanded state for collapsible blocks
  expandedBlocks: {},

  // Recording state
  isRecording: false,
  sttSupported: false,

  // Source content modal state
  selectedSource: null,
  showSourceModal: false,

  // Tracks whether stream completed normally (received 'done' event)
  // If false on connection close, response may be incomplete (MTP stall / timeout)
  streamEndedNormally: false,

  // Context transparency panel (Phase 6) — what the model saw this turn
  contextInfo: null,
  contextOpen: false,

  // ─── Composer getters ────────────────────────────────
  get selectedModelName() {
    const m = this.availableModels.find(x => x.id === this.selectedModel)
    const label = m ? (m.name || m.id) : 'Model'
    if (m?.provider_name) return `${label}`
    return label
  },

  modelsByProvider(providerId) {
    return this.availableModels.filter(m => (m.provider_id || '') === providerId)
  },

  get selectedAgentName() {
    const a = this.availableAgents.find(x => x.id === this.selectedAgentId)
    return a ? a.name : 'Agent'
  },

  get mcpSummary() {
    const n = this.mcpServers.length
    if (!n) return 'MCP'
    const on = this.mcpServers.filter(s => s.enabled).length
    return on + '/' + n
  },

  // ─── Getters (reactive to store) ──────────────────────
  get messages() {
    // Deduplicate by version_group — keep only the latest version of each group.
    // This ensures consistent visual layout during live chat and after refresh.
    const raw = this.$store.chat.messages || []
    const seen = {}
    const result = []
    for (const msg of raw) {
      const vg = msg.version_group
      if (!vg) {
        result.push(msg)
      } else if (vg in seen) {
        const idx = seen[vg]
        if (msg.version > result[idx].version) {
          result[idx] = msg
        }
      } else {
        seen[vg] = result.length
        result.push(msg)
      }
    }
    return result
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
    await Promise.all([this.loadModels(), this.loadAgents(), this.loadDocuments(), this.loadMCPServers()])
    this.$store.chat.loadSavedModel()
    this.selectedModel = this.$store.chat.selectedModel
    this.selectedProviderId = this.$store.chat.selectedProviderId
    this.$store.chat.loadSavedAgent()
    this.selectedAgentId = this.$store.chat.selectedAgentId
    this.selectedDocumentIds = [...(this.$store.chat.selectedDocumentIds || [])]
    await ttsService.checkAvailability()

    // Single writer: TTS icon state mirrors the real audio element, never drifts
    ttsService.onStateChange = ({ playing, paused, msgId }) => {
      this.$store.ui.isPlaying = playing
      this.$store.ui.isPaused = paused
      this.$store.ui.currentAudioMessageId = msgId
      this.$store.ui.currentAudio = playing ? ttsService.player : null
    }

    this.sttSupported = sttService.isSupported()
    this.checkActiveStream()

    window.addEventListener('sync-agent', (e) => {
      this.selectedAgentId = e.detail.agentId
    })

    // Focus the input only when the loaded conversation is new/empty (no history
    // to read); when browsing an existing conversation, blur it so it collapses
    // to its small height — the user clicks in to resume typing.
    window.addEventListener('conversation-loaded', () => {
      this.$nextTick(() => {
        if (this.$store.chat.messages.length === 0) {
          this.$refs?.chatInput?.focus()
        } else {
          this.$refs?.chatInput?.blur()
        }
      })
    })

    // Auto-focus the chat input on init only for a fresh chat (no history)
    this.$nextTick(() => {
      if (this.$store.chat.messages.length === 0) {
        this.$refs?.chatInput?.focus()
      }
    })
  },

  syncFromStore() {
    this.isLoading = this.$store.chat.isLoading
    this.toolStatus = { ...this.$store.chat.toolStatus }
    this.selectedModel = this.$store.chat.selectedModel
    this.selectedProviderId = this.$store.chat.selectedProviderId
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
      this.availableProviders = data.providers || []
      this.$store.chat.availableModels = this.availableModels
      this.$store.chat.availableProviders = this.availableProviders
      // Restore provider selection or default to the default provider
      if (!this.selectedProviderId && this.availableProviders.length) {
        const def = this.availableProviders.find(p => p.is_default) || this.availableProviders[0]
        this.selectedProviderId = def.id
      }
      if (!this.selectedModel && this.availableModels.length) {
        const m = this.availableModels.find(x => x.provider_id === this.selectedProviderId) || this.availableModels[0]
        if (m) {
          this.selectedModel = m.id
          this.selectedProviderId = m.provider_id
        }
      }
      this.$store.chat.selectedProviderId = this.selectedProviderId
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

  // ─── MCP Servers (composer dropdown) ──────────────────
  async loadMCPServers() {
    try {
      const [serversData, toolsData] = await Promise.all([
        api.get('/api/mcp/servers'),
        api.get('/api/mcp/tools')
      ])
      this.mcpServers = (serversData.servers || []).map(s => ({
        ...s,
        tools: (toolsData.tools || []).filter(t => t.server === s.name),
        disabled_tools: s.disabled_tools || [],
        toolsExpanded: false,
        enabled: s.enabled !== false,
        error: s.error || null
      }))
    } catch (e) { console.error('[chat] MCP servers:', e) }
  },

  async toggleServerEnabled(server, enabled) {
    try {
      await api.post(`/api/mcp/servers/${encodeURIComponent(server.name)}/toggle`, { enabled })
      server.enabled = enabled
    } catch (e) { this.$store.ui.showToast('Toggle failed: ' + e.message, 'error') }
  },

  async toggleToolEnabled(server, toolName, disabled) {
    try {
      await api.put(`/api/mcp/servers/${encodeURIComponent(server.name)}/tools/toggle`, { tool_name: toolName, disabled })
      const idx = server.disabled_tools.indexOf(toolName)
      if (disabled && idx === -1) server.disabled_tools.push(toolName)
      if (!disabled && idx !== -1) server.disabled_tools.splice(idx, 1)
    } catch (e) { this.$store.ui.showToast('Tool toggle failed', 'error') }
  },

  async reconnectServer(name) {
    try {
      await api.post(`/api/mcp/servers/${encodeURIComponent(name)}/reconnect`)
      await this.loadMCPServers()
      this.$store.ui.showToast('Reconnected', 'success')
    } catch (e) { this.$store.ui.showToast('Reconnect failed: ' + e.message, 'error') }
  },

  async refreshServerTools(name) {
    try {
      await api.post(`/api/mcp/servers/${encodeURIComponent(name)}/refresh`)
      await this.loadMCPServers()
      this.$store.ui.showToast('Tools refreshed', 'success')
    } catch (e) { this.$store.ui.showToast('Refresh failed: ' + e.message, 'error') }
  },

  async removeServer(name) {
    if (!confirm(`Remove MCP server "${name}"?`)) return
    try {
      await api.delete(`/api/mcp/servers/${encodeURIComponent(name)}`)
      await this.loadMCPServers()
      this.$store.ui.showToast('Server removed', 'success')
    } catch (e) { this.$store.ui.showToast('Remove failed: ' + e.message, 'error') }
  },

  openAddServer() {
    this.mcpForm = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', headers: '{}', timeout: 60, originalName: null }
    this.mcpView = 'form'
  },

  openEditServer(server) {
    this.mcpForm = {
      name: server.name, transport_type: server.transport_type,
      command: server.command || '', args: JSON.stringify(server.args || []),
      url: server.url || '', env: server.env ? JSON.stringify(server.env) : '{}',
      headers: server.headers ? JSON.stringify(server.headers) : '{}',
      timeout: server.timeout || 60, originalName: server.name
    }
    this.mcpView = 'form'
  },

  cancelMCPForm() { this.mcpView = 'list' },

  async saveMCPServer() {
    const f = this.mcpForm
    const name = (f.name || '').trim()
    if (!name) { this.$store.ui.showToast('Name is required', 'error'); return }
    if (f.transport_type !== 'stdio' && !f.url) { this.$store.ui.showToast('URL required for SSE/HTTP', 'error'); return }
    if (f.transport_type === 'stdio' && !f.command) { this.$store.ui.showToast('Command required for stdio', 'error'); return }

    let args = [], env = {}, headers = {}
    try { args = JSON.parse(f.args || '[]'); if (!Array.isArray(args)) args = [] }
    catch (e) { this.$store.ui.showToast('Invalid JSON in Args', 'error'); return }
    try { env = JSON.parse(f.env || '{}'); if (typeof env !== 'object' || Array.isArray(env)) env = {} }
    catch (e) { this.$store.ui.showToast('Invalid JSON in Env', 'error'); return }
    try { headers = JSON.parse(f.headers || '{}'); if (typeof headers !== 'object' || Array.isArray(headers)) headers = {} }
    catch (e) { this.$store.ui.showToast('Invalid JSON in Headers', 'error'); return }

    const payload = { name, transport_type: f.transport_type, command: f.command, args, env, headers, url: f.url || null, timeout: f.timeout || 60 }
    try {
      const res = f.originalName
        ? await api.put(`/api/mcp/servers/${encodeURIComponent(f.originalName)}`, payload)
        : await api.post('/api/mcp/servers', payload)
      await this.loadMCPServers()
      this.mcpView = 'list'
      if (res && res.connected === false) {
        this.$store.ui.showToast(res.error || res.message || 'Saved but connection failed', 'warning')
      } else {
        this.$store.ui.showToast(f.originalName ? 'Server updated!' : 'Server added!', 'success')
      }
    } catch (e) { this.$store.ui.showToast('Error: ' + e.message, 'error') }
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

  // ─── File Upload ──────────────────────────────────────
  triggerFileUpload() {
    this.$refs.fileInput?.click()
  },

  handleFileSelect(e) {
    this.addFiles(e.target.files || [])
    e.target.value = ''
  },

  addFiles(fileList) {
    const files = Array.from(fileList || [])
    if (!files.length) return
    const remaining = 10 - this.attachedFiles.length
    if (files.length > remaining) {
      this.$store.ui.showToast('Maximum 10 files per message', 'warning')
    }
    files.slice(0, Math.max(0, remaining)).forEach(f => {
      this.attachedFiles.push({
        file: f,
        preview: f.type?.startsWith('image/') ? URL.createObjectURL(f) : null,
        filename: f.name,
        type: f.type,
        size: f.size,
        uploading: false,
        uploaded: false,
        url: null,
        error: null
      })
    })
  },

  handleDragEnter() {
    this.dragCounter++
    this.dragOver = true
  },

  handleDragLeave() {
    this.dragCounter = Math.max(0, this.dragCounter - 1)
    if (this.dragCounter === 0) this.dragOver = false
  },

  handleDrop(e) {
    this.dragCounter = 0
    this.dragOver = false
    if (this.isLoading || this.isUploading) {
      this.$store.ui.showToast('Cannot attach files while a request is running', 'warning')
      return
    }
    this.addFiles(e.dataTransfer?.files)
  },

  removeFile(idx) {
    const item = this.attachedFiles[idx]
    if (item?.preview) URL.revokeObjectURL(item.preview)
    this.attachedFiles.splice(idx, 1)
  },

  async uploadAttachedFiles() {
    const toUpload = this.attachedFiles.filter(f => !f.uploaded && !f.uploading)
    if (!toUpload.length) return []
    this.isUploading = true
    const results = []
    for (const item of toUpload) {
      item.uploading = true
      try {
        const formData = new FormData()
        formData.append('file', item.file)
        const res = await fetch('/api/upload/chat-file', { method: 'POST', body: formData })
        if (!res.ok) throw new Error('Upload failed')
        const data = await res.json()
        item.uploaded = true
        item.uploading = false
        item.url = data.url
        results.push({ url: data.url, filename: data.filename, type: data.type, size: data.size })
        if (item.preview) URL.revokeObjectURL(item.preview)
      } catch (e) {
        item.error = e.message
        item.uploading = false
        this.$store.ui.showToast(`Upload failed: ${item.filename}`, 'error')
      }
    }
    this.isUploading = false
    return results
  },

  isImageType(type) {
    return type?.startsWith('image/')
  },

  formatFileSize(bytes) {
    return formatters.formatFileSize(bytes)
  },

  getFileIcon(type) {
    if (type?.startsWith('image/')) return 'ph-file-image'
    if (type?.includes('pdf')) return 'ph-file-pdf'
    if (type?.includes('spreadsheet') || type?.includes('excel')) return 'ph-file-xls'
    if (type?.includes('word') || type?.includes('document')) return 'ph-file-doc'
    if (type?.includes('text') || type?.includes('json') || type?.includes('markdown')) return 'ph-file-text'
    if (type?.includes('script') || type?.includes('python') || type?.includes('javascript')) return 'ph-file-code'
    if (type?.includes('zip') || type?.includes('tar') || type?.includes('gz')) return 'ph-file-archive'
    return 'ph-file'
  },

  // ─── Send Message ─────────────────────────────────────
  async sendMessage() {
    const text = this.inputMessage?.trim()
    const hasFiles = this.attachedFiles.length > 0
    if (!text && !hasFiles) return
    if (this.isLoading || this.isUploading) return

    this.inputMessage = ''
    this.isLoading = true
    this.$store.chat.isLoading = true

    // Upload attached files first
    let uploadedFiles = []
    if (hasFiles) {
      uploadedFiles = await this.uploadAttachedFiles()
    }

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

    // Build display content with file references embedded as markdown
    // (belt-and-suspenders: files are also stored separately for icon rendering)
    let displayContent = text
    if (uploadedFiles.length > 0) {
      const fileRefs = uploadedFiles.map(f => {
        const fname = f.filename || 'file'
        if (f.type?.startsWith('image/')) {
          return `![${fname}](${f.url})`
        }
        return `📎 [${fname}](${f.url})`
      }).join('  \n')
      displayContent = fileRefs + (text ? '\n\n' + text : '')
    }

    // Add user message with file metadata
    const userMsg = {
      id: helpers.generateId(),
      role: 'user',
      content: displayContent,
      files: uploadedFiles,
      created_at: new Date().toISOString()
    }
    this.$store.chat.addMessage(userMsg)
    this.attachedFiles = []  // Clear attachments

    try {
      const data = await api.post(
        `/api/conversations/${conversationId}/messages`,
        { 
          message: text, 
          enable_rag: this.isRAGActive, 
          document_ids: this.selectedDocumentIds.includes('all') ? null : this.selectedDocumentIds,
          files: uploadedFiles,
          model: this.selectedModel,
          provider_id: this.selectedProviderId
        }
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
      model: this.selectedModel,
      providerId: this.selectedProviderId
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
      if (!this.streamEndedNormally) {
        console.warn('[chat] Stream ended without done event — response may be incomplete')
        this.$store.ui.showToast('Response may be incomplete (stream interrupted)', 'warning')
      }
      this.streamEndedNormally = false
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

      case 'context_info':
        this.contextInfo = data.context || null
        this.$store.chat.contextInfo = this.contextInfo
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

      case 'tool_approval_required':
        this.$store.chat.toolStatus = {
          active: true, tool: data.tool || 'run_command', status: 'Awaiting approval', progress: null
        }
        const approvalBlock = [...msg.blocks].reverse().find(b =>
          b.type === 'tool_call' && b.status !== 'completed' && b.status !== 'error'
        )
        if (approvalBlock) {
          approvalBlock.status = 'approval'
          approvalBlock.command = data.command || ''
          approvalBlock.working_dir = data.working_dir || ''
          approvalBlock.approval_reason = data.reason || ''
          approvalBlock.approval_key = data.approval_key || null
        }
        break

      case 'tool_error':
        this.$store.chat.toolStatus = {
          active: false, tool: data.tool, status: 'Error', progress: null
        }
        msg.blocks.push({
          type: 'tool_call', name: data.tool, arguments: {},
          status: 'error', progress: 0, result: {error: data.error},
          sources: [], progress_history: []
        })
        this.$store.ui.showToast(`Tool call incomplete: ${data.tool}`, 'warning')
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
        this.streamEndedNormally = true
        this.pendingTitleTimeout = setTimeout(() => sseService.close(), 5000)
        this.isLoading = false
        this.$store.chat.isLoading = false
        this.$store.chat.toolStatus.active = false
        // Re-focus input after streaming completes
        this.$nextTick(() => {
          this.$refs?.chatInput?.focus()
        })
        // Auto-read aloud if enabled
        if (this.$store.ui.settingsData?.tts_auto_read) {
          const lastMsg = this.messages[this.messages.length - 1]
          if (lastMsg?.role === 'assistant' && lastMsg.content?.trim()) {
            const plainText = formatters.stripMarkdown(lastMsg.content)
            const plainMsg = { ...lastMsg, content: plainText }
            const slot = document.getElementById('tts-slot-' + lastMsg.id)
            ttsService.speak(plainMsg, slot, null, () => {
              this.$store.ui.currentAudio = null
              this.$store.ui.currentAudioMessageId = null
              this.$store.ui.isPlaying = false
              this.$store.ui.isPaused = false
            })
          }
        }
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

  // ─── Tool Approval ─────────────────────────────────────
  async respondApproval(block, approved) {
    const active = this.$store.chat.activeStreaming
    if (!active?.requestId) return
    try {
      await api.post(`/api/tools/${active.requestId}/approve`, {
        decision: approved,
        approval_key: block.approval_key || null
      })
      block.status = approved ? 'approved' : 'denied'
    } catch (e) {
      this.$store.ui.showToast('Failed to send decision', 'error')
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

    // Store the old message before removing it (for potential version switching)
    const oldMsg = this.messages.find(m => m.id === id)

    try {
      const data = await api.post(
        `/api/conversations/${this.$store.chat.currentConversationId}/regenerate`,
        { message_id: id }
      )

      // Remove the old assistant message from display (it stays in DB as a prior version)
      const idx = this.messages.findIndex(m => m.id === id)
      if (idx !== -1) {
        this.$store.chat.messages = this.$store.chat.messages.slice(0, idx)
        this.messages = [...this.$store.chat.messages]
      }

      const newMsg = {
        id: helpers.generateId(), role: 'assistant', content: '',
        blocks: [], created_at: new Date().toISOString(),
        version: data.version || 1,
        version_group: data.version_group || null,
        max_version: data.version || 1  // Total version count (all pills shown up to this)
      }
      this.$store.chat.addMessage(newMsg)
      const newIdx = this.messages.length - 1

      // Pass version info to SSE stream for backend to save correctly
      const handlers = sseService.stream(data.request_id, this.$store.chat.currentConversationId, {
        model: this.selectedModel,
        isRegenerate: true,
        version: data.version,
        versionGroup: data.version_group,
        providerId: this.selectedProviderId
      })
      handlers.onData((d) => this.processEvent(d, newIdx))
    } catch (e) {
      console.error('[chat] Regenerate:', e)
      this.isLoading = false
      this.$store.chat.isLoading = false
      this.$store.ui.showToast('Failed to regenerate', 'error')
    }
  },

  async switchVersion(messageId, version) {
    // Find the message in the current list
    const msg = this.messages.find(m => m.id === messageId)
    if (!msg || !msg.version_group) return

    try {
      // Use version_group to fetch all versions (avoids frontend/backend ID mismatch)
      const data = await api.get(`/api/versions/${encodeURIComponent(msg.version_group)}`)
      const versions = data.versions || []
      const targetVersion = versions.find(v => v.version === version)
      if (!targetVersion) return

      // Determine the highest version number (for max_version)
      const maxVer = versions.reduce((max, v) => Math.max(max, v.version || 1), 1)

      // Swap the message content with the target version
      msg.content = targetVersion.content
      msg.version = targetVersion.version
      msg.max_version = maxVer
      msg.blocks = targetVersion.blocks || []
      msg.tool_calls = targetVersion.tool_calls
      msg.thinking = targetVersion.thinking

      // Force reactivity
      this.messages = [...this.$store.chat.messages]
    } catch (e) {
      console.error('[chat] Switch version:', e)
      this.$store.ui.showToast('Failed to switch version', 'error')
    }
  },

  // ─── Voice Recording ──────────────────────────────────
  async toggleRecording() {
    if (this.isRecording) {
      // Stop recording and transcribe
      try {
        const text = await sttService.stopRecording()
        this.isRecording = false
        if (text) {
          this.inputMessage = text
          // Auto-focus input after transcription
          this.$nextTick(() => this.$refs?.chatInput?.focus())
        }
      } catch (e) {
        this.isRecording = false
        this.$store.ui.showToast(e.message || 'Recording failed', 'error')
      }
    } else {
      // Start recording
      try {
        await sttService.startRecording()
        this.isRecording = true
      } catch (e) {
        this.isRecording = false
        this.$store.ui.showToast(e.message || 'Could not start recording', 'error')
      }
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
    // Re-focus input after cancellation
    this.$nextTick(() => {
      this.$refs?.chatInput?.focus()
    })
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
  handleTtsClick(msg) {
    const ui = this.$store.ui
    if (ui.currentAudioMessageId === msg.id) {
      if (ui.isPlaying) this.pauseAudio()
      else if (ui.isPaused) this.resumeAudio()
      else this.speakMessage(msg)
    } else {
      this.speakMessage(msg)
    }
  },

  pauseAudio() {
    ttsService.pause()
  },

  resumeAudio() {
    ttsService.resume()
  },

  async speakMessage(msg) {
    const loading = this.$store.ui
    // Set playing state immediately BEFORE the async request so the pause icon shows right away
    this.$store.ui.currentAudioMessageId = msg.id
    this.$store.ui.isPlaying = true
    this.$store.ui.isPaused = false

    // Send plain text (no markdown) to TTS so it doesn't read out asterisks, hyphens, etc.
    const plainText = formatters.stripMarkdown(msg.content || '')
    const plainMsg = { ...msg, content: plainText }

    const slot = document.getElementById('tts-slot-' + msg.id)
    const success = await ttsService.speak(plainMsg, slot,
      (err) => this.$store.ui.showToast(err, 'error'),
      () => {
        // Audio finished or failed — reset UI state
        this.$store.ui.currentAudio = null
        this.$store.ui.currentAudioMessageId = null
        this.$store.ui.isPlaying = false
        this.$store.ui.isPaused = false
      }
    )
    if (success) {
      this.$store.ui.currentAudio = ttsService.player
    } else {
      // TTS request failed — revert the optimistic UI state
      this.$store.ui.currentAudio = null
      this.$store.ui.currentAudioMessageId = null
      this.$store.ui.isPlaying = false
      this.$store.ui.isPaused = false
    }
  },

  stopAudio() {
    ttsService.stop()
    this.$store.ui.currentAudio = null
    this.$store.ui.currentAudioMessageId = null
    this.$store.ui.isPlaying = false
    this.$store.ui.isPaused = false
  },

  // ─── Toggle Helpers ───────────────────────────────────
  toggleBlock(key) {
    this.expandedBlocks[key] = !this.expandedBlocks[key]
  },

  isExpanded(key) {
    return !!this.expandedBlocks[key]
  },

  handleSourceClick(source) {
    // For chunk sources (from query_documents), show in modal
    if (source.type === 'chunk') {
      this.selectedSource = source
      this.showSourceModal = true
      return
    }
    
    // For URL sources (from web search tools) or sources without type, open in new tab
    // This maintains backward compatibility with external tools that don't set type
    if (source.url) {
      window.open(source.url, '_blank', 'noopener')
    }
  },

  closeSourceModal() {
    this.showSourceModal = false
    this.selectedSource = null
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
    this.$store.chat.setModel(this.selectedModel, this.selectedProviderId)
  },

  selectModel(id, providerId) {
    const model = this.availableModels.find(m => m.id === id)
    this.selectedModel = id
    this.selectedProviderId = providerId || model?.provider_id || this.selectedProviderId
    this.updateSelectedModel()
  },

  selectAgent(id) {
    this.selectedAgentId = id || null
    this.updateSelectedAgent()
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

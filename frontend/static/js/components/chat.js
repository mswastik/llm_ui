/**
 * Chat Component — Messages, streaming, tools, TTS
 */
import { sseService } from '../services/sse.js?v=70'
import { ttsService } from '../services/tts.js?v=45'
import { sttService } from '../services/stt.js'
import { offline } from '../services/offline.js'
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
  thinkingMode: 'auto',
  // MCP server management (composer dropdown)
  mcpServers: [],
  // Servers the user manually enabled for this session even though the
  // selected agent doesn't allow them (one-off override; sent per request).
  sessionMcpOverrides: [],
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

  // ─── Pull-to-refresh (mobile) ──────────────────────────
  // Tracked in component state, not the store: only one chat instance cares.
  pullRefreshVisible: false,
  pullRefreshStatus: 'idle', // 'idle' | 'pulling' | 'refreshing'
  _ptrStartY: 0,
  _ptrDistance: 0,
  _ptrTriggered: false,

  // ─── Pull-to-refresh handlers ──────────────────────────
  // Threshold: 70px of vertical travel at the top of the messages container
  // fires a refresh. Anything shorter just lets the native scroll resume.
  _ptrThreshold: 70,
  onPTRStart(e) {
    if (window.matchMedia('(max-width: 768px)').matches === false) return
    const el = document.getElementById('messages-container')
    if (!el || el.scrollTop > 4) { this._ptrTriggered = false; return }
    this._ptrStartY = e.touches[0].clientY
    this._ptrDistance = 0
    this._ptrTriggered = true
  },
  onPTRMove(e) {
    if (!this._ptrTriggered) return
    const dy = e.touches[0].clientY - this._ptrStartY
    if (dy <= 0) { this.pullRefreshVisible = false; return }
    this._ptrDistance = dy
    this.pullRefreshVisible = dy > 24
    this.pullRefreshStatus = 'pulling'
  },
  async onPTREnd() {
    if (!this._ptrTriggered) return
    this._ptrTriggered = false
    if (this._ptrDistance >= this._ptrThreshold) {
      this.pullRefreshStatus = 'refreshing'
      try {
        const cid = this.$store.chat.currentConversationId
        if (cid) {
          const data = await api.get(`/api/conversations/${cid}`)
          this.$store.chat.messages = (data.messages || []).map((m) => ({ ...m, blocks: m.blocks || m.metadata?.blocks || null }))
          if (offline.isSupported()) {
            offline.saveConversation(data.conversation, data.messages || []).catch(() => {})
          }
        } else {
          // No active conversation: just refresh the sidebar list.
          window.dispatchEvent(new CustomEvent('refresh-conversations'))
        }
        this.$store.ui.showToast('Refreshed', 'success')
      } catch (e) {
        this.$store.ui.showToast('Refresh failed', 'error')
      } finally {
        setTimeout(() => {
          this.pullRefreshVisible = false
          this.pullRefreshStatus = 'idle'
          this._ptrDistance = 0
        }, 400)
      }
    } else {
      this.pullRefreshVisible = false
      this.pullRefreshStatus = 'idle'
      this._ptrDistance = 0
    }
  },

  get selectedModelName() {
    const m = this.availableModels.find(x => x.id === this.selectedModel)
    if (m) return (m.name || m.id)
    // Agent-configured model may not be in the cached model list; show it anyway.
    return this.selectedModel || 'Model'
  },

  modelsByProvider(providerId) {
    return this.availableModels.filter(m => (m.provider_id || '') === providerId)
  },

  get selectedAgentName() {
    const a = this.availableAgents.find(x => x.id === this.selectedAgentId)
    return a ? a.name : 'Agent'
  },

  // ─── Thinking modes ───────────────────────────────
  thinkingModes: [
    { id: 'auto', label: 'Auto', icon: 'ph-brain', desc: 'Model default' },
    { id: 'off', label: 'Off', icon: 'ph-prohibit', desc: 'No reasoning' },
    { id: 'on', label: 'On', icon: 'ph-brain', desc: 'Thinking on' },
    { id: 'low', label: 'Low', icon: 'ph-lightning', desc: 'Quick reasoning' },
    { id: 'medium', label: 'Med', icon: 'ph-lightning', desc: 'Balanced' },
    { id: 'high', label: 'High', icon: 'ph-lightning', desc: 'Deep reasoning' },
  ],
  get thinkingModeLabel() {
    const m = this.thinkingModes.find(x => x.id === this.thinkingMode)
    return m ? m.label : 'Auto'
  },
  get thinkingModeIcon() {
    const m = this.thinkingModes.find(x => x.id === this.thinkingMode)
    return m ? m.icon : 'ph-brain'
  },
  selectThinkingMode(mode) {
    this.thinkingMode = mode || 'auto'
    this.$store.chat.thinkingMode = this.thinkingMode
    this.$store.chat.setThinkingMode(this.thinkingMode)
  },

  get mcpSummary() {
    const n = this.mcpServers.length
    if (!n) return 'MCP'
    if (this.agentMcpRestricted) {
      const allowed = this.agentAllowedServerNames
      const on = this.mcpServers.filter(s => allowed.includes(s.name) || this.sessionMcpOverrides.includes(s.name)).length
      return on + '/' + n
    }
    const on = this.mcpServers.filter(s => s.enabled).length
    return on + '/' + n
  },

  // Selected agent's enabled_mcp_servers, or null when nothing is restricted
  // (empty list = allow everything, matching the backend tool filter).
  get agentAllowedServerNames() {
    const agent = this.availableAgents.find(a => a.id === this.selectedAgentId)
    const allowed = agent?.enabled_mcp_servers
    return (Array.isArray(allowed) && allowed.length > 0) ? allowed : null
  },

  get agentMcpRestricted() {
    return !!this.agentAllowedServerNames
  },

  // Server is visible but blocked by the agent and not session-overridden.
  isMcpBlocked(server) {
    const allowed = this.agentAllowedServerNames
    if (!allowed) return false
    return !allowed.includes(server.name) && !this.sessionMcpOverrides.includes(server.name)
  },

  isMcpOverridden(server) {
    return this.sessionMcpOverrides.includes(server.name)
  },

  toggleSessionOverride(server) {
    const i = this.sessionMcpOverrides.indexOf(server.name)
    if (i >= 0) this.sessionMcpOverrides.splice(i, 1)
    else this.sessionMcpOverrides.push(server.name)
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
    if (typeof this.$store.chat.loadSavedThinkingMode === 'function') this.$store.chat.loadSavedThinkingMode()
    this.thinkingMode = this.$store.chat.thinkingMode || 'auto'
    if (this.selectedAgentId) {
      // Agent config overrides chat defaults: apply its model/provider on load too.
      this.$store.chat.applyAgentConfig()
      this.selectedModel = this.$store.chat.selectedModel
      this.selectedProviderId = this.$store.chat.selectedProviderId
    }
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

    // When the network comes back, replay any messages the user typed while
    // offline. The first one goes into the active conversation; the rest
    // append to whichever thread they were addressed to.
    if (typeof window !== 'undefined') {
      this._drainOfflineQueue = async () => {
        if (!offline.isSupported() || !offline.isOnline()) return
        await offline.drainQueue(async (item) => {
          try {
            const data = await api.post(`/api/conversations/${item.conversationId}/messages`, {
              message: item.body,
              files: item.files || [],
              model: item.model,
            })
            if (data?.request_id) await this.streamResponse(data.request_id)
            return true
          } catch { return false }
        })
      }
      window.addEventListener('online', this._drainOfflineQueue)
    }

    window.addEventListener('conversation-loaded', async () => {
      // Show what the model WILL see on next turn — fetch preview for past chats
      // (previously cleared to "No context yet", confusing for old threads).
      const cid = this.$store.chat.currentConversationId
      if (cid) {
        try {
          const data = await api.get(`/api/conversations/${encodeURIComponent(cid)}/context`)
          this.contextInfo = data.context || null
          this.$store.chat.contextInfo = this.contextInfo
        } catch (e) {
          console.warn('[chat] context preview failed', e)
          this.contextInfo = null
          this.$store.chat.contextInfo = null
        }
      } else {
        this.contextInfo = null
        this.$store.chat.contextInfo = null
      }
      // Fix G: don't leak attached files across threads
      if (this.attachedFiles.length) {
        this.attachedFiles.forEach(f => { if (f.preview) URL.revokeObjectURL(f.preview) })
        this.attachedFiles = []
      }
      // Fix E: session overrides are per-thread; clear on switch
      this.sessionMcpOverrides = []
      this.$nextTick(() => {
        if (this.$store.chat.messages.length === 0) {
          this.$refs?.chatInput?.focus()
        } else {
          this.$refs?.chatInput?.blur()
        }
      })
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
    this.thinkingMode = this.$store.chat.thinkingMode || 'auto'
  },
  // ─── Data Loading ─────────────────────────────────────
  async loadModels() {
    try {
      const data = await api.get('/api/models')
      this.availableModels = data.models || []
      this.availableProviders = data.providers || []
      this.$store.chat.availableModels = this.availableModels
      this.$store.chat.availableProviders = this.availableProviders
      // Single entity: provider derived from model
      if (!this.selectedModel && this.availableModels.length) {
        // default from settings store or first available
        const defModel = this.$store.chat.selectedModel || this.availableModels[0]?.id
        if (defModel) {
          this.selectedModel = defModel
          const m = this.availableModels.find(x=>x.id===defModel)
          this.selectedProviderId = m?.provider_id || this.availableProviders.find(p=>p.is_default)?.id || this.availableProviders[0]?.id || ''
        }
      } else if (this.selectedModel) {
        const m = this.availableModels.find(x=>x.id===this.selectedModel)
        if (m) this.selectedProviderId = m.provider_id
      }
      if (!this.selectedProviderId && this.availableProviders.length) {
        const def = this.availableProviders.find(p => p.is_default) || this.availableProviders[0]
        this.selectedProviderId = def.id
      }
      this.$store.chat.selectedModel = this.selectedModel
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
    if (this.isLoading || this.isUploading) {
      // Steering: a message sent while the model is responding interrupts the
      // current response and sends this as a follow-up instruction.
      if (this.isLoading && !this.isUploading && this.$store.chat.activeStreaming?.isStreaming) {
        return this.steerCurrentStream()
      }
      return
    }

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
      // single entity: provider derived from model live
      const _m = this.availableModels.find(x=>x.id===this.selectedModel)
      const _prov = _m?.provider_id || this.selectedProviderId
      const data = await api.post(
        `/api/conversations/${conversationId}/messages`,
        {
          message: text,
          enable_rag: this.isRAGActive,
          document_ids: this.selectedDocumentIds.includes('all') ? null : this.selectedDocumentIds,
          files: uploadedFiles,
          model: this.selectedModel,
          provider_id: _prov,
          thinking_mode: this.thinkingMode || 'auto'
        }
      )
      await this.streamResponse(data.request_id)
    } catch (e) {
      console.error('[chat] Send error:', e)
      this.isLoading = false
      this.$store.chat.isLoading = false
      // Ponytail: if the network is gone, queue the message so it auto-sends
      // the next time we're online. If the request just 5xx'd, surface a toast.
      if (offline.isSupported() && !offline.isOnline()) {
        await offline.enqueue({ conversationId, body: text, files: uploadedFiles, model: this.selectedModel })
        this.$store.ui.showToast('Offline — message queued, will send when online', 'info')
        return
      }
      this.$store.ui.showToast('Failed to send message', 'error')
    }
  },

  // ─── Steering (KV-preserving graceful interrupt) ─────────
  // Old: hard abort -> llama.cpp slot KV evicted -> full re-prefill.
  // New: queue a steer on the backend, keep rendering for ~0.8-1.2s until
  // the server breaks gracefully at the next chunk boundary (clean aclose
  // -> slot KV retained). Then abort fallback and send the follow-up; its
  // prompt prefix reuses the cached KV (cached_tokens ~ prompt - steer).
  async steerCurrentStream() {
    const text = this.inputMessage.trim()
    if (!text) return
    if (!this.$store.chat.currentConversationId) {
      this.$store.ui.showToast('No active conversation to steer', 'error')
      return
    }
    const cid = this.$store.chat.currentConversationId
    this.inputMessage = ''
    // 1) Queue steer server-side — the active stream will break gracefully
    // at the next chunk boundary, flush the partial, and fast-unregister.
    try {
      await api.post(`/api/conversations/${encodeURIComponent(cid)}/steer`, { message: text })
    } catch (e) {
      console.warn('[steer] queue failed, falling back to hard abort', e)
    }
    this.$store.ui.showToast('Steering queued — finishing thought…', 'info')
    // Flag for steer_ack from the stream
    this._steerAcked = false
    // 2) Grace window: keep streaming so the model can finish its current
    // sentence and the backend can do a clean aclose (KV kept). Wait for
    // steer_ack or at most ~1200ms. Minimum 400ms avoids aborting mid-token.
    const t0 = Date.now()
    while (Date.now() - t0 < 1200) {
      if (this._steerAcked) break
      await new Promise(r => setTimeout(r, 80))
      // after 700ms we have given the server enough time to flush/break
      if (Date.now() - t0 > 700 && this._steerAcked) break
    }
    // 3) Fallback abort if still streaming (server already broke gracefully,
    // this is just client cleanup). The backend already fast-unregistered.
    sseService.abort()
    this.$store.chat.stopStreaming()
    this.isLoading = false
    this.$store.chat.isLoading = false
    this.$store.chat.toolStatus.active = false
    // 4) Small settle so the backend finalizes the partial before the
    // steering message is appended (send_message polls _active_streams).
    await new Promise(r => setTimeout(r, 250))
    this.inputMessage = text
    await this.sendMessage()
  },

  // ─── Stream Response ──────────────────────────────────
  async streamResponse(requestId) {
    if (!requestId) {
      this.isLoading = false
      this.$store.chat.isLoading = false
      return
    }

    const assistantMsg = {
      id: helpers.generateId(), role: 'assistant', content: '',
      blocks: [], created_at: new Date().toISOString()
    }
    this.$store.chat.addMessage(assistantMsg)
    const msgIndex = this.messages.length - 1
    // A previous stream's pending close timer would abort THIS connection
    // silently (AbortError is swallowed by sse.js) — disarm it on new streams.
    clearTimeout(this.pendingTitleTimeout)

    this.$store.chat.startStreaming(
      requestId,
      this.$store.chat.currentConversationId,
      msgIndex,
      this.$store.chat.currentConversationTitle,
      [...this.$store.chat.messages]
    )


    const _effModel = this.selectedModel || this.availableModels[0]?.id
    const _effSm = this.availableModels.find(x=>x.id===_effModel)
    const _effSp = _effSm?.provider_id || this.selectedProviderId
    const handlers = sseService.stream(requestId, this.$store.chat.currentConversationId, {
      enableRag: this.isRAGActive,
      documentIds: this.selectedDocumentIds.includes('all') ? null : this.selectedDocumentIds,
      model: _effModel,
      providerId: _effSp,
      overrideServers: this.sessionMcpOverrides,
      thinkingMode: this.thinkingMode || 'auto'
    })
    handlers.onData((data) => this.processEvent(data, msgIndex))

    handlers.onError((error) => {
      const msg = this.messages[msgIndex]
      if (msg) msg.content += `\n\n❌ Error: ${error.message}`
      this.$store.ui.showToast(`Stream error: ${error.message}`, 'error')
      // Terminal safety net: done may never arrive after a transport error.
      this.isLoading = false
      this.$store.chat.stopStreaming()
    })

    handlers.onComplete(() => {
      if (!this.streamEndedNormally) {
        console.warn('[chat] Stream ended without done event — response may be incomplete')
        this.$store.ui.showToast('Response may be incomplete (stream interrupted)', 'warning')
      }
      this.streamEndedNormally = false
      this.$store.chat.stopStreaming()
      // Always release the UI, even when done was never processed.
      this.isLoading = false
    })
  },

  // ─── Event Processing ─────────────────────────────────
  processEvent(data, msgIndex) {
    // Skip CONTENT updates if we're viewing a different conversation — but
    // terminal events (done/error) must ALWAYS run: they are what clears the
    // loading state and repairs placeholder ids, and skipping them used to
    // wedge the spinner permanently once the user returned to the thread.
    const active = this.$store.chat.activeStreaming
    const foreignConv = active.isStreaming && active.conversationId !== this.$store.chat.currentConversationId
    if (foreignConv && data.type !== 'done' && data.type !== 'error') return

    const msg = this.messages[msgIndex]

    // steer_ack is a control signal — handle even without a target row
    if (data.type === 'steer_ack') {
      this._steerAcked = true
      console.log('[steer] ack received — KV preserved')
      return
    }

    // Content/tool cases need a target row; done/error must survive without
    // one so global cleanup always runs.
    const needsMsg = data.type !== 'done' && data.type !== 'error'
    if (!msg && needsMsg) return
    // Ensure blocks array exists
    if (msg && !msg.blocks) msg.blocks = []

    switch (data.type) {
      case 'content':
        if (!msg) return
        this.appendBlock(msg, 'content', data.content)
        break

      case 'thinking':
        if (!msg) return
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
        // Update the pending Running block (from tool_call_start) instead of pushing a second one
        const errBlock = [...msg.blocks].reverse().find(b =>
          b.type === 'tool_call' && b.status !== 'completed' && b.status !== 'error'
        )
        if (errBlock) {
          errBlock.status = 'error'
          errBlock.result = { error: data.error }
          errBlock.progress = 0
          msg.blocks.push({
            type: 'tool_call', name: data.tool, arguments: {},
            status: 'error', progress: 0, result: {error: data.error},
            sources: [], progress_history: []
          })
        }
        this.$store.ui.showToast(`Tool error: ${data.tool} — ${data.error}`, 'warning')
        break

      case 'auto_action':
        // Autonomous post-response actions: memory extraction, skill reflection, title, jobs
        // Show live in-thread and persist as blocks (backend also saves to DB)
        {
          const action = data.action || 'unknown'
          const status = data.status || 'running'
          const detail = data.detail || {}
          // Find existing running block for same action to update, otherwise push
          let aaBlock = null
          if (status === 'running') {
            msg.blocks.push({
              type: 'auto_action',
              action,
              status,
              detail,
              ts: Date.now()
            })
          } else {
            // completed / skipped / error -> update the last running block for this action
            aaBlock = [...msg.blocks].reverse().find(b => b.type === 'auto_action' && b.action === action && b.status === 'running')
            if (aaBlock) {
              aaBlock.status = status
              aaBlock.detail = detail
              aaBlock.ts = Date.now()
            } else {
              // No running placeholder (e.g. reload case or skipped immediately) -> push final
              msg.blocks.push({
                type: 'auto_action',
                action,
                status,
                detail,
                ts: Date.now()
              })
            }
            // Toast for interesting completions (not for every skipped)
            if (status === 'completed' && action === 'memory' && detail.count) {
              this.$store.ui.showToast(`Saved ${detail.count} memory fact(s)`, 'success')
            } else if (status === 'completed' && action === 'skill' && detail.skill) {
              this.$store.ui.showToast(`Skill draft: ${detail.skill}`, 'success')
            } else if (status === 'error') {
              console.warn(`[auto_action] ${action} error:`, detail.reason)
            }
          }
        }
        break

      case 'error':
        this.$store.chat.toolStatus.active = false
        this.isLoading = false
        this.$store.chat.isLoading = false
        if (msg) msg.blocks.push({ type: 'content', content: `\n\n❌ Error: ${data.error}` })
        this.$store.ui.showToast(`Error: ${data.error}`, 'error')
        break

      case 'title_update':
        this.$store.chat.currentConversationTitle = data.title
        const ci = this.$store.chat.conversations.findIndex(c => c.id === this.$store.chat.currentConversationId)
        if (ci !== -1) this.$store.chat.conversations[ci].title = data.title
        break

      case 'metrics':
        {
          const m = data.metrics || {}
          msg.metadata = msg.metadata || {}
          // Merge — aggregated final overwrites per-iteration partials
          msg.metadata.metrics = m
          // Keep a direct reference for simpler template access and reactivity
          msg._metrics = m
          // Ensure model badge exists for live streaming (not yet persisted)
          if (!msg.metadata.model) {
            const fallbackModel = this.contextInfo?.model || this.selectedModel || ''
            if (fallbackModel) msg.metadata.model = fallbackModel
          }
        }
        break

      case 'done': {
        // Placeholder created with a fake helpers.generateId() gets patched
        // with the DB id — second regenerate works without refresh.
        //
        // Repair is group-wide: if a conversation switch or an interrupted
        // stream left sibling rows of the same version_group holding
        // never-patched client ids, later DELETE/regen calls would 404
        // against the DB. Patch the indexed row and drop the stale siblings.
        if (msg) {
          if (data.message_id) msg.id = data.message_id
          if (data.version_group) {
            msg.version_group = data.version_group
            msg.version = data.version || msg.version
            msg.max_version = Math.max(msg.max_version || 0, data.version || 1)
          }
          // Client-generated ids are never UUIDs; anything else sharing the
          // group after this patch is a stale local duplicate.
          const g = data.version_group || msg.version_group
          if (g) {
            const isDbId = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
            this.$store.chat.messages = this.$store.chat.messages.filter(
              m => m === msg || m.version_group !== g ||
                   (!!m.id && isDbId.test(m.id)))
          }
        }
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

  // ─── Metrics helpers ───────────────────────────────
  getMetrics(message) {
    return message?.metadata?.metrics || message?._metrics || null
  },
  formatTps(m) {
    const v = m?.tokens_per_second
    if (v == null) return null
    return (v >= 10 ? Math.round(v) : v.toFixed(1)) + ' tok/s'
  },
  formatDuration(ms) {
    if (ms == null) return null
    if (ms < 1000) return ms + ' ms'
    return (ms / 1000).toFixed(1) + ' s'
  },
  metricsShortLabel(message) {
    const m = this.getMetrics(message)
    if (!m) return ''
    const parts = []
    const tps = this.formatTps(m)
    if (tps) parts.push(tps)
    if (m.ttft_ms != null) parts.push('TTFT ' + this.formatDuration(m.ttft_ms))
    return parts.join(' · ')
  },
  metricsTooltip(message) {
    const m = this.getMetrics(message)
    if (!m) return ''
    const lines = []
    if (m.prompt_tokens != null) lines.push('Prompt: ' + m.prompt_tokens + (m.cached_tokens ? ' (' + m.cached_tokens + ' cached)' : '') + ' tok')
    if (m.completion_tokens != null) lines.push('Completion: ' + m.completion_tokens + ' tok')
    if (m.total_tokens != null) lines.push('Total: ' + m.total_tokens + ' tok')
    if (m.tokens_per_second != null) lines.push('Speed: ' + this.formatTps(m) + (m.prompt_per_second ? ' (prompt ' + (m.prompt_per_second >= 10 ? Math.round(m.prompt_per_second) : m.prompt_per_second.toFixed(1)) + ' tok/s)' : ''))
    if (m.ttft_ms != null) lines.push('TTFT: ' + this.formatDuration(m.ttft_ms))
    if (m.total_duration_ms != null) lines.push('Duration: ' + this.formatDuration(m.total_duration_ms))
    return lines.join('\n')
  },

  // Token counts render as 12.4K / 1.2M — raw counts are unreadable at a glance.
  formatTokens(n) {
    if (n == null || isNaN(n)) return '—'
    const v = Number(n)
    if (v >= 1e6) {
      const m = v / 1e6
      return (m >= 100 ? Math.round(m) : m.toFixed(1)).toString().replace(/\.0$/, '') + 'M'
    }
    if (v >= 1e3) {
      const k = v / 1e3
      return (k >= 100 ? Math.round(k) : k.toFixed(1)).toString().replace(/\.0$/, '') + 'K'
    }
    return String(Math.round(v))
  },

  // id→context_window index rebuilt only when the model list reference changes;
  // _mwv bumps so cached per-message context invalidates when the list is replaced.
  _ensureWinIdx() {
    if (this._modelWindowRef !== this.availableModels) {
      const idx = new Map()
      for (const m of (this.availableModels || [])) if (m && m.context_window) idx.set(m.id, m.context_window)
      this._modelWindowRef = this.availableModels
      this._modelWindowIdx = idx
      this._mwv = (this._mwv || 0) + 1
    }
  },
  _winFor(modelId) {
    if (!modelId) return null
    return (this._modelWindowIdx && this._modelWindowIdx.get(modelId)) || null
  },

  // Compute {used, win, pct} ONCE per message and cache by message identity.
  // The template calls these ~30× per message (10-tick bar + label + hover card);
  // this collapses that to a single computation. It self-invalidates when the
  // message's token totals, model, selected model, or the model list change — so
  // live streaming and model switches stay correct.
  _ctx(message) {
    this._ensureWinIdx()
    if (!this._ctxCache) this._ctxCache = new WeakMap()
    const metrics = message ? this.getMetrics(message) : null
    const meta = message && message.metadata
    const model = meta ? meta.model : undefined
    const sel = this.selectedModel
    const sig = metrics
      ? `${metrics.total_tokens | 0}:${metrics.prompt_tokens | 0}:${(metrics._iterations && metrics._iterations.length) | 0}`
      : '-'
    const mv = this._mwv || 0
    const hit = this._ctxCache.get(message)
    if (hit && hit.mv === mv && hit.sig === sig && hit.model === model && hit.sel === sel) return hit.val
    let used = null
    if (metrics) {
      const its = metrics._iterations
      if (Array.isArray(its) && its.length) {
        const last = its[its.length - 1] || {}
        used = last.total_tokens ?? ((last.prompt_tokens || 0) + (last.completion_tokens || 0))
      }
      if (!used) used = metrics.total_tokens ?? ((metrics.prompt_tokens || 0) + (metrics.completion_tokens || 0)) ?? null
    }
    const win = this._winFor(model) || this._winFor(sel) || null
    const pct = (!win || !used) ? null : Math.min(100, Math.round((used / win) * 100))
    const val = { used: used || null, win, pct }
    this._ctxCache.set(message, { mv, sig, model, sel, val })
    return val
  },

  // Tokens occupying the window. A tool-loop turn re-sends a growing prompt each
  // iteration, so the aggregated `total_tokens` double-counts — the last
  // iteration's total is the real occupancy.
  contextUsed(message) { return this._ctx(message).used },

  // 0-100, clamped; null when the window is unknown.
  contextPct(message) { return this._ctx(message).pct },

  // Context window size for the message's model (falls back to the live selection).
  contextWindow(message) { return this._ctx(message).win },

  // Colour escalates with pressure so a glance is enough: accent → warning → error.
  contextColor(pct) {
    if (pct == null) return 'var(--text-tertiary)'
    if (pct >= 90) return 'var(--error)'
    if (pct >= 70) return 'var(--warning)'
    return 'var(--accent-primary)'
  },

  // Always-visible label, e.g. "12.4K / 32K · 39%" or "12.4K tok" when no window.
  contextLabel(message) {
    const used = this.contextUsed(message)
    if (!used) return ''
    const win = this.contextWindow(message)
    if (!win) return this.formatTokens(used) + ' tok'
    const pct = this.contextPct(message)
    return `${this.formatTokens(used)} / ${this.formatTokens(win)} · ${pct}%`
  },

  contextTitle(message) {
    const used = this.contextUsed(message)
    const win = this.contextWindow(message)
    if (!used) return 'Context usage unavailable'
    if (!win) return `${used.toLocaleString()} tokens used (context window unknown for this model)`
    const pct = this.contextPct(message)
    return `${used.toLocaleString()} of ${win.toLocaleString()} tokens — ${pct}% of the context window`
  },

  // ─── Tool Approval ─────────────────────────────────────
  async respondApproval(block, approved) {
    if (block._approving) return
    block._approving = true
    const active = this.$store.chat.activeStreaming
    if (!active?.requestId) {
      block._approving = false
      this.$store.ui.showToast('No active request — page was refreshed. The partial response was saved; send a follow-up to retry.', 'warning')
      block.status = 'error'
      block.result = { error: 'Approval expired due to page reload. Send a message like "retry" or "approve and continue" to retry.' }
      this.messages = [...this.$store.chat.messages]
      return
    }
    try {
      await api.post(`/api/tools/${active.requestId}/approve`, {
        decision: approved,
        approval_key: block.approval_key || null
      })
      block.status = approved ? 'approved' : 'denied'
    } catch (e) {
      const msg = (e && e.message) ? e.message : String(e)
      if (msg.includes('404') || msg.includes('No pending')) {
        this.$store.ui.showToast('Approval window expired (15 min) or already handled — send a follow-up message to retry.', 'warning')
        block.status = 'error'
        block.result = { error: 'Approval expired — no pending gate. The previous response was saved; send a follow-up to retry the command.' }
        this.messages = [...this.$store.chat.messages]
      } else {
        this.$store.ui.showToast('Failed to send decision: ' + msg, 'error')
      }
    } finally {
      block._approving = false
    }
  },

  async deleteMessage(id, e) {
    e?.stopPropagation()
    const target = this.messages.find(m => m.id === id) || this.$store.chat.messages.find(m => m.id === id)
    if (!target) return
    const vg = target.version_group || null
    const ver = target.version || null
    if (!confirm('Delete this message?')) return
    try {
      // Version-aware delete: backend resolves the exact row via
      // (version_group, version) even if id is a stale representative.
      // Without ?version it always deletes the row matching id, which for a
      // deduplicated view is the *latest* — the bug you observed.
      try {
        if (vg && ver != null) {
          await api.delete(`/api/messages/${encodeURIComponent(id)}?version=${encodeURIComponent(ver)}`)
        } else {
          await api.delete(`/api/messages/${encodeURIComponent(id)}`)
        }
      } catch (err) {
        // Stale client-side id (e.g. a placeholder whose done event never
        // patched it) — resolve the authoritative row for this slot and retry
        // with the version-aware form.
        if (!vg) throw err
        const vs = await api.get(`/api/versions/${encodeURIComponent(vg)}`)
        const tv = (vs.versions || []).find(v => v.version === ver) ||
                   (vs.versions || []).slice(-1)[0]
        if (!tv) throw err
        await api.delete(`/api/messages/${encodeURIComponent(tv.id)}?version=${encodeURIComponent(tv.version)}`)
        // Use the resolved id for optimistic removal below
        id = tv.id
      }
      // Optimistically remove the deleted row from the raw store
      this.$store.chat.removeMessage(id)
      // If versioned, reload to show the next-latest version in place
      if (vg) {
        try {
          const cid = this.$store.chat.currentConversationId
          if (cid) {
            const data = await api.get(`/api/conversations/${encodeURIComponent(cid)}`)
            this.$store.chat.messages = data.messages || []
          }
        } catch (err) {
          console.warn('[chat] reload after delete failed', err)
        }
      }
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

    // Position bookkeeping BEFORE any mutation: map raw store objects to their
    // index so we can splice the replacement into exactly the same slot.
    const rawMsgs = this.$store.chat.messages
    const posOf = new Map(rawMsgs.map((m, i) => [m, i]))
    const rawIdx = rawMsgs.findIndex(m => m.id === id)

    try {
      const data = await api.post(
        `/api/conversations/${this.$store.chat.currentConversationId}/regenerate`,
        { message_id: id }
      )
      // Disarm any previous stream's pending close timer — it would abort
      // this connection silently.
      clearTimeout(this.pendingTitleTimeout)


      // Displayed position of the replaced message = count of shown messages
      // that sit before it in the raw array (dedup getter preserves order).
      let dispIdx = 0
      for (const m of this.messages) {
        const p = posOf.get(m)
        if (p === undefined || p >= rawIdx) break
        dispIdx++
      }

      // Remove ONLY the replaced entry — earlier code sliced off every later
      // turn, wiping the rest of a mid-thread regeneration from view.
      // The old version stays in the DB as a prior version of the group.
      if (rawIdx !== -1) rawMsgs.splice(rawIdx, 1)

      const newMsg = {
        id: helpers.generateId(), role: 'assistant', content: '',
        blocks: [], created_at: new Date().toISOString(),
        version: data.version || 1,
        version_group: data.version_group || null,
        max_version: data.version || 1,  // Total version count (all pills shown up to this)
        turn_index: data.turn_index ?? null
      }
      // Insert the placeholder into the freed slot so later turns stay put and
      // processEvent(data, msgIndex) streams into the right message.
      rawMsgs.splice(dispIdx, 0, newMsg)
      const newIdx = dispIdx

      // Pass version info to SSE stream for backend to save correctly
      const _effRModel = this.selectedModel || this.availableModels[0]?.id
      const _effRm = this.availableModels.find(x=>x.id===_effRModel)
      const _effRp = _effRm?.provider_id || this.selectedProviderId
      const handlers = sseService.stream(data.request_id, this.$store.chat.currentConversationId, {
        model: _effRModel,
        isRegenerate: true,
        version: data.version,
        anchorMessageId: data.anchor_user_message_id,
        providerId: _effRp,
        versionGroup: data.version_group,
        overrideServers: this.sessionMcpOverrides,
        thinkingMode: this.thinkingMode || 'auto'
      })
      handlers.onData((data) => this.processEvent(data, newIdx))

      handlers.onError((error) => {
        console.error('[chat] Regenerate stream error:', error)
        const msg = this.messages[newIdx]
        if (msg) msg.content += `\n\n❌ Error: ${error.message}`
        this.$store.ui.showToast(`Stream error: ${error.message}`, 'error')
        // Terminal safety net: done may never arrive after a transport error.
        this.isLoading = false
        this.$store.chat.stopStreaming()
      })

      handlers.onComplete(() => {
        if (!this.streamEndedNormally) {
          console.warn('[chat] Regenerate stream ended without done event — response may be incomplete')
          this.$store.ui.showToast('Response may be incomplete (stream interrupted)', 'warning')
        }
        this.streamEndedNormally = false
        this.$store.chat.stopStreaming()
        // Always release the UI, even when done was never processed.
        this.isLoading = false
      })
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

      // Sync id to the target row so subsequent delete/edit act on the
      // displayed version, not the stale latest. Keep group stable.
      msg.id = targetVersion.id
      msg.version_group = targetVersion.version_group || msg.version_group
      // Swap the message content with the target version
      msg.content = targetVersion.content
      msg.version = targetVersion.version
      msg.max_version = maxVer
      msg.blocks = targetVersion.blocks || []
      msg.tool_calls = targetVersion.tool_calls
      msg.thinking = targetVersion.thinking
      msg.metadata = targetVersion.metadata || msg.metadata
      msg._metrics = targetVersion.metadata?.metrics || null
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
  smartText: (t) => helpers.smartText(t),
  toolResultView: (r) => helpers.toolResultView(r),
  assetPreviews: (t) => helpers.assetPreviews(t),
  formatDate: (s) => formatters.formatDate(s),

  // Arguments preview: pretty JSON with real line breaks, collapsed when huge
  // (e.g. write_file with an entire file body as content).
  argsText(args, key) {
    const raw = typeof args === 'object' ? JSON.stringify(args, null, 2) : String(args ?? '')
    const text = helpers.parseEscapes(raw)
    if (!this.isExpanded(key) && text.length > 2000) {
      return text.slice(0, 2000) + '\n… (truncated — click "Show full" to expand)'
    }
    return text
  },

  async copyText(text) {
    const success = await helpers.copyToClipboard(text)
    this.$store.ui.showToast(success ? 'Copied!' : 'Copy failed', success ? 'success' : 'error')
  },

  async copyMessage(msg) {
    const success = await helpers.copyToClipboard(msg.content)
    this.$store.ui.showToast(success ? 'Copied!' : 'Copy failed', success ? 'success' : 'error')
  },

  // ─── UI Handlers ──────────────────────────────────────
  updateSelectedModel() {
    this.$store.chat.setModel(this.selectedModel)
    this.selectedProviderId = this.$store.chat.selectedProviderId
  },

  selectModel(id) {
    this.selectedModel = id
    const m = this.availableModels.find(x=>x.id===id)
    this.selectedProviderId = m?.provider_id || this.selectedProviderId
    this.$store.chat.setModel(this.selectedModel)
    this.selectedProviderId = this.$store.chat.selectedProviderId
  },

  selectAgent(id) {
    this.selectedAgentId = id || null
    this.sessionMcpOverrides = []  // overrides are per-agent; clear on switch
    this.updateSelectedAgent()
  },

  async updateSelectedAgent() {
    this.$store.chat.setAgent(this.selectedAgentId)
    this.$store.chat.applyAgentConfig(true)
    this.selectedModel = this.$store.chat.selectedModel
    const m = this.availableModels.find(x=>x.id===this.selectedModel)
    this.selectedProviderId = m?.provider_id || this.$store.chat.selectedProviderId
    this.$store.chat.selectedProviderId = this.selectedProviderId
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

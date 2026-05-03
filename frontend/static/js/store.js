/**
 * Simplified Alpine.js Stores
 * Merged: chat + ui (replaces: chat, settings, documents, tts)
 */

const stores = {
  // ─── Chat Store ─────────────────────────────────────────
  chat: {
    // Conversations
    conversations: [],
    currentConversationId: null,
    currentConversationTitle: 'New Chat',

    // Messages
    messages: [],

    // Input
    inputMessage: '',

    // Loading / Streaming
    isLoading: false,
    activeStreaming: {
      isStreaming: false,
      requestId: null,
      conversationId: null,
      msgIndex: null,
      conversationTitle: '',
      messages: []
    },

    // Tool status
    toolStatus: { active: false, tool: '', status: '', progress: null },

    // Model & Agent
    selectedModel: '',
    availableModels: [],
    selectedAgentId: null,
    availableAgents: [],
    currentAgentConfig: null,

    // RAG
    enableRAG: false,

    // Editing
    editingMessageId: null,
    editContent: '',

    // ─── Actions ────────────────────────────────────────
    addConversation(conv) {
      this.conversations.unshift(conv)
    },
    updateConversation(id, updates) {
      const idx = this.conversations.findIndex(c => c.id === id)
      if (idx !== -1) Object.assign(this.conversations[idx], updates)
    },
    removeConversation(id) {
      this.conversations = this.conversations.filter(c => c.id !== id)
    },
    addMessage(msg) {
      this.messages.push(msg)
    },
    updateMessage(id, updates) {
      const msg = this.messages.find(m => m.id === id)
      if (msg) Object.assign(msg, updates)
    },
    removeMessage(id) {
      this.messages = this.messages.filter(m => m.id !== id)
    },
    clearMessages() {
      this.messages = []
    },
    setModel(modelId) {
      this.selectedModel = modelId
      localStorage.setItem('selectedModel', modelId)
    },
    loadSavedModel() {
      const saved = localStorage.getItem('selectedModel')
      if (saved && this.availableModels.some(m => m.id === saved)) {
        this.selectedModel = saved
      }
    },
    setAgent(agentId) {
      this.selectedAgentId = agentId
      const agent = this.availableAgents.find(a => a.id === agentId)
      this.currentAgentConfig = agent || null
      localStorage.setItem('selectedAgentId', agentId)
    },
    loadSavedAgent() {
      const saved = localStorage.getItem('selectedAgentId')
      if (saved && this.availableAgents.some(a => a.id === parseInt(saved))) {
        this.selectedAgentId = parseInt(saved)
        this.currentAgentConfig = this.availableAgents.find(a => a.id === this.selectedAgentId)
      }
    },
    startStreaming(requestId, conversationId, msgIndex, title = '', msgs = []) {
      this.activeStreaming = {
        isStreaming: true,
        requestId,
        conversationId,
        msgIndex,
        conversationTitle: title,
        messages: [...msgs]
      }
      this.isLoading = true
    },
    stopStreaming() {
      this.activeStreaming = {
        isStreaming: false,
        requestId: null,
        conversationId: null,
        msgIndex: null,
        conversationTitle: '',
        messages: []
      }
      this.isLoading = false
      this.toolStatus = { active: false, tool: '', status: '', progress: null }
    },
    applyAgentConfig() {
      const agent = this.currentAgentConfig
      if (!agent) return
      if (agent.model && this.availableModels.some(m => m.id === agent.model)) {
        this.selectedModel = agent.model
      }
      this.enableRAG = !!agent.enable_rag
    }
  },

  // ─── UI Store ──────────────────────────────────────────
  ui: {
    // Theme
    darkMode: localStorage.getItem('darkMode') === 'true',

    // Sidebar
    sidebarCollapsed: false,
    sidebarWidth: parseInt(localStorage.getItem('sidebarWidth') || '280'),

    // Panels
    showMcpPanel: false,
    showAgents: false,
    showSettings: false,
    showDocuments: false,

    // Toast
    toast: { show: false, message: '', type: 'success' },

    // Settings data
    settingsData: {},
    mcpServers: [],
    mcpTools: [],
    documents: [],

    // MCP form state
    newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}' },
    editingServer: false,
    editServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', enabled: true, originalName: '' },

    // Settings tab
    settingsTab: 'general',

    // ─── Actions ────────────────────────────────────────
    toggleDarkMode() {
      this.darkMode = !this.darkMode
      localStorage.setItem('darkMode', this.darkMode)
      document.documentElement.setAttribute('data-theme', this.darkMode ? 'dark' : 'light')
    },
    initTheme() {
      document.documentElement.setAttribute('data-theme', this.darkMode ? 'dark' : 'light')
    },
    toggleSidebar() {
      this.sidebarCollapsed = !this.sidebarCollapsed
      this.sidebarWidth = this.sidebarCollapsed ? 68 : 280
      localStorage.setItem('sidebarWidth', this.sidebarWidth)
    },
    setSidebarWidth(width) {
      this.sidebarWidth = width
      localStorage.setItem('sidebarWidth', width)
    },
    showToast(message, type = 'success') {
      this.toast = { show: true, message, type }
      setTimeout(() => { this.toast = { show: false, message: '', type: 'success' } }, 3000)
    },
    openMcpPanel() {
      this.showMcpPanel = true
    },
    closeMcpPanel() {
      this.showMcpPanel = false
    },
    openAgents() {
      this.showAgents = true
    },
    closeAgents() {
      this.showAgents = false
    },
    openSettings() {
      this.showSettings = true
    },
    closeSettings() {
      this.showSettings = false
    },
    openDocuments() {
      this.showDocuments = true
    },
    closeDocuments() {
      this.showDocuments = false
    }
  }
}

// ═══════════════════════════════════════════════════════════
// Modal Component Factory — reusable modal with consistent behavior
// ═══════════════════════════════════════════════════════════

export function createModal(storeKey, openMethod, closeMethod) {
  return function () {
    return {
      open: false,
      openModal() {
        this.open = true
        if (typeof openMethod === 'function') {
          try { openMethod() } catch(e) { console.error('[modal] openMethod error:', e) }
        }
      },
      closeModal() {
        // Close via store method if provided
        if (closeMethod) {
          try {
            if (typeof closeMethod === 'function') closeMethod()
            else if (typeof $store !== 'undefined' && typeof $store.ui[closeMethod] === 'function') {
              $store.ui[closeMethod]()
            }
          } catch(e) { console.error('[modal] closeMethod error:', e) }
        }
        this.open = false
      }
    }
  }
}

export {}

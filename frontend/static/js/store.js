/**
 * Alpine.js Store - Shared State Management
 */

// Store definitions
const storeDefinitions = {
  chat: {
    conversations: [],
    currentConversationId: null,
    currentConversationTitle: 'New Chat',
    messages: [],
    inputMessage: '',
    isLoading: false,
    selectedModel: '',
    availableModels: [],
    editingMessageId: null,
    editContent: '',
    enableWebSearch: false,
    enableRAG: false,
    toolStatus: { active: false, tool: '', status: '', progress: null, data: null },
    sidebarCollapsed: false,
    expandedToolCallBlocks: {},
    expandedThinkingBlocks: {},
    expandedThinking: {},
    expandedSources: {},
    toast: { show: false, message: '', type: 'success' },

    // Agent selection state
    availableAgents: [],
    selectedAgentId: null,
    currentAgentConfig: null, // Stores the full agent config when selected

    // Active streaming state - persists across navigation
    activeStreaming: {
      isStreaming: false,
      requestId: null,
      conversationId: null,
      msgIndex: null
    },

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

    // Agent selection methods
    setAgent(agentId) {
      this.selectedAgentId = agentId
      // Find and store the full agent config
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

    showToast(message, type = 'success') {
      this.toast.message = message
      this.toast.type = type
      this.toast.show = true
      setTimeout(() => { this.toast.show = false }, 2500)
    },

    // Streaming state management
    startStreaming(requestId, conversationId, msgIndex) {
      this.activeStreaming = {
        isStreaming: true,
        requestId,
        conversationId,
        msgIndex
      }
      this.isLoading = true
    },

    stopStreaming() {
      this.activeStreaming = {
        isStreaming: false,
        requestId: null,
        conversationId: null,
        msgIndex: null
      }
      this.isLoading = false
      this.toolStatus.active = false
    }
  },
  settings: {
    show: false,
    data: {},
    mcpServers: [],
    mcpTools: [],
    activeTab: 'general',
    newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '' },
    update(data) {
      Object.assign(this.data, data)
    }
  },
  documents: {
    show: false,
    list: []
  },
  tts: {
    available: false,
    currentAudio: null,
    currentAudioMessageId: null,
    isPlaying: false,
    loading: {},
    cleanup() {
      this.currentAudio = null
      this.currentAudioMessageId = null
      this.isPlaying = false
    }
  }
}

// Register stores when Alpine initializes
document.addEventListener('alpine:init', () => {
  Alpine.store('chat', storeDefinitions.chat)
  Alpine.store('settings', storeDefinitions.settings)
  Alpine.store('documents', storeDefinitions.documents)
  Alpine.store('tts', storeDefinitions.tts)
})

export { storeDefinitions }

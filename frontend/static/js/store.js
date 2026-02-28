/**
 * Alpine.js Store - Shared State Management
 */
console.log('[STORE] Module loading...')

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
    showToast(message, type = 'success') {
      this.toast.message = message
      this.toast.type = type
      this.toast.show = true
      setTimeout(() => { this.toast.show = false }, 2500)
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
  console.log('[STORE] alpine:init event fired')
  console.log('[STORE] Alpine available:', typeof window.Alpine !== 'undefined')
  Alpine.store('chat', storeDefinitions.chat)
  Alpine.store('settings', storeDefinitions.settings)
  Alpine.store('documents', storeDefinitions.documents)
  Alpine.store('tts', storeDefinitions.tts)
  console.log('[STORE] All stores registered')
})

console.log('[STORE] Module loaded, storeDefinitions:', Object.keys(storeDefinitions))
export { storeDefinitions }

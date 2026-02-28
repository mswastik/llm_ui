/**
 * Settings Component - Application settings and MCP server management
 */
import { api } from '../utils.js'

console.log('[SETTINGS] Module loading...')

// Define component factory as global function
window.settings = () => {
  console.log('[SETTINGS] Component factory called')
  return {
    // Local state
    show: false,
    activeTab: 'general',
    settings: {},
    mcpServers: [],
    mcpTools: [],
    newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '' },

    // Initialization
    async init() {
      console.log('[SETTINGS] init() called')
      console.log('[SETTINGS] $store available:', !!this.$store)
      
      // Sync from store
      this.show = this.$store.settings.show
      this.activeTab = this.$store.settings.activeTab
      this.settings = this.$store.settings.data
      this.mcpServers = this.$store.settings.mcpServers
      this.newServer = { ...this.$store.settings.newServer }
      
      console.log('[SETTINGS] Initial state:', { show: this.show, activeTab: this.activeTab })
      
      await this.loadSettings()
    },

    // Settings
    async loadSettings() {
      console.log('[SETTINGS] loadSettings() called')
      try {
        const data = await api.get('/api/settings')
        this.settings = data
        this.$store.settings.data = { ...data }
      } catch (error) {
        console.error('[SETTINGS] Error loading settings:', error)
      }
    },

    async saveSettings() {
      console.log('[SETTINGS] saveSettings() called')
      try {
        const data = await api.put('/api/settings', this.settings)
        this.settings = data
        this.$store.settings.data = { ...data }
        this.$store.chat.showToast('Settings saved!', 'success')
        this.show = false
        this.$store.settings.show = false
      } catch (error) {
        this.$store.chat.showToast('Error saving settings', 'error')
      }
    },

    onTtsEngineChange() {
      console.log('[SETTINGS] onTtsEngineChange() called')
      if (this.settings.tts_engine === 'kokoro') {
        this.settings.tts_voice = 'af_bella'
        this.settings.kokoro_lang = 'a'
        if (!this.settings.kokoro_device) {
          this.settings.kokoro_device = 'cpu'
        }
      } else if (this.settings.tts_engine === 'edge-tts') {
        this.settings.tts_voice = 'en-IN-NeerjaNeural'
      } else if (this.settings.tts_engine === 'pyttsx3') {
        this.settings.tts_voice = ''
      }
    },

    // MCP Servers
    async loadMCPServers() {
      console.log('[SETTINGS] loadMCPServers() called')
      try {
        const data = await api.get('/api/mcp/servers')
        this.mcpServers = data.servers
        this.$store.settings.mcpServers = data.servers
      } catch (error) {
        console.error('[SETTINGS] Error loading MCP servers:', error)
      }
    },

    async loadMCPTools() {
      console.log('[SETTINGS] loadMCPTools() called')
      try {
        const data = await api.get('/api/mcp/tools')
        this.mcpTools = data.tools || []
        this.$store.settings.mcpTools = data.tools || []
      } catch (error) {
        console.error('[SETTINGS] Error loading MCP tools:', error)
      }
    },

    async addMCPServer() {
      console.log('[SETTINGS] addMCPServer() called')
      try {
        let args = []
        try { 
          args = JSON.parse(this.newServer.args) 
        } catch {
          this.$store.chat.showToast('Invalid JSON for args', 'error')
          return
        }
        
        if (this.newServer.transport_type !== 'stdio' && !this.newServer.url) {
          this.$store.chat.showToast('URL required for SSE/HTTP', 'error')
          return
        }
        
        if (this.newServer.transport_type === 'stdio' && !this.newServer.command) {
          this.$store.chat.showToast('Command required for stdio', 'error')
          return
        }
        
        await api.post('/api/mcp/servers', {
          name: this.newServer.name,
          transport_type: this.newServer.transport_type,
          command: this.newServer.command,
          args: args,
          env: {},
          url: this.newServer.url || null
        })
        
        await this.loadMCPServers()
        await this.loadMCPTools()
        
        this.newServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '' }
        this.$store.chat.showToast('Server added!', 'success')
      } catch (error) {
        this.$store.chat.showToast('Error adding server', 'error')
      }
    },

    async removeMCPServer(serverName) {
      console.log('[SETTINGS] removeMCPServer() called with:', serverName)
      if (!confirm(`Remove "${serverName}"?`)) return
      try {
        await api.delete(`/api/mcp/servers/${serverName}`)
        await this.loadMCPServers()
        await this.loadMCPTools()
        this.$store.chat.showToast('Server removed', 'success')
      } catch (error) {
        this.$store.chat.showToast('Error removing server', 'error')
      }
    },

    async reconnectMCPServer(serverName) {
      console.log('[SETTINGS] reconnectMCPServer() called with:', serverName)
      try {
        await api.post(`/api/mcp/servers/${serverName}/reconnect`)
        await this.loadMCPServers()
        await this.loadMCPTools()
        this.$store.chat.showToast('Reconnected', 'success')
      } catch (error) {
        this.$store.chat.showToast('Error reconnecting', 'error')
      }
    },

    async refreshMCPServerTools(serverName) {
      console.log('[SETTINGS] refreshMCPServerTools() called with:', serverName)
      try {
        await api.post(`/api/mcp/servers/${serverName}/refresh`)
        await this.loadMCPServers()
        await this.loadMCPTools()
        this.$store.chat.showToast('Tools refreshed', 'success')
      } catch (error) {
        this.$store.chat.showToast('Error refreshing', 'error')
      }
    }
  }
}

console.log('[SETTINGS] window.settings defined:', typeof window.settings)

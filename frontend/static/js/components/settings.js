/**
 * Settings Component - Application settings and MCP server management
 */
import { api } from '../utils.js'

// Define component factory as global function
window.settings = () => {
  const component = {
    // Local state
    activeTab: 'general',
    settings: {},
    mcpServers: [],
    mcpTools: [],
    newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '' },
    
    // show is a getter/setter that syncs with store
    get show() {
      return this.$store?.settings?.show || false
    },
    set show(val) {
      this.$store.settings.show = val
    },

    // Initialization
    async init() {
      // Sync from store
      this.activeTab = this.$store.settings.activeTab
      this.settings = this.$store.settings.data
      this.mcpServers = this.$store.settings.mcpServers
      this.newServer = { ...this.$store.settings.newServer }
      
      await this.loadSettings()
    },

    // Settings
    async loadSettings() {
      try {
        const data = await api.get('/api/settings')
        this.settings = data
        this.$store.settings.data = { ...data }
      } catch (error) {
        console.error('[SETTINGS] Error loading settings:', error)
      }
    },

    async saveSettings() {
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
      try {
        const data = await api.get('/api/mcp/servers')
        this.mcpServers = data.servers
        this.$store.settings.mcpServers = data.servers
      } catch (error) {
        console.error('[SETTINGS] Error loading MCP servers:', error)
      }
    },

    async loadMCPTools() {
      try {
        const data = await api.get('/api/mcp/tools')
        this.mcpTools = data.tools || []
        this.$store.settings.mcpTools = data.tools || []
      } catch (error) {
        console.error('[SETTINGS] Error loading MCP tools:', error)
      }
    },

    async addMCPServer() {
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
  
  return component
}


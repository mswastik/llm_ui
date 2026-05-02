/**
 * Settings Component — Unified settings modal with MCP server management
 */
import { api } from '../utils.js'

const settings = () => ({
  activeTab: 'general',
  settings: {},
  mcpServers: [],
  mcpTools: [],
  newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}' },
  editingServer: false,
  editServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', enabled: true, originalName: '' },

  async init() {
    await this.loadSettings()
    await this.loadMCPServers()
  },

  // ─── Settings ─────────────────────────────────────────
  async loadSettings() {
    try {
      this.settings = await api.get('/api/settings')
      this.$store.ui.settingsData = this.settings
    } catch (e) { console.error('[settings] Error:', e) }
  },

  async saveSettings() {
    try {
      await api.put('/api/settings', this.$store.ui.settingsData)
      this.$store.ui.showToast('Settings saved!', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error saving: ' + e.message, 'error')
    }
  },

  onTtsEngineChange() {
    if (this.settings.tts_engine === 'kokoro') {
      this.settings.tts_voice = 'af_bella'
      this.settings.kokoro_lang = 'a'
      this.settings.kokoro_device = this.settings.kokoro_device || 'cpu'
    } else if (this.settings.tts_engine === 'edge-tts') {
      this.settings.tts_voice = 'en-IN-NeerjaNeural'
    }
  },

  // ─── MCP Servers ──────────────────────────────────────
  async loadMCPServers() {
    try {
      const [serversData, toolsData] = await Promise.all([
        api.get('/api/mcp/servers'),
        api.get('/api/mcp/tools')
      ])
      this.mcpServers = (serversData.servers || []).map(s => ({
        ...s,
        tools: (toolsData.tools || []).filter(t => t.server === s.name),
        toolsExpanded: false,
        enabled: s.enabled !== false
      }))
      this.$store.ui.mcpServers = this.mcpServers
      this.$store.ui.mcpTools = toolsData.tools || []
    } catch (e) { console.error('[settings] MCP:', e) }
  },

  async addMCPServer() {
    try {
      const args = JSON.parse(this.newServer.args)
      const env = this.newServer.env?.trim() ? JSON.parse(this.newServer.env) : {}

      if (this.newServer.transport_type !== 'stdio' && !this.newServer.url) {
        this.$store.ui.showToast('URL required for SSE/HTTP', 'error')
        return
      }
      if (this.newServer.transport_type === 'stdio' && !this.newServer.command) {
        this.$store.ui.showToast('Command required for stdio', 'error')
        return
      }

      await api.post('/api/mcp/servers', {
        name: this.newServer.name,
        transport_type: this.newServer.transport_type,
        command: this.newServer.command,
        args, env,
        url: this.newServer.url || null
      })

      await this.loadMCPServers()
      this.newServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}' }
      this.$store.ui.showToast('Server added!', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error: ' + e.message, 'error')
    }
  },

  openEdit(server) {
    this.editServer = {
      name: server.name, transport_type: server.transport_type,
      command: server.command || '',
      args: JSON.stringify(server.args || []),
      url: server.url || '',
      env: server.env ? JSON.stringify(server.env) : '{}',
      enabled: server.enabled !== false,
      originalName: server.name
    }
    this.editingServer = true
  },

  closeEdit() {
    this.editingServer = false
    this.editServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', enabled: true, originalName: '' }
  },

  async saveEdit() {
    try {
      const args = JSON.parse(this.editServer.args)
      const env = this.editServer.env?.trim() ? JSON.parse(this.editServer.env) : {}

      if (this.editServer.transport_type !== 'stdio' && !this.editServer.url) {
        this.$store.ui.showToast('URL required for SSE/HTTP', 'error')
        return
      }
      if (this.editServer.transport_type === 'stdio' && !this.editServer.command) {
        this.$store.ui.showToast('Command required for stdio', 'error')
        return
      }

      await api.delete(`/api/mcp/servers/${this.editServer.originalName}`)
      await api.post('/api/mcp/servers', {
        name: this.editServer.name, transport_type: this.editServer.transport_type,
        command: this.editServer.command, args, env,
        url: this.editServer.url || null
      })
      if (this.editServer.enabled) {
        await api.post(`/api/mcp/servers/${this.editServer.name}/reconnect`)
      } else {
        await api.post(`/api/mcp/servers/${this.editServer.name}/toggle`, { enabled: false })
      }

      await this.loadMCPServers()
      this.closeEdit()
      this.$store.ui.showToast('Server updated!', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error: ' + e.message, 'error')
    }
  },

  async removeServer(name) {
    if (!confirm(`Remove "${name}"?`)) return
    try {
      await api.delete(`/api/mcp/servers/${name}`)
      await this.loadMCPServers()
      this.$store.ui.showToast('Server removed', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error: ' + e.message, 'error')
    }
  },

  async reconnectServer(name) {
    try {
      await api.post(`/api/mcp/servers/${name}/reconnect`)
      await this.loadMCPServers()
      this.$store.ui.showToast('Reconnected', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error: ' + e.message, 'error')
    }
  },

  async refreshTools(name) {
    try {
      await api.post(`/api/mcp/servers/${name}/refresh`)
      await this.loadMCPServers()
      this.$store.ui.showToast('Tools refreshed', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error: ' + e.message, 'error')
    }
  },

  toggleServer(name, enabled) {
    const server = this.mcpServers.find(s => s.name === name)
    if (server) server.enabled = enabled
  }
})

// Export factory for registration in main.js
export { settings }

/**
 * Settings Component — Unified settings modal with MCP server management
 */
import { api } from '../utils.js'

const settings = () => {
  console.log('[settings] Factory function called, creating component')
  return {
  open: false,
  activeTab: 'general',
  settings: {},
  mcpServers: [],
  mcpTools: [],
  newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', headers: '{}', timeout: 60 },
  editingServer: false,
  editServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', headers: '{}', enabled: true, originalName: '', timeout: 60 },
  backupStatus: null,
  backupFiles: [],
  backupRunning: false,

  async init() {
    console.log('[settings] init() called, loading settings and MCP servers')
    await this.loadSettings()
    await this.loadMCPServers()
    await this.loadBackupStatus()
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
      // Restart backup scheduler after settings change
      try { await api.post('/api/backup/scheduler/restart') } catch (e) { /* ignore */ }
      await this.loadBackupStatus()
      this.$store.ui.showToast('Settings saved!', 'success')
    } catch (e) {
      this.$store.ui.showToast('Error saving: ' + e.message, 'error')
    }
  },

  closeModal() {
    this.open = false
    this.$store.ui.closeSettings()
  },

  onTtsEngineChange() {
    const settings = this.$store.ui.settingsData
    if (settings.tts_engine === 'kokoro') {
      settings.tts_voice = 'af_bella'
      settings.kokoro_lang = 'a'
      settings.kokoro_device = settings.kokoro_device || 'cpu'
    } else if (settings.tts_engine === 'edge-tts') {
      settings.tts_voice = 'en-IN-NeerjaNeural'
    }
  },

  // ─── Backup ──────────────────────────────────────────
  async loadBackupStatus() {
    try {
      const data = await api.get('/api/backup/status')
      this.backupStatus = data
      this.backupFiles = data.backup_files || []
    } catch (e) { console.error('[settings] Backup status error:', e) }
  },
  async backupNow() {
    if (this.backupRunning) return
    this.backupRunning = true
    try {
      const result = await api.post('/api/backup/run')
      await this.loadBackupStatus()
      this.$store.ui.showToast('Backup completed!', 'success')
    } catch (e) {
      this.$store.ui.showToast('Backup error: ' + e.message, 'error')
    } finally {
      this.backupRunning = false
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
        disabled_tools: s.disabled_tools || [],
        toolsExpanded: false,
        enabled: s.enabled !== false,
        error: s.error || null
      }))
      this.$store.ui.mcpServers = this.mcpServers
      this.$store.ui.mcpTools = toolsData.tools || []
    } catch (e) { console.error('[settings] MCP:', e) }
  },

  async addMCPServer() {
    try {
      console.log('[settings] addMCPServer called', { newServer: this.newServer })
      
      let args = []
      let env = {}
      
      // Parse args (or use empty array)
      const argsStr = (this.newServer.args || '').trim()
      if (argsStr && argsStr !== '[]') {
        try {
          args = JSON.parse(argsStr)
          if (!Array.isArray(args)) args = []
        } catch (parseErr) {
          console.error('[settings] Failed to parse args:', { value: argsStr, error: parseErr.message })
          this.$store.ui.showToast(`Invalid JSON in Args field: ${parseErr.message}. Expected: ["arg1", "arg2"]`, 'error')
          return
        }
      }
      
      // Parse env (or use empty object)
      const envStr = (this.newServer.env || '').trim()
      if (envStr && envStr !== '{}') {
        try {
          env = JSON.parse(envStr)
          if (typeof env !== 'object' || Array.isArray(env)) env = {}
        } catch (parseErr) {
          console.error('[settings] Failed to parse env:', { value: envStr, error: parseErr.message })
          this.$store.ui.showToast(`Invalid JSON in Env field: ${parseErr.message}. Expected: {"KEY": "value"}`, 'error')
          return
        }
      }

      // Parse headers (or use empty object)
      let headers = {}
      const headersStr = (this.newServer.headers || '').trim()
      if (headersStr && headersStr !== '{}') {
        try {
          headers = JSON.parse(headersStr)
          if (typeof headers !== 'object' || Array.isArray(headers)) headers = {}
        } catch (parseErr) {
          console.error('[settings] Failed to parse headers:', { value: headersStr, error: parseErr.message })
          this.$store.ui.showToast(`Invalid JSON in Headers field: ${parseErr.message}. Expected: {"Authorization": "Bearer token"}`, 'error')
          return
        }
      }

      if (!this.newServer.name?.trim()) {
        this.$store.ui.showToast('Name is required for MCP server', 'error')
        return
      }
      if (this.newServer.transport_type !== 'stdio' && !this.newServer.url) {
        this.$store.ui.showToast('URL required for SSE/HTTP', 'error')
        return
      }
      if (this.newServer.transport_type === 'stdio' && !this.newServer.command) {
        this.$store.ui.showToast('Command required for stdio', 'error')
        return
      }

      const response = await api.post('/api/mcp/servers', {
        name: this.newServer.name,
        transport_type: this.newServer.transport_type,
        command: this.newServer.command,
        args, env,
        url: this.newServer.url || null,
        headers,
        timeout: this.newServer.timeout || 60
      })

      await this.loadMCPServers()
      this.newServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', headers: '{}', timeout: 60 }
      if (response.connected === false) {
        this.$store.ui.showToast(response.error || response.message || 'Server added but connection failed', 'warning')
      } else {
        this.$store.ui.showToast('Server added!', 'success')
      }
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
      headers: server.headers ? JSON.stringify(server.headers) : '{}',
      enabled: server.enabled !== false,
      originalName: server.name,
      timeout: server.timeout || 60
    }
    this.editingServer = true
  },

  closeEdit() {
    this.editingServer = false
    this.editServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', headers: '{}', enabled: true, originalName: '', timeout: 60 }
  },

  async saveEdit() {
    try {
      const args = JSON.parse(this.editServer.args)
      const env = this.editServer.env?.trim() ? JSON.parse(this.editServer.env) : {}
      let headers = {}
      const headersStr = (this.editServer.headers || '').trim()
      if (headersStr && headersStr !== '{}') {
        headers = JSON.parse(headersStr)
        if (typeof headers !== 'object' || Array.isArray(headers)) headers = {}
      }

      if (!this.editServer.name?.trim()) {
        this.$store.ui.showToast('Name is required for MCP server', 'error')
        return
      }
      if (this.editServer.transport_type !== 'stdio' && !this.editServer.url) {
        this.$store.ui.showToast('URL required for SSE/HTTP', 'error')
        return
      }
      if (this.editServer.transport_type === 'stdio' && !this.editServer.command) {
        this.$store.ui.showToast('Command required for stdio', 'error')
        return
      }

      const response = await api.put(`/api/mcp/servers/${this.editServer.originalName}`, {
        name: this.editServer.name, transport_type: this.editServer.transport_type,
        command: this.editServer.command, args, env,
        url: this.editServer.url || null,
        headers,
        timeout: this.editServer.timeout || 60
      })

      await this.loadMCPServers()
      this.closeEdit()
      if (response.connected === false) {
        this.$store.ui.showToast(response.error || response.message || 'Server updated but connection failed', 'warning')
      } else {
        this.$store.ui.showToast('Server updated!', 'success')
      }
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
  },

  async toggleTool(serverName, toolName, disabled) {
    try {
      await api.put(`/api/mcp/servers/${encodeURIComponent(serverName)}/tools/toggle`, {
        tool_name: toolName,
        disabled
      })
      // Sync the local state reactively
      const server = this.mcpServers.find(s => s.name === serverName)
      if (server) {
        if (disabled) {
          if (!server.disabled_tools.includes(toolName)) server.disabled_tools.push(toolName)
        } else {
          server.disabled_tools = server.disabled_tools.filter(t => t !== toolName)
        }
      }
    } catch (e) { console.error('[settings] toggleTool error:', e) }
  }
  }
}

// Export factory for registration in main.js
export { settings }

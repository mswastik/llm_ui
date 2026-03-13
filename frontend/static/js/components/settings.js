/**
 * Settings Component - Application settings and MCP server management
 */
import { api } from '../utils.js'

// Define the component
const settingsComponent = {
  // Local state
  activeTab: 'general',
  settings: {},
  mcpServers: [],
  mcpTools: [],
  newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}' },
  editingServer: false,
  editServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', enabled: true, originalName: '' },

  // Initialization
  async init() {
    await this.loadSettings()
    await this.loadMCPServers()
  },

  // Settings
  async loadSettings() {
    try {
      const data = await api.get('/api/settings')
      this.settings = data
    } catch (error) {
      console.error('[SETTINGS] Error loading settings:', error)
    }
  },

  async saveSettings() {
    try {
      console.log('[SETTINGS] Saving settings:', this.settings)
      const data = await api.put('/api/settings', this.settings)
      console.log('[SETTINGS] Settings saved:', data)
      this.settings = data
      // Show success toast using Alpine store
      const toastStore = window.Alpine?.store('chat') || { toast: {} }
      window.Alpine.store('chat', {
        ...toastStore,
        toast: { show: true, message: 'Settings saved!', type: 'success' }
      })
      setTimeout(() => {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: false, message: '', type: 'success' }
        })
      }, 3000)
    } catch (error) {
      console.error('[SETTINGS] Error saving settings:', error)
      const toastStore = window.Alpine?.store('chat') || { toast: {} }
      window.Alpine.store('chat', {
        ...toastStore,
        toast: { show: true, message: 'Error saving settings: ' + error.message, type: 'error' }
      })
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
      this.mcpServers = data.servers || []
    } catch (error) {
      console.error('[SETTINGS] Error loading MCP servers:', error)
    }
  },

  async loadMCPTools() {
    try {
      const data = await api.get('/api/mcp/tools')
      this.mcpTools = data.tools || []
    } catch (error) {
      console.error('[SETTINGS] Error loading MCP tools:', error)
    }
  },

  async addMCPServer() {
    try {
      let args = []
      let env = {}
      try {
        args = JSON.parse(this.newServer.args)
      } catch {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'Invalid JSON for args', type: 'error' }
        })
        return
      }
      try {
        if (this.newServer.env && this.newServer.env.trim()) {
          env = JSON.parse(this.newServer.env)
        }
      } catch {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'Invalid JSON for env', type: 'error' }
        })
        return
      }

      if (this.newServer.transport_type !== 'stdio' && !this.newServer.url) {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'URL required for SSE/HTTP', type: 'error' }
        })
        return
      }

      if (this.newServer.transport_type === 'stdio' && !this.newServer.command) {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'Command required for stdio', type: 'error' }
        })
        return
      }

      await api.post('/api/mcp/servers', {
        name: this.newServer.name,
        transport_type: this.newServer.transport_type,
        command: this.newServer.command,
        args: args,
        env: env,
        url: this.newServer.url || null
      })

      await this.loadMCPServers()
      await this.loadMCPTools()

      this.newServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}' }
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Server added!', type: 'success' }
      })
    } catch (error) {
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Error adding server', type: 'error' }
      })
    }
  },

  openEditModal(server) {
    this.editServer = {
      name: server.name,
      transport_type: server.transport_type,
      command: server.command || '',
      args: JSON.stringify(server.args || []),
      url: server.url || '',
      env: server.env ? JSON.stringify(server.env) : '{}',
      enabled: server.enabled !== false,
      originalName: server.name
    }
    this.editingServer = true
  },

  closeEditModal() {
    this.editingServer = false
    this.editServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '', env: '{}', enabled: true, originalName: '' }
  },

  async saveEditServer() {
    try {
      let args = []
      let env = {}
      try {
        args = JSON.parse(this.editServer.args)
      } catch {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'Invalid JSON for args', type: 'error' }
        })
        return
      }
      try {
        if (this.editServer.env && this.editServer.env.trim()) {
          env = JSON.parse(this.editServer.env)
        }
      } catch {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'Invalid JSON for env', type: 'error' }
        })
        return
      }

      if (this.editServer.transport_type !== 'stdio' && !this.editServer.url) {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'URL required for SSE/HTTP', type: 'error' }
        })
        return
      }

      if (this.editServer.transport_type === 'stdio' && !this.editServer.command) {
        window.Alpine.store('chat', {
          ...window.Alpine.store('chat'),
          toast: { show: true, message: 'Command required for stdio', type: 'error' }
        })
        return
      }

      // First remove the old server
      await api.delete(`/api/mcp/servers/${this.editServer.originalName}`)
      
      // Then add the updated server
      await api.post('/api/mcp/servers', {
        name: this.editServer.name,
        transport_type: this.editServer.transport_type,
        command: this.editServer.command,
        args: args,
        env: env,
        url: this.editServer.url || null
      })

      // If server should be enabled, reconnect
      if (this.editServer.enabled) {
        await api.post(`/api/mcp/servers/${this.editServer.name}/reconnect`)
      } else {
        // Disable the server
        await api.post(`/api/mcp/servers/${this.editServer.name}/toggle`, { enabled: false })
      }

      await this.loadMCPServers()
      await this.loadMCPTools()

      this.closeEditModal()
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Server updated!', type: 'success' }
      })
    } catch (error) {
      console.error('Error updating server:', error)
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Error updating server: ' + error.message, type: 'error' }
      })
    }
  },

  async removeMCPServer(serverName) {
    if (!confirm(`Remove "${serverName}"?`)) return
    try {
      await api.delete(`/api/mcp/servers/${serverName}`)
      await this.loadMCPServers()
      await this.loadMCPTools()
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Server removed', type: 'success' }
      })
    } catch (error) {
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Error removing server', type: 'error' }
      })
    }
  },

  async reconnectMCPServer(serverName) {
    try {
      await api.post(`/api/mcp/servers/${serverName}/reconnect`)
      await this.loadMCPServers()
      await this.loadMCPTools()
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Reconnected', type: 'success' }
      })
    } catch (error) {
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Error reconnecting', type: 'error' }
      })
    }
  },

  async refreshMCPServerTools(serverName) {
    try {
      await api.post(`/api/mcp/servers/${serverName}/refresh`)
      await this.loadMCPServers()
      await this.loadMCPTools()
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Tools refreshed', type: 'success' }
      })
    } catch (error) {
      window.Alpine.store('chat', {
        ...window.Alpine.store('chat'),
        toast: { show: true, message: 'Error refreshing', type: 'error' }
      })
    }
  }
}

// Register with Alpine.js when it becomes available
if (window.Alpine) {
  window.Alpine.data('settings', () => settingsComponent)
} else {
  // Wait for Alpine to load
  document.addEventListener('alpine:init', () => {
    window.Alpine.data('settings', () => settingsComponent)
  })
}

// Export for potential external use
export { settingsComponent }

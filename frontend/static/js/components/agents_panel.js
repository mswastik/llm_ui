/**
 * Agents Panel Component — AI agent management, now hosted inside the
 * Settings dialog (Agents tab). Persona + capability configuration.
 */
import { api } from '../utils.js'

// Custom tool catalogue (names must match backend exclusion list in main.py).
const TOOL_GROUPS = [
  { label: 'General', tools: [
    { name: 'generate_speech', label: 'Text-to-speech (generate_speech)' },
    { name: 'run_command', label: 'Terminal (run_command)' },
    { name: 'run_job', label: 'Jobs (run_job)' },
  ]},
  { label: 'Skills', tools: [
    { name: 'load_skill', label: 'Load skills' },
    { name: 'create_skill', label: 'Create skills' },
    { name: 'search_skills', label: 'Search skill registry' },
    { name: 'install_skill', label: 'Install registry skills' },
  ]},
  { label: 'Memory', tools: [
    { name: 'memory_write', label: 'Write memory' },
    { name: 'memory_read', label: 'Read memory' },
    { name: 'memory_search', label: 'Search memory' },
    { name: 'memory_delete', label: 'Delete memory' },
  ]},
  { label: 'Admin', tools: [
    { name: 'list_agents', label: 'List agents' },
    { name: 'create_agent', label: 'Create agents' },
    { name: 'delete_agent', label: 'Delete agents' },
    { name: 'list_mcp_servers', label: 'List MCP servers' },
    { name: 'add_mcp_server', label: 'Add MCP servers' },
    { name: 'remove_mcp_server', label: 'Remove MCP servers' },
    { name: 'list_providers', label: 'List providers' },
    { name: 'add_provider', label: 'Add providers' },
  ]},
]

const agentsPanel = () => ({
  agents: [],
  loading: true,
  showCreate: false,
  editing: null,
  formData: {
    name: '', description: '', system_prompt: '', model: '',
    provider_id: '', temperature: 0.7, top_k: 3, max_tokens: 4096,
    enable_rag: false, rag_similarity_threshold: 0.7,
    enabled_tools: [], enabled_mcp_servers: [], enabled_skills: []
  },
  availableModels: [],
  availableProviders: [],
  mcpServers: [],
  skills: [],
  toolGroups: TOOL_GROUPS,

  async init() {
    await Promise.all([
      this.loadAgents(), this.loadModels(), this.loadProviders(),
      this.loadMcpServers(), this.loadSkills()
    ])
  },

  async loadAgents() {
    this.loading = true
    try {
      const r = await api.get('/api/agents')
      this.agents = r.agents || []
    } catch (e) { this.agents = [] }
    this.loading = false
  },

  async loadMcpServers() {
    try {
      const r = await api.get('/api/mcp/servers')
      this.mcpServers = r.servers || []
    } catch (e) { this.mcpServers = [] }
  },

  async loadSkills() {
    try {
      const r = await api.get('/api/skills')
      this.skills = r.skills || []
    } catch (e) { this.skills = [] }
  },

  async loadModels() {
    try {
      const r = await api.get('/api/models')
      this.availableModels = r.models || []
      this.availableProviders = r.providers || []
    } catch (e) { this.availableModels = []; this.availableProviders = [] }
  },

  async loadProviders() {
    try {
      const r = await api.get('/api/providers')
      this.availableProviders = r.providers || []
    } catch (e) { this.availableProviders = [] }
  },

  modelsForProvider(providerId) {
    return this.availableModels.filter(m => (m.provider_id || '') === providerId)
  },

  providerName(id) {
    return this.availableProviders.find(p => p.id === id)?.name || ''
  },

  newAgent() {
    this.editing = null
    this.formData = {
      name: '', description: '', system_prompt: '', model: '',
      provider_id: this.availableProviders.find(p => p.is_default)?.id || this.availableProviders[0]?.id || '',
      temperature: 0.7, top_k: 3, max_tokens: 4096,
      enable_rag: false, rag_similarity_threshold: 0.7,
      enabled_tools: [], enabled_mcp_servers: [], enabled_skills: []
    }
    this.showCreate = true
  },

  editAgent(agent) {
    this.editing = agent
    this.formData = {
      name: agent.name || '',
      description: agent.description || '',
      system_prompt: agent.system_prompt || '',
      model: agent.model || '',
      provider_id: agent.provider_id || '',
      temperature: agent.temperature ?? 0.7,
      top_k: agent.top_k ?? 3,
      max_tokens: agent.max_tokens ?? 4096,
      enable_rag: !!agent.enable_rag,
      rag_similarity_threshold: agent.rag_similarity_threshold ?? 0.7,
      enabled_tools: agent.enabled_tools || [],
      enabled_mcp_servers: agent.enabled_mcp_servers || [],
      enabled_skills: agent.enabled_skills || []
    }
    this.showCreate = true
  },

  async saveAgent() {
    try {
      const url = this.editing ? '/api/agents/' + this.editing.id : '/api/agents'
      const method = this.editing ? 'PUT' : 'POST'
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.formData)
      })
      if (res.ok) {
        const d = await res.json()
        if (this.editing) {
          const idx = this.agents.findIndex(a => a.id === this.editing.id)
          if (idx !== -1) this.agents[idx] = d.agent
        } else {
          this.agents.unshift(d.agent)
        }
        this.showCreate = false
        this.$store.ui.showToast(this.editing ? 'Agent updated' : 'Agent created', 'success')
      } else {
        this.$store.ui.showToast('Failed to save agent', 'error')
      }
    } catch (e) { this.$store.ui.showToast('Failed to save agent', 'error') }
  },

  async deleteAgent(id) {
    if (!confirm('Delete this agent?')) return
    try {
      await api.delete('/api/agents/' + id)
      this.agents = this.agents.filter(a => a.id !== id)
      this.$store.ui.showToast('Agent deleted', 'success')
    } catch (e) { this.$store.ui.showToast('Failed to delete', 'error') }
  }
})

export { agentsPanel }

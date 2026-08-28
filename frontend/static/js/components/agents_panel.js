/**
 * Agents Panel Component — AI agent management, now hosted inside the
 * Settings dialog (Agents tab). Persona + capability configuration.
 *
 * The custom-tool list comes from GET /api/tools (the executor's own registry)
 * rather than a hand-maintained copy here, so the picker can never fall behind
 * the tools an agent is actually allowed to call.
 */
import { api, markdownUtils } from '../utils.js'

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
  toolCatalogue: [],
  modelSearch: '',

  async init() {
    await Promise.all([
      this.loadAgents(), this.loadModels(), this.loadProviders(),
      this.loadMcpServers(), this.loadSkills(), this.loadToolCatalogue()
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

  // Authoritative custom-tool list — grouped server-side by owning module.
  async loadToolCatalogue() {
    try {
      const r = await api.get('/api/tools')
      this.toolCatalogue = r.tools || []
    } catch (e) { this.toolCatalogue = [] }
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

  // ─── Capability picker options ───────────────────────────
  // Shape consumed by capabilityPicker(): {value, label, group, description, badge}
  get mcpOptions() {
    return this.mcpServers.map(s => ({
      value: s.name,
      label: s.name,
      group: s.enabled ? 'Enabled' : 'Disabled',
      description: `${(s.tools || []).length} tool(s)`,
      badge: s.enabled ? '' : 'disabled',
    }))
  },

  get toolOptions() {
    return this.toolCatalogue.map(t => ({
      value: t.name,
      label: t.name,
      group: t.group || 'Other',
      description: (t.description || '').split('\n')[0].slice(0, 140),
      badge: '',
    }))
  },

  get skillOptions() {
    return this.skills.map(s => ({
      value: s.name,
      label: s.name,
      group: s.draft ? 'Drafts' : 'Installed',
      description: s.description || '',
      badge: s.draft ? 'draft' : '',
    }))
  },

  renderMarkdown: (t) => markdownUtils.render(t),

  newAgent() {
    this.editing = null
    this.modelSearch = ''
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
    this.modelSearch = agent.model || ''
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
      const text = await res.text()
      let d
      try { d = text ? JSON.parse(text) : {} } catch { d = { detail: text } }
      if (res.ok) {
        const agentData = d.agent || d
        const wasEditing = !!this.editing
        if (wasEditing) {
          const idx = this.agents.findIndex(a => a.id === this.editing.id)
          if (idx !== -1) this.agents[idx] = agentData
        } else {
          this.agents.unshift(agentData)
        }
        this.showCreate = false
        this.editing = null
        this.$store.ui.showToast(wasEditing ? 'Agent updated' : 'Agent created', 'success')
        // refresh chat's agent list
        window.dispatchEvent(new CustomEvent('agents-updated'))
      } else {
        const msg = d.detail || d.message || text || 'Failed to save agent'
        console.error('[agents] save failed', res.status, msg)
        this.$store.ui.showToast(msg, 'error')
      }
    } catch (e) {
      console.error('[agents] save exception', e)
      this.$store.ui.showToast('Failed to save agent: ' + (e.message || e), 'error')
    }
  },

  async deleteAgent(id) {
    if (!confirm('Delete this agent?')) return
    try {
      await api.delete('/api/agents/' + id)
      this.agents = this.agents.filter(a => a.id !== id)
      this.$store.ui.showToast('Agent deleted', 'success')
      window.dispatchEvent(new CustomEvent('agents-updated'))
    } catch (e) { this.$store.ui.showToast('Failed to delete', 'error') }
  }
})

export { agentsPanel }

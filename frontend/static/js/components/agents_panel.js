/**
 * Agents Panel Component — AI agent management, now hosted inside the
 * Settings dialog (Agents tab). Persona + capability configuration.
 */
import { api } from '../utils.js'

const agentsPanel = () => ({
  agents: [],
  loading: true,
  showCreate: false,
  editing: null,
  formData: {
    name: '', description: '', system_prompt: '', model: '',
    provider_id: '', temperature: 0.7, top_k: 3, max_tokens: 4096,
    enable_rag: false, rag_similarity_threshold: 0.7
  },
  availableModels: [],
  availableProviders: [],

  async init() {
    await Promise.all([this.loadAgents(), this.loadModels(), this.loadProviders()])
  },

  async loadAgents() {
    this.loading = true
    try {
      const r = await api.get('/api/agents')
      this.agents = r.agents || []
    } catch (e) { this.agents = [] }
    this.loading = false
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
      enable_rag: false, rag_similarity_threshold: 0.7
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
      rag_similarity_threshold: agent.rag_similarity_threshold ?? 0.7
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

/**
 * Skills Panel Component (Settings → Skills tab) (agent platform Phase 3)
 * List / view / create / edit / delete agent skills; accept or reject
 * self-improvement drafts (Phase 4).
 */
import { api } from '../utils.js'

const skillsPanel = () => ({
  skills: [],
  showForm: false,
  editingName: null,
  form: { name: '', description: '', instructions: '' },
  viewingName: null,
  viewContent: '',
  // Registry browser (Phase 4.5)
  view: 'installed', // 'installed' | 'registry'
  registryQuery: '',
  registryResults: [],
  registryLoading: false,
  registryError: '',



  async loadSkills() {
    try {
      const data = await api.get('/api/skills?include_drafts=true')
      this.skills = data.skills || []
    } catch (e) { console.error('[skills] load error:', e) }
  },

  setView(v) {
    this.view = v
    if (v === 'registry' && this.registryResults.length === 0) {
      this.searchRegistry()
    }
  },

  async searchRegistry() {
    const q = (this.registryQuery || '').trim()
    this.registryLoading = true
    this.registryError = ''
    try {
      // Blank query = most popular skills (backend merges broad queries).
      const data = await api.get(`/api/skills/registry?query=${encodeURIComponent(q)}&limit=25`)
      this.registryResults = data.skills || []
      this.registryError = data.error || ''
    } catch (e) {
      this.registryError = e.message
      this.registryResults = []
    } finally {
      this.registryLoading = false
    }
  },

  async installFromRegistry(skill) {
    try {
      const data = await api.post('/api/skills/install', { id: skill.id })
      this.$store.ui.showToast(`Installed '${data.skill.name}' (${data.skill.files.length} files)`, 'success')
      this.view = 'installed'
      await this.loadSkills()
    } catch (e) {
      this.$store.ui.showToast(e.message, 'error')
    }
  },

  installedNames() {
    const names = new Set(this.skills.map(s => s.name))
    return names
  },

  startCreate() {
    this.editingName = null
    this.form = { name: '', description: '', instructions: '' }
    this.showForm = true
  },

  startEdit(skill) {
    this.editingName = skill.name
    this.form = { name: skill.name, description: skill.description || '', instructions: skill.body || '' }
    this.showForm = true
  },

  cancelForm() {
    this.showForm = false
    this.editingName = null
  },

  async saveSkill() {
    const name = (this.form.name || '').trim()
    const description = (this.form.description || '').trim()
    const instructions = (this.form.instructions || '').trim()
    if (!name || !instructions) return
    try {
      if (this.editingName) {
        await api.put(`/api/skills/${encodeURIComponent(this.editingName)}`, { description, instructions })
      } else {
        await api.post('/api/skills', { name, description, instructions })
      }
      this.showForm = false
      this.editingName = null
      await this.loadSkills()
      this.$store.ui.showToast('Skill saved', 'success')
    } catch (e) { this.$store.ui.showToast('Error: ' + e.message, 'error') }
  },

  async viewSkill(skill) {
    if (this.viewingName === skill.name && !skill.draft) {
      this.viewingName = null
      return
    }
    this.viewingName = skill.name
    try {
      const data = await api.get(`/api/skills/${encodeURIComponent(skill.name)}`)
      this.viewContent = data.skill?.body || '(empty)'
    } catch (e) { this.viewContent = '(unavailable)' }
  },

  async deleteSkill(name, draft) {
    if (!confirm(`Delete ${draft ? 'draft ' : ''}skill '${name}'?`)) return
    try {
      const url = draft
        ? `/api/skills/drafts/${encodeURIComponent(name)}`
        : `/api/skills/${encodeURIComponent(name)}`
      await api.delete(url)
      if (this.viewingName === name) this.viewingName = null
      await this.loadSkills()
    } catch (e) { this.$store.ui.showToast('Error: ' + e.message, 'error') }
  },

  async acceptDraft(name) {
    try {
      await api.post(`/api/skills/drafts/${encodeURIComponent(name)}/accept`)
      await this.loadSkills()
      this.$store.ui.showToast('Draft accepted — skill is now live', 'success')
    } catch (e) { this.$store.ui.showToast('Error: ' + e.message, 'error') }
  },

  async rejectDraft(name) {
    try {
      await api.delete(`/api/skills/drafts/${encodeURIComponent(name)}`)
      if (this.viewingName === name) this.viewingName = null
      await this.loadSkills()
    } catch (e) { this.$store.ui.showToast('Error: ' + e.message, 'error') }
  }
})

export { skillsPanel }

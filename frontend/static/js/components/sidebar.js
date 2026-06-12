/**
 * Sidebar Component — Conversation management with search, agent filters, tags
 */
import { api } from '../utils.js'

const sidebar = () => ({
  conversations: [],
  currentConversationId: null,
  currentConversationTitle: 'New Chat',
  searchQuery: '',
  isResizing: false,
  minWidth: 68,
  maxWidth: 450,
  availableAgents: [],

  // ─── Computed ─────────────────────────────────────────
  get activeAgentFilters() { return this.$store.ui.activeAgentFilters },
  get activeTagFilter() { return this.$store.ui.activeTagFilter },
  get editingTagsForConversation() { return this.$store.ui.editingTagsForConversation },

  get filteredConversations() {
    let list = this.conversations

    // Apply agent filter (multi-select: OR logic across selected filters)
    if (this.activeAgentFilters.length > 0) {
      list = list.filter(c => {
        return this.activeAgentFilters.some(f => {
          if (f === 'default') return c.agent_id === null
          return c.agent_id === f
        })
      })
    }

    // Apply tag filter
    if (this.activeTagFilter) {
      list = list.filter(c => c.tags?.includes(this.activeTagFilter))
    }

    // Apply search
    if (this.searchQuery?.trim()) {
      const q = this.searchQuery.toLowerCase()
      list = list.filter(c => c.title?.toLowerCase().includes(q))
    }

    return list
  },

  get allTags() {
    const tagSet = new Set()
    this.conversations.forEach(c => {
      if (c.tags?.length) c.tags.forEach(t => tagSet.add(t))
    })
    return [...tagSet].sort()
  },

  async init() {
    this._initResize()
    this.currentConversationId = this.$store.chat.currentConversationId
    this.currentConversationTitle = this.$store.chat.currentConversationTitle
    this.conversations = this.$store.chat.conversations || []
    this.availableAgents = this.$store.chat.availableAgents || []
    await this.loadConversations()

    // Listen for load-conversation events (e.g., from Notes modal)
    window.addEventListener('load-conversation', (e) => {
      if (e.detail?.id) {
        this.loadConversation(e.detail.id)
      }
    })
  },

  // ─── Conversations ────────────────────────────────────
  async loadConversations() {
    try {
      const data = await api.get('/api/conversations')
      this.conversations = data.conversations || []
      this.$store.chat.conversations = this.conversations
    } catch (e) { console.error('[sidebar] Error:', e) }
  },

  async createNewConversation() {
    try {
      const agentId = this.$store.chat.selectedAgentId || null
      const data = await api.post('/api/conversations', { title: 'New Chat', agent_id: agentId })
      this.$store.chat.addConversation(data.conversation)
      this.conversations = this.$store.chat.conversations
      await this.loadConversation(data.conversation.id)
    } catch (e) {
      this.$store.ui.showToast('Failed to create conversation', 'error')
    }
  },

  async loadConversation(id) {
    // Check for active streaming
    const active = this.$store.chat.activeStreaming
    if (active.isStreaming && active.conversationId === id) {
      this.$store.chat.isLoading = true
      this.$store.chat.currentConversationId = id
      this.$store.chat.currentConversationTitle = active.conversationTitle || 'Streaming...'
      this.$store.chat.messages = [...active.messages]
      this.currentConversationId = id
      this.currentConversationTitle = this.$store.chat.currentConversationTitle
      return
    }

    this.$store.chat.isLoading = false
    try {
      const data = await api.get(`/api/conversations/${id}`)
      this.$store.chat.currentConversationId = id
      this.$store.chat.currentConversationTitle = data.conversation.title
      this.$store.chat.messages = this.normalizeMessages(data.messages || [])
      this.currentConversationId = id
      this.currentConversationTitle = data.conversation.title

      const convAgentId = data.conversation.agent_id || null
      this.$store.chat.setAgent(convAgentId)
      this.$store.chat.applyAgentConfig()
      window.dispatchEvent(new CustomEvent('sync-agent', { detail: { agentId: convAgentId } }))

      // Notify chat component to focus the input
      window.dispatchEvent(new CustomEvent('conversation-loaded'))
    } catch (e) {
      this.$store.ui.showToast('Failed to load conversation', 'error')
    }
  },

  async deleteConversation(id, e) {
    e?.stopPropagation()
    if (!confirm('Delete this conversation?')) return
    try {
      await api.delete(`/api/conversations/${id}`)
      this.$store.chat.removeConversation(id)
      this.conversations = this.$store.chat.conversations
      if (id === this.currentConversationId) {
        if (this.conversations.length > 0) {
          await this.loadConversation(this.conversations[0].id)
        } else {
          await this.createNewConversation()
        }
      }
    } catch (e) {
      this.$store.ui.showToast('Failed to delete', 'error')
    }
  },

  // ─── Tag Management ───────────────────────────────────
  getConvTags(convId) {
    const conv = this.conversations.find(c => c.id === convId)
    return conv?.tags || []
  },

  async addTag(convId, tag) {
    if (!tag?.trim()) return
    const conv = this.conversations.find(c => c.id === convId)
    if (!conv) return
    const currentTags = conv.tags || []
    if (currentTags.includes(tag.trim())) return
    const newTags = [...currentTags, tag.trim()]
    try {
      await api.put(`/api/conversations/${convId}/tags`, { tags: newTags })
      conv.tags = newTags
      this.$store.ui.tagInput = ''
      this.$store.ui.showToast('Tag added', 'success')
    } catch (e) {
      this.$store.ui.showToast('Failed to add tag', 'error')
    }
  },

  async removeTag(convId, tag) {
    const conv = this.conversations.find(c => c.id === convId)
    if (!conv) return
    const newTags = (conv.tags || []).filter(t => t !== tag)
    try {
      await api.put(`/api/conversations/${convId}/tags`, { tags: newTags })
      conv.tags = newTags
    } catch (e) {
      this.$store.ui.showToast('Failed to remove tag', 'error')
    }
  },

  // ─── Resize ───────────────────────────────────────────
  startResize(e) {
    this.isResizing = true
    e.preventDefault()
    e.stopPropagation()
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('mousemove', this._mousemoveBound, { passive: false })
    document.addEventListener('mouseup', this._mouseupBound)
  },

  handleMouseMove(e) {
    if (!this.isResizing) return
    const w = e.clientX
    if (w >= this.minWidth && w <= this.maxWidth) {
      this.$store.ui.setSidebarWidth(w)
    }
  },

  handleMouseUp() {
    if (this.isResizing) {
      this.isResizing = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', this._mousemoveBound)
      document.removeEventListener('mouseup', this._mouseupBound)
    }
  },

  _initResize() {
    this._mousemoveBound = (e) => this.handleMouseMove(e)
    this._mouseupBound = () => this.handleMouseUp()
  },

  // ─── Helpers ──────────────────────────────────────────
  normalizeMessages(messages) {
    // Deduplicate by version_group — keep only the latest version of each group.
    // This ensures consistent visual layout during live chat and after refresh.
    const deduped = []
    const seen = {}
    for (const raw of messages) {
      const vg = raw.version_group
      if (!vg) {
        deduped.push(raw)
      } else if (vg in seen) {
        const idx = seen[vg]
        if ((raw.version || 1) > (deduped[idx].version || 1)) {
          deduped[idx] = raw
        }
      } else {
        seen[vg] = deduped.length
        deduped.push(raw)
      }
    }

    return deduped.map(msg => {
      // Extract files from metadata if not already on the message
      const files = msg.files || (msg.metadata?.files) || []
      const blocks = msg.blocks || (msg.metadata?.blocks)
      if (blocks?.length) {
        return { ...msg, files, blocks, tool_calls: msg.tool_calls || [] }
      }
      const blocksArr = []
      if (msg.thinking) {
        blocksArr.push({ type: 'thinking', content: msg.thinking })
      }
      if (msg.tool_calls?.length) {
        msg.tool_calls.forEach(tc => {
          blocksArr.push({
            type: 'tool_call',
            name: tc.name || 'tool',
            arguments: tc.arguments || tc.input || {},
            status: tc.status || 'completed',
            progress: tc.progress || 0,
            result: tc.result || null,
            sources: tc.sources || tc.result?.sources || [],
            search_steps: tc.search_steps || tc.result?.search_steps || [],
            search_terms: tc.search_terms || tc.result?.search_terms || []
          })
        })
      }
      if (msg.content) {
        blocksArr.push({ type: 'content', content: msg.content })
      }
      return { ...msg, files, blocks: blocksArr.length ? blocksArr : null, tool_calls: msg.tool_calls || [] }
    })
  },

  getAgentName(agentId) {
    if (!agentId) return null
    const agents = this.$store.chat.availableAgents || []
    const agent = agents.find(a => a.id === agentId)
    return agent?.name || null
  },

  formatDate: (s) => api ? new Date(s).toLocaleDateString() : ''
})

// Export factory for registration in main.js
export { sidebar }

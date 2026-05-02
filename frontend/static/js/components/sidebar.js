/**
 * Sidebar Component — Conversation management with search, animations
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

  get filteredConversations() {
    if (!this.searchQuery?.trim()) return this.conversations
    const q = this.searchQuery.toLowerCase()
    return this.conversations.filter(c => c.title?.toLowerCase().includes(q))
  },

  async init() {
    this._initResize()
    this.currentConversationId = this.$store.chat.currentConversationId
    this.currentConversationTitle = this.$store.chat.currentConversationTitle
    this.conversations = this.$store.chat.conversations || []
    await this.loadConversations()
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
      const data = await api.post('/api/conversations', { title: 'New Chat' })
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

  // ─── Resize ───────────────────────────────────────────
  startResize(e) {
    this.isResizing = true
    e.preventDefault()
    e.stopPropagation()
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    // Use { passive: false } so preventDefault() actually works
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

  // Bind handlers once so we can remove them later
  _initResize() {
    this._mousemoveBound = (e) => this.handleMouseMove(e)
    this._mouseupBound = () => this.handleMouseUp()
  },

  // ─── Helpers ──────────────────────────────────────────
  normalizeMessages(messages) {
    return messages.map(msg => {
      const blocks = msg.blocks || (msg.metadata?.blocks)
      if (blocks?.length) {
        return { ...msg, blocks, tool_calls: msg.tool_calls || [] }
      }
      // Backward compat: build blocks from old format
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
      return { ...msg, blocks: blocksArr.length ? blocksArr : null, tool_calls: msg.tool_calls || [] }
    })
  },

  formatDate: (s) => api ? new Date(s).toLocaleDateString() : ''
})

// Export factory for registration in main.js
export { sidebar }

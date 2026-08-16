/**
 * Utilities — API client, formatters, markdown, helpers
 */

// ─── API Client ───────────────────────────────────────────
export const api = {
  async get(endpoint) {
    const res = await fetch(endpoint)
    if (!res.ok) throw new Error('API Error')
    return res.json()
  },
  async post(endpoint, data) {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'API Error')
    }
    return res.json()
  },
  async put(endpoint, data) {
    const res = await fetch(endpoint, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!res.ok) throw new Error('API Error')
    return res.json()
  },
  async patch(endpoint, data) {
    const res = await fetch(endpoint, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!res.ok) throw new Error('API Error')
    return res.json()
  },
  async delete(endpoint) {
    const res = await fetch(endpoint, { method: 'DELETE' })
    if (!res.ok) throw new Error('API Error')
    return res.json()
  }
}

// ─── Formatters ───────────────────────────────────────────
export const formatters = {
  formatDate(isoString) {
    const date = new Date(isoString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
    return date.toLocaleDateString()
  },
  formatFileSize(bytes) {
    if (!bytes) return '0 B'
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  },
  stripMarkdown(text) {
    if (!text) return ''
    return text
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/`(.*?)`/g, '$1')
      .replace(/```[\s\S]*?```/g, '')
      .replace(/\[\d+\]/g, '')
      .trim()
  }
}

// ─── Markdown ─────────────────────────────────────────────
export const markdownUtils = {
  render(text) {
    if (!text) return ''
    return marked.parse(text)
  },
  renderWithCitations(text, sources = []) {
    if (!text) return ''
    let html = this.render(text)
    if (sources?.length > 0) {
      html = html.replace(/\[(\d+)\]/g, (match, num) => {
        const idx = parseInt(num) - 1
        if (idx >= 0 && idx < sources.length) {
          const s = sources[idx]
          return `<sup><a href="${s.url || '#'}" target="_blank" rel="noopener" class="citation-link" title="${s.title || ''}">${num}</a></sup>`
        }
        return match
      })
    }
    return html
  },
  extractSources(blocks) {
    if (!blocks?.length) return []
    const sources = []
    blocks.forEach(block => {
      if (block.type === 'tool_call' && block.sources?.length) {
        block.sources.forEach(s => {
          if (!sources.find(x => x.url === s.url)) {
            sources.push({ ...s, citationId: sources.length + 1 })
          }
        })
      }
    })
    return sources
  }
}

// ─── Helpers ──────────────────────────────────────────────
export const helpers = {
  copyToClipboard(text) {
    return navigator.clipboard.writeText(text).catch(() => false)
  },
  generateId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7) },
  formatToolResult(result) {
    if (!result) return ''
    if (typeof result === 'string') return result
    if (Array.isArray(result.content)) {
      const texts = result.content.filter(i => i.type === 'text').map(i => i.text)
      if (texts.length) return texts.join('\n')
    }
    if (typeof result.text === 'string') return result.text
    if (typeof result.content === 'string') return result.content
    return JSON.stringify(result, null, 2)
  },
  parseEscapes(text) {
    if (typeof text !== 'string') return text
    return text.replace(/\\n/g, '\n').replace(/\\r/g, '\r').replace(/\\t/g, '\t')
      .replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\\\/g, '\\')
  }
}

// Note: api, formatters, markdownUtils, helpers are already exported individually above

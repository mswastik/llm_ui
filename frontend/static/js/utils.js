/**
 * Utility Functions
 * Shared helpers for formatters, markdown, API, and common operations
 */

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
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  },

  stripMarkdown(text) {
    if (!text) return ''
    text = text.replace(/^#{1,6}\s+/gm, '')
    text = text.replace(/\*\*(.*?)\*\*/g, '$1')
    text = text.replace(/\*(.*?)\*/g, '$1')
    text = text.replace(/`(.*?)`/g, '$1')
    text = text.replace(/```[\s\S]*?```/g, '')
    text = text.replace(/\[\d+\]/g, '')
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    text = text.replace(/\n{3,}/g, '\n\n')
    return text.trim()
  }
}

export const markdownUtils = {
  render(text) {
    if (!text) return ''
    return marked.parse(text)
  },

  renderWithCitations(text, sources = []) {
    if (!text) return ''
    let html = this.render(text)
    if (sources && sources.length > 0) {
      html = html.replace(/\[(\d+)\]/g, (match, num) => {
        const index = parseInt(num) - 1
        if (index >= 0 && index < sources.length) {
          const source = sources[index]
          const url = source.url || '#'
          const title = source.title || 'Source'
          return `<sup><a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-link" title="${title}">[${num}]</a></sup>`
        }
        return match
      })
    }
    return html
  },

  getMessageSources(message) {
    if (!message?.tool_calls || message.tool_calls.length === 0) return []
    const allSources = []
    message.tool_calls.forEach(toolCall => {
      // Check for sources in multiple locations for backward compatibility
      const sources = toolCall.sources || toolCall.result?.sources || []
      if (sources && sources.length > 0) {
        sources.forEach(source => {
          // Only add unique sources by URL
          if (!allSources.some(s => s.url === source.url)) {
            // Add citation index to source for proper citation tracking
            const existingIndex = allSources.findIndex(s => s.url === source.url)
            if (existingIndex === -1) {
              allSources.push({
                ...source,
                citationId: allSources.length + 1
              })
            }
          }
        })
      }
    })
    // Assign sequential citation IDs
    return allSources.map((source, idx) => ({ ...source, citationId: idx + 1 }))
  },

  getMessageSourcesFromBlocks(blocks) {
    if (!blocks || blocks.length === 0) return []
    const allSources = []
    blocks.forEach(block => {
      // Only tool_call blocks have sources
      if (block.type === 'tool_call') {
        const sources = block.sources || block.result?.sources || []
        if (sources && sources.length > 0) {
          sources.forEach(source => {
            // Only add unique sources by URL
            if (!allSources.some(s => s.url === source.url)) {
              allSources.push({
                ...source,
                citationId: allSources.length + 1
              })
            }
          })
        }
      }
    })
    // Assign sequential citation IDs
    return allSources.map((source, idx) => ({ ...source, citationId: idx + 1 }))
  }
}

export const helpers = {
  scrollToBottom(container) {
    if (container) container.scrollTop = container.scrollHeight
  },

  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      const textArea = document.createElement('textarea')
      textArea.value = text
      textArea.style.position = 'fixed'
      textArea.style.left = '-999999px'
      document.body.appendChild(textArea)
      textArea.select()
      try {
        document.execCommand('copy')
        document.body.removeChild(textArea)
        return true
      } catch {
        document.body.removeChild(textArea)
        return false
      }
    }
  },

  generateId() { return Date.now() },

  toggleExpansion(state, key) { state[key] = !state[key]; return state[key] },

  isExpanded(state, key) { return state[key] === true },

  createExpansionKey(id, index) { return `${id}-${index}` },

  /**
   * Parse escape characters in a string to improve readability
   * Converts escape sequences like \n, \t, \" etc. to their actual characters
   */
  parseEscapeCharacters(text) {
    if (typeof text !== 'string') return text
    // Replace escape sequences - order matters, handle backslash last
    return text
      .replace(/\\n/g, '\n')  // Newline
      .replace(/\\r/g, '\r')  // Carriage return
      .replace(/\\t/g, '\t')  // Tab
      .replace(/\\"/g, '"')   // Double quote
      .replace(/\\'/g, "'")   // Single quote
      .replace(/\\\\/g, '\\') // Backslash (must be last)
  },

  /**
   * Format tool result for display, handling various response structures
   */
  formatToolResult(result) {
    if (!result) return ''
    
    // Handle string results directly
    if (typeof result === 'string') {
      return this.parseEscapeCharacters(result)
    }
    
    // Handle content array (MCP standard format)
    if (Array.isArray(result.content)) {
      const textContents = result.content
        .filter(item => item.type === 'text' && typeof item.text === 'string')
        .map(item => this.parseEscapeCharacters(item.text))
      if (textContents.length > 0) {
        return textContents.join('\n')
      }
    }
    
    // Handle direct text field
    if (typeof result.text === 'string') {
      return this.parseEscapeCharacters(result.text)
    }
    
    // Handle structured content
    if (result.structured_content) {
      return JSON.stringify(result.structured_content, null, 2)
    }
    
    // Fallback: stringify the whole object
    return JSON.stringify(result, null, 2)
  },

  /**
   * Extract text content from tool result for summary display
   * Returns plain text string from various result formats
   */
  extractToolContent(result) {
    if (!result) return ''
    
    // Handle string results directly
    if (typeof result === 'string') {
      return result
    }
    
    // Handle content array (MCP standard format)
    if (Array.isArray(result.content)) {
      const textContents = result.content
        .filter(item => item.type === 'text' && typeof item.text === 'string')
        .map(item => item.text)
      if (textContents.length > 0) {
        return textContents.join('\n')
      }
    }
    
    // Handle direct text field
    if (typeof result.text === 'string') {
      return result.text
    }
    
    // Handle content field (legacy format)
    if (typeof result.content === 'string') {
      return result.content
    }
    
    // Fallback: stringify
    return JSON.stringify(result)
  }
}

export const api = {
  async get(endpoint) {
    const response = await fetch(endpoint)
    if (!response.ok) throw new Error('API Error')
    return response.json()
  },

  async post(endpoint, data) {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(error.detail || 'API Error')
    }
    return response.json()
  },

  async put(endpoint, data) {
    const response = await fetch(endpoint, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) throw new Error('API Error')
    return response.json()
  },

  async delete(endpoint) {
    const response = await fetch(endpoint, { method: 'DELETE' })
    if (!response.ok) throw new Error('API Error')
    return response.json()
  }
}

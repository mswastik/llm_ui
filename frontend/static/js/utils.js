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

// Custom marked renderer: fenced code blocks get the app's themed
// .code-block / .code-header / .code-body treatment (background, border,
// language label + copy button) instead of an unstyled <pre><code>.
function buildCodeRenderer() {
  if (typeof marked === 'undefined' || !marked.Renderer) return null
  const renderer = new marked.Renderer()
  // marked >= v9 passes the code TOKEN object; older versions pass
  // (code, infostring, escaped). Handle both signatures.
  renderer.code = (code, infostring) => {
    let text = '', lang = ''
    if (code && typeof code === 'object') {
      text = code.text ?? ''
      lang = code.lang ?? ''
    } else {
      text = code ?? ''
      lang = infostring ?? ''
    }
    const langLabel = lang.trim().split(/\s+/)[0]
    const label = helpers.escapeHtml(langLabel || 'code')
    const body = helpers.escapeHtml(text)
    // HTML/SVG/XML blocks get a preview toggle that renders the code in a
    // sandboxed iframe. Also content-sniff: some models label SVG as "xml",
    // so any block whose text contains an <svg root is treated as previewable.
    const previewable = /^(html|htm|svg|xml)$/i.test(langLabel) || /<svg[\s>]/i.test(text)
    const previewBtn = previewable
      ? `<button type="button" class="code-copy-btn code-preview-btn" onclick="window.toggleHtmlPreview && window.toggleHtmlPreview(this)" title="Preview rendered output"><i class="ph ph-eye"></i></button>`
      : ''
    // The rendered-output iframe is created lazily by toggleHtmlPreview (only when
    // Preview is clicked); emitting one sandboxed iframe per HTML/SVG block made
    // long, code-heavy threads slow to open.
    return `<div class="code-block">` +
      `<div class="code-header">` +
      `<span>${label}</span>` +
      `<span class="flex items-center gap-1">${previewBtn}` +
      `<button type="button" class="code-copy-btn" onclick="window.copyCodeBlock && window.copyCodeBlock(this)" title="Copy code"><i class="ph ph-copy"></i></button>` +
      `</span>` +
      `</div>` +
      `<pre class="code-body"><code>${body}</code></pre>` +
      `</div>`
  }
  return renderer
}

// Global helper used by the code-block copy buttons (inline onclick).
if (typeof window !== 'undefined') {
  window.copyCodeBlock = async function (btn) {
    const pre = btn?.closest('.code-block')?.querySelector('.code-body')
    if (!pre) return
    await helpers.copyToClipboard(pre.textContent)
    const icon = btn.querySelector('.ph')
    if (icon) {
      icon.classList.remove('ph-copy')
      icon.classList.add('ph-check')
      setTimeout(() => {
        icon.classList.add('ph-copy')
        icon.classList.remove('ph-check')
      }, 1200)
    }
  }

  // Toggle the rendered-output iframe for html/svg code blocks. The iframe is
  // created on first use (not eagerly at markdown-render time) and fully sandboxed
  // (no scripts, no parent access). Scripts + inline event handlers are stripped
  // from the srcdoc first, so the sandbox can never run model code and the browser
  // no longer logs "Blocked script execution in about:blank".
  window.toggleHtmlPreview = function (btn) {
    const block = btn?.closest('.code-block')
    if (!block) return
    let iframe = block.querySelector('.code-html-preview')
    if (iframe && iframe.style.display !== 'none') {
      iframe.style.display = 'none'
      btn.classList.remove('code-preview-active')
      return
    }
    if (!iframe) {
      iframe = document.createElement('iframe')
      iframe.className = 'code-html-preview'
      iframe.setAttribute('sandbox', '')
      iframe.setAttribute('title', 'Rendered output')
      iframe.style.display = 'none'
      block.appendChild(iframe)
    }
    const raw = block.querySelector('.code-body')?.textContent || ''
    const html = raw
      .replace(/<script[\s\S]*?<\/script\s*>/gi, '')
      .replace(/<script[^>]*\/>/gi, '')
      .replace(/\son[a-z]+\s*=\s*"[^"]*"|\son[a-z]+\s*=\s*'[^']*'|\son[a-z]+\s*=\s*[^\s>]+/gi, '')
      .replace(/javascript:/gi, '')
    iframe.srcdoc = html
    iframe.style.display = 'block'
    btn.classList.add('code-preview-active')
  }
}

export const markdownUtils = {
  renderer: null,
  _renderer() {
    if (!this.renderer) this.renderer = buildCodeRenderer()
    return this.renderer
  },
  render(text) {
    if (!text) return ''
    const renderer = this._renderer()
    return renderer ? marked.parse(text, { renderer }) : marked.parse(text)
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
  },

  // Escape HTML so user/model content can never inject markup (always escape
  // before building any HTML for x-html rendering).
  escapeHtml(text) {
    return String(text ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
  },

  // Detect unified diff output (git diff / git show / git log -p …)
  looksLikeDiff(text) {
    if (typeof text !== 'string' || !text) return false
    let meta = 0, hunks = 0, plus = 0, minus = 0
    for (const line of text.split('\n')) {
      if (/^diff --git /.test(line) || /^index [0-9a-f]{7,}/.test(line) || /^--- /.test(line) || /^\+\+\+ /.test(line)) meta++
      else if (/^@@ /.test(line)) hunks++
      else if (/^\+[^+]/.test(line)) plus++
      else if (/^-[^-]/.test(line)) minus++
    }
    return meta > 0 && (hunks > 0 || (plus > 0 && minus > 0))
  },

  // Colorize unified diff output. Input is escaped; only our own span tags are added.
  formatDiffHtml(text) {
    const esc = (s) => helpers.escapeHtml(s)
    return String(text ?? '').split('\n').map(line => {
      let cls = ''
      if (/^@@ /.test(line)) cls = 'text-[var(--accent-primary)] font-semibold'
      else if (/^diff --git |^index [0-9a-f]{7,}/.test(line)) cls = 'text-[var(--text-tertiary)] font-semibold'
      else if (/^--- |^\+\+\+ /.test(line)) cls = 'text-[var(--text-tertiary)] font-semibold'
      else if (/^\+[^+]/.test(line)) cls = 'text-[var(--success)]'
      else if (/^-[^-]/.test(line)) cls = 'text-[var(--error)]'
      return cls ? `<span class="${cls}">${esc(line)}</span>` : esc(line)
    }).join('\n')
  },

  // Smart display text: preserve real line breaks and colorize diffs.
  smartText(text) {
    const t = String(text ?? '')
    if (helpers.looksLikeDiff(t)) return helpers.formatDiffHtml(t)
    return helpers.escapeHtml(t)
  },

  // Normalize ANY tool result into display sections for the generic renderer.
  // Handles: run_command ({exit_code, stdout, stderr}), MCP CallToolResult
  // ({content: [{type:'text',text}]}), plain strings, plain dicts (custom tools).
  toolResultView(result) {
    if (!result) return { summary: '', stdout: '', stderr: '', body: '', kind: 'empty', ok: true }
    if (typeof result === 'string') return { summary: '', stdout: '', stderr: '', body: result, kind: 'text', ok: true }
    if (Array.isArray(result.content)) {
      const texts = result.content.filter(i => i && i.type === 'text').map(i => i.text)
      return { summary: '', stdout: '', stderr: '', body: texts.join('\n'), kind: 'text', ok: true }
    }
    if (typeof result.text === 'string') return { summary: '', stdout: '', stderr: '', body: result.text, kind: 'text', ok: true }
    if (typeof result.content === 'string') return { summary: '', stdout: '', stderr: '', body: result.content, kind: 'text', ok: true }
    if ('exit_code' in result || 'stdout' in result || 'stderr' in result) {
      const exit = result.exit_code
      const ok = exit === undefined || exit === null || exit === 0
      let summary = `exit ${exit === undefined || exit === null ? '?' : exit}`
      if (result.duration_ms != null) summary += ` · ${result.duration_ms}ms`
      if (result.truncated) summary += ' · output truncated'
      return { summary, stdout: result.stdout || '', stderr: result.stderr || '', body: '', kind: 'command', ok }
    }
    if (typeof result.error === 'string') return { summary: '', stdout: '', stderr: '', body: result.error, kind: 'error', ok: false }
    // Plain dict from custom tools — pretty print with real line breaks.
    const body = helpers.parseEscapes(JSON.stringify(result, null, 2))
    return { summary: '', stdout: '', stderr: '', body, kind: 'json', ok: true }
  },

  // Find image assets referenced in tool output that this app actually serves
  // (outputs/ or uploads/ are mounted at /outputs and /uploads). Returns
  // [{ src, kind }] so the UI can render an inline preview.
  assetPreviews(text) {
    if (typeof text !== 'string') return []
    const out = []
    const seen = new Set()
    // Matches outputs/foo.svg or uploads/foo.png anywhere in the text, incl.
    // inside absolute paths, quotes, or backticks. Lookbehind avoids matching
    // e.g. "myoutputs/x.svg".
    const re = /(?<![A-Za-z0-9_-])(outputs|uploads)\/[\w@./-]+\.(svg|png|jpe?g|gif|webp)/gi
    let m
    while ((m = re.exec(text))) {
      const src = '/' + m[0]
      if (!seen.has(src)) {
        seen.add(src)
        out.push({ src, kind: m[2].toLowerCase() })
      }
    }
    return out
  }
}

// Note: api, formatters, markdownUtils, helpers are already exported individually above

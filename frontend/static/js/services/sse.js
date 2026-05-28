/**
 * SSE Service — Streaming Server-Sent Events
 */

export class SSEService {
  constructor() {
    this.controller = null
    this.handlers = { data: [], error: [], complete: [] }
  }

  stream(requestId, conversationId, options = {}) {
    this.controller = new AbortController()
    // Clear old handlers so stale closures don't process new stream data
    this.handlers = { data: [], error: [], complete: [] }
    const url = `/api/stream/${requestId}?conversation_id=${conversationId}` +
      (options.enableRag ? '&enable_rag=1' : '') +
      (options.model ? `&model=${encodeURIComponent(options.model)}` : '') +
      (options.documentIds?.length ? `&document_ids=${encodeURIComponent(options.documentIds.join(','))}` : '')

    fetch(url, { signal: this.controller.signal })
      .then(response => {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        const read = () => {
          reader.read().then(({ done, value }) => {
            if (done) {
              this.handlers.complete.forEach(h => h())
              return
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              try {
                const data = JSON.parse(line.slice(6))
                this.handlers.data.forEach(h => h(data))
              } catch (e) { /* skip malformed */ }
            }

            read()
          }).catch(err => {
            if (err.name !== 'AbortError') {
              this.handlers.error.forEach(h => h(err))
            }
          })
        }

        read()
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          this.handlers.error.forEach(h => h(err))
        }
      })

    return {
      onData: (fn) => this.handlers.data.push(fn),
      onError: (fn) => this.handlers.error.push(fn),
      onComplete: (fn) => this.handlers.complete.push(fn),
      close: () => this.close()
    }
  }

  close() {
    this.controller?.abort()
    this.controller = null
  }
}

export const sseService = new SSEService()

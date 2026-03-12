/**
 * SSE Service - Server-Sent Events for streaming LLM responses
 * 
 * Uses fetch with AbortController instead of EventSource to allow
 * streaming to continue in background when navigating away from chat page.
 */
export class SSEService {
  constructor() {
    this.controller = null
    this.streamingConversationId = null
    this.streamActive = false
    this.dataHandler = null
    this.errorHandler = null
    this.completeHandler = null
  }

  async stream(requestId, conversationId, options = {}) {
    let url = `/api/stream/${requestId}?conversation_id=${conversationId}`
    if (options.enableWebSearch) url += '&enable_web_search=true'
    if (options.enableRag) url += '&enable_rag=true'
    if (options.model) url += `&model=${encodeURIComponent(options.model)}`

    this.streamingConversationId = conversationId
    this.streamActive = true
    this.controller = new AbortController()

    // Return handlers immediately, then start streaming in background
    const handlers = {
      onData: (handler) => {
        this.dataHandler = handler
      },
      onError: (handler) => {
        this.errorHandler = handler
      },
      onComplete: (handler) => {
        this.completeHandler = handler
      }
    }

    // Start streaming in background (don't block on this)
    this.setupFetchStream(url)

    return handlers
  }

  async setupFetchStream(url) {
    let streamCompleted = false

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        signal: this.controller.signal
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (this.streamActive && !streamCompleted) {
        const { done, value } = await reader.read()

        if (done) {
          streamCompleted = true
          break
        }

        buffer += decoder.decode(value, { stream: true })

        // Process complete lines
        const lines = buffer.split('\n')
        buffer = lines.pop() // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmedLine = line.trim()
          if (!trimmedLine || trimmedLine === 'data: [DONE]') {
            continue
          }

          if (trimmedLine.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmedLine.slice(6))
              this.emitData(data)
            } catch (error) {
              console.error('Error parsing SSE data:', error)
            }
          }
        }
      }

      streamCompleted = true
      this.emitComplete()

    } catch (error) {
      if (error.name === 'AbortError') {
        // Stream was cancelled by user - this is expected
        console.log('Stream cancelled by user')
      } else {
        console.error('SSE stream error:', error)
        this.emitError(error)
      }
      streamCompleted = true
    }
  }

  emitData(data) {
    if (this.dataHandler) {
      this.dataHandler(data)
    }
  }

  emitError(error) {
    if (this.errorHandler) {
      this.errorHandler(error)
    }
  }

  emitComplete() {
    if (this.completeHandler) {
      this.completeHandler()
    }
  }

  close() {
    if (this.controller) {
      this.controller.abort()
      this.controller = null
    }
    this.streamActive = false
    this.dataHandler = null
    this.errorHandler = null
    this.completeHandler = null
  }

  // Check if currently streaming
  isStreaming() {
    return this.streamActive && this.controller !== null
  }

  // Get current streaming conversation ID
  getStreamingConversationId() {
    return this.streamingConversationId
  }
}

export const sseService = new SSEService()

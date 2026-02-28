/**
 * SSE Service - Server-Sent Events for streaming LLM responses
 */
export class SSEService {
  constructor() {
    this.eventSource = null
    this.streamingConversationId = null
  }

  stream(requestId, conversationId, options = {}) {
    let url = `/api/stream/${requestId}?conversation_id=${conversationId}`
    if (options.enableWebSearch) url += '&enable_web_search=true'
    if (options.enableRag) url += '&enable_rag=true'
    if (options.model) url += `&model=${encodeURIComponent(options.model)}`
    
    this.streamingConversationId = conversationId
    this.eventSource = new EventSource(url)
    
    return this.setupListeners()
  }

  setupListeners() {
    let streamCompleted = false
    
    return {
      onData: (handler) => {
        this.eventSource.onmessage = (event) => {
          try {
            handler(JSON.parse(event.data))
          } catch (error) {
            console.error('Error parsing SSE:', error)
          }
        }
      },
      onError: (handler) => {
        this.eventSource.onerror = () => {
          if (!streamCompleted) {
            streamCompleted = true
            this.close()
          }
        }
      }
    }
  }

  close() {
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
  }
}

export const sseService = new SSEService()

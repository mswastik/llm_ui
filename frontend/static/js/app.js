/**
 * Main Application Entry Point
 * Loads all Alpine.js components and starts the application
 */

// Import utilities (makes them available globally for components)
import { formatters, markdownUtils, helpers, api } from './utils.js'
import { SSEService, sseService } from './services/sse.js'
import { TTSService, ttsService } from './services/tts.js'

// Make services globally available for inline handlers
window.sseService = sseService
window.ttsService = ttsService

// Import store (MUST be first - registers all stores)
import './store.js'

// Import components (register Alpine.data definitions)
import './components/sidebar.js'
import './components/chat.js'
import './components/settings.js'
import './components/documents.js'

// Now start Alpine after all components are registered
if (window.Alpine) {
  window.Alpine.start()
}

// Export for potential external use
export { formatters, markdownUtils, helpers, api }
export { sseService, ttsService }

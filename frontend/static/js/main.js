/**
 * Main Application Entry Point
 *
 * Uses Alpine's ESM module build for full control over initialization.
 * No auto-start — we manually call Alpine.start() after all registration.
 *
 * Execution order (module import graph):
 * 1. Alpine ESM module → sets window.Alpine
 * 2. Component files → export their factories
 * 3. main.js → imports Alpine + factories, registers stores + components, calls start()
 */

// ═══════════════════════════════════════════════════════════
// 1. Import Alpine.js ESM module (sets window.Alpine)
// ═══════════════════════════════════════════════════════════
import Alpine from 'https://cdn.jsdelivr.net/npm/alpinejs@3.15.12/+esm'
import collapse from 'https://cdn.jsdelivr.net/npm/@alpinejs/collapse@3.15.12/+esm'

// Install collapse plugin
Alpine.plugin(collapse)

// Make Alpine available globally (for any inline scripts or devtools)
window.Alpine = Alpine

console.log('[main] Alpine ESM module loaded')

// ═══════════════════════════════════════════════════════════
// 2. Import SSE + TTS services (no Alpine dependency)
// ═══════════════════════════════════════════════════════════
import { SSEService, sseService } from './services/sse.js'
import { TTSService, ttsService } from './services/tts.js?v=38'
window.sseService = sseService
window.ttsService = ttsService

// ═══════════════════════════════════════════════════════════
// 3. Import utils (provides formatters, markdownUtils, helpers, api)
// ═══════════════════════════════════════════════════════════
import { helpers, formatters } from './utils.js'
window.helpers = helpers
window.formatters = formatters

// ═══════════════════════════════════════════════════════════
// 4. Register stores from inline data (available in base.html)
// ═══════════════════════════════════════════════════════════
if (window.__chatStoreData__ && window.__uiStoreData__) {
  Alpine.store('chat', window.__chatStoreData__)
  Alpine.store('ui', window.__uiStoreData__)
  window.__uiStoreData__.initTheme()
  console.log('[main] Stores registered: chat, ui')
} else {
  console.error('[main] Store data not available on window')
}

// ═══════════════════════════════════════════════════════════
// 5. Import component factories and register them
// ═══════════════════════════════════════════════════════════
import { sidebar } from './components/sidebar.js'
import { chatComponent } from './components/chat.js'
import { settings } from './components/settings.js'
// ─── Modal factory (extracted from store.js which is now removed) ────
function createModal(storeKey, openMethod, closeMethod) {
  return function () {
    return {
      open: false,
      openModal() {
        this.open = true
        if (typeof openMethod === 'function') {
          try { openMethod() } catch(e) { console.error('[modal] openMethod error:', e) }
        }
      },
      closeModal() {
        if (closeMethod) {
          try {
            if (typeof closeMethod === 'function') closeMethod()
            else if (typeof $store !== 'undefined' && typeof $store.ui[closeMethod] === 'function') {
              $store.ui[closeMethod]()
            }
          } catch(e) { console.error('[modal] closeMethod error:', e) }
        }
        this.open = false
      }
    }
  }
}

Alpine.data('sidebar', sidebar)
Alpine.data('chat', chatComponent)
Alpine.data('settings', settings)
Alpine.data('modalDocuments', createModal('documents', 'openDocuments', 'closeDocuments'))
Alpine.data('modalSettings', createModal('settings', 'openSettings', 'closeSettings'))
Alpine.data('modalAgents', createModal('agents', 'openAgents', 'closeAgents'))
Alpine.data('modalNotes', createModal('notes', 'openNotes', 'closeNotes'))
console.log('[main] Components registered: sidebar, chat, settings, modalDocuments, modalSettings, modalAgents, modalNotes')

// ═══════════════════════════════════════════════════════════
// 6. Start Alpine AFTER all stores and components are registered
// ═══════════════════════════════════════════════════════════
console.log('[main] Starting Alpine.js...')
Alpine.start()
console.log('[main] Alpine.js started successfully')

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
import { SSEService, sseService } from './services/sse.js?v=70'
import { TTSService, ttsService } from './services/tts.js?v=46'
window.sseService = sseService
window.ttsService = ttsService

// ═══════════════════════════════════════════════════════════
// 3. Import utils (provides formatters, markdownUtils, helpers, api)
// ═══════════════════════════════════════════════════════════
import { helpers, formatters } from './utils.js?v=63'
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
import { sidebar } from './components/sidebar.js?v=101'
import { chatComponent } from './components/chat.js?v=80'
import { settings } from './components/settings.js?v=73'
import { skillsPanel } from './components/skills.js?v=67'
import { jobsModal } from './components/jobs.js?v=64'
import { agentsPanel } from './components/agents_panel.js?v=70'
import { capabilityPicker } from './components/tag_picker.js?v=1'
import { library } from './components/library.js?v=25'
import { reader } from './components/reader.js?v=33'
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
Alpine.data('modalNotes', createModal('notes', 'openNotes', 'closeNotes'))
Alpine.data('modalJobs', jobsModal)
Alpine.data('agentsPanel', agentsPanel)
Alpine.data('skillsPanel', skillsPanel)
Alpine.data('capabilityPicker', capabilityPicker)
Alpine.data('library', library)
Alpine.data('reader', reader)
console.log('[main] Components registered: sidebar, chat, settings, modalDocuments, modalSettings, modalNotes, modalJobs, agentsPanel, skillsPanel, capabilityPicker, library, reader')

// ═══════════════════════════════════════════════════════════
// 6. Start Alpine AFTER all stores and components are registered
// ═══════════════════════════════════════════════════════════
console.log('[main] Starting Alpine.js...')
Alpine.start()
console.log('[main] Alpine.js started successfully')

// ═════════════════════════════════════════════════
// 7. Register PWA service worker (app-shell offline + installability)
// ═════════════════════════════════════════════════
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      console.warn('[PWA] Service worker registration failed:', err);
    });
  });
}

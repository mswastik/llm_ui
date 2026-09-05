/**
 * TTS Service — streaming text-to-speech playback
 *
 * Reads /api/tts/generate/stream (NDJSON): one segment URL per line, played as
 * soon as it arrives. One shared <audio controls> element is injected into the
 * active message's slot, so the browser's OWN pause control always works —
 * and a native pause means "stop" (same semantics as the UI pause button).
 */

import { getTtsVolume } from '../utils.js'

export class TTSService {
  constructor() {
    this.ttsAvailable = false
    this.isPlaying = false
    this.isPaused = false
    this.currentAudioMessageId = null
    // Per-call cancellation token: stop() bumps it, invalidating every in-flight speak()
    this.stopCounter = 0
    this.abortController = null
    // Single writer: the chat component registers this to mirror audio state into the store
    this.onStateChange = null

    // Streaming player state
    this.segments = []       // [{ url }] generated so far (queue)
    this.segmentIndex = -1   // index of the segment currently in the player
    this.streamComplete = false
    this.waiting = false     // player ended but next segment not generated yet
    this.player = null       // the one shared <audio controls> element
    this.playerSlot = null   // DOM element it's currently injected in
    this._finishCb = null
    this._stopping = false
    // Volume-boost graph (Web Audio gain). Built lazily only when the
    // Settings → TTS volume differs from 1× so default playback is never
    // routed through (or dependent on) an AudioContext.
    this._actx = null
    this._gainNode = null
  }

  // Route the player through a gain node when the configured volume isn't
  // 1×. The <audio> element's own volume tops out at 1.0; boosting past
  // 100% needs Web Audio. Must be (re)applied before each play() — also
  // resumes the context (a gesture may be needed; see the listeners).
  #ensureVolume() {
    const el = this.player
    if (!el) return
    const vol = getTtsVolume()
    if (Math.abs(vol - 1) > 0.001 && !this._gainNode) {
      try {
        const Ctor = window.AudioContext || window.webkitAudioContext
        if (!Ctor) {
          el.volume = Math.min(1, vol)
          return
        }
        const actx = new Ctor()
        const src = actx.createMediaElementSource(el)
        const gain = actx.createGain()
        src.connect(gain)
        gain.connect(actx.destination)
        this._actx = actx
        this._gainNode = gain
        // Resume on the next user gesture (contexts start suspended). The
        // native play button on the visible player is the reliable one; the
        // window listeners also catch auto-read cases.
        const warm = () => { try { actx.resume() } catch { /* noop */ } }
        window.addEventListener('pointerdown', warm, { once: true, capture: true })
        window.addEventListener('keydown', warm, { once: true, capture: true })
        window.addEventListener('touchstart', warm, { once: true, capture: true })
        el.addEventListener('play', () => {
          if (actx.state === 'suspended') actx.resume().catch(() => {})
        })
      } catch {
        this._gainNode = null
      }
    }
    if (this._actx && this._actx.state === 'suspended') {
      this._actx.resume().catch(() => {})
    }
    if (this._gainNode) this._gainNode.gain.value = Math.max(0.2, Math.min(2, vol))
  }

  // Emit the authoritative state (drift-proof: derived from real playback events)
  #emit() {
    this.onStateChange?.({
      playing: this.isPlaying,
      paused: this.isPaused,
      msgId: this.currentAudioMessageId,
    })
  }

  #getPlayer() {
    if (this.player) return this.player
    const el = document.createElement('audio')
    el.controls = true
    el.preload = 'auto'
    el.className = 'tts-player max-w-48 h-8 rounded flex-shrink-0'
    el.addEventListener('ended', () => this.#advance())
    el.addEventListener('pause', (e) => {
      if (this._stopping) return
      // Natural end-of-media fires a trusted 'pause' before 'ended' — that
      // must ADVANCE, not pause. Anything else that pauses mid-segment
      // (user pressed the native control) means paused, not stopped.
      if (e.isTrusted && !el.ended && (this.isPlaying || this.isPaused)) {
        this.isPlaying = false
        this.isPaused = true
        this.#emit()
      }
    })
    el.addEventListener('play', () => {
      if (!this._stopping) {
        this.isPlaying = true
        this.isPaused = false
        this.#emit()
      }
    })
    el.addEventListener('error', () => {
      if (this.isPlaying && !this._stopping) {
        this.#fail(null, 'Playback error')
      }
    })
    this.player = el
    return el
  }

  // Show the shared player inside the active message's slot.
  // Re-query the slot at insert time — Alpine re-renders can replace the DOM
  // node captured earlier, leaving us with a detached reference.
  #showPlayer() {
    document.querySelectorAll('.tts-slot').forEach((s) => s.classList.add('hidden'))
    const slot = (this.currentAudioMessageId && document.getElementById('tts-slot-' + this.currentAudioMessageId)) || this.playerSlot
    if (slot) {
      this.playerSlot = slot
      slot.classList.remove('hidden')
      slot.appendChild(this.#getPlayer())
    }
  }

  // Advance to the next queued segment; wait if it hasn't been generated yet
  #advance() {
    if (this._stopping || !this.segments.length) return
    const next = this.segmentIndex + 1
    if (next >= this.segments.length) {
      if (this.streamComplete) this.#finish()
      else this.waiting = true
      return
    }
    this.waiting = false
    this.segmentIndex = next
    const el = this.#getPlayer()
    el.src = this.segments[next].url
    this.#ensureVolume()
    // play() returns a promise — a rejected one (e.g. element detached during
    // an Alpine re-render) must be caught or it surfaces as an unhandled rejection
    el.play().catch(() => {})
  }

  #finish() {
    this.isPlaying = false
    this.isPaused = false
    this.currentAudioMessageId = null
    const cb = this._finishCb
    this._finishCb = null
    this.#emit()
    cb?.()
  }

  #fail(onError, msg) {
    this.stop()
    onError?.(msg)
  }

  async checkAvailability() {
    try {
      const res = await fetch('/api/tts/status')
      const data = await res.json()
      this.ttsAvailable = data.available || false
    } catch { this.ttsAvailable = false }
  }

  /**
   * Stream + play a message. Returns true if playback started (or will start
   * once the first segment arrives), false if it failed or was stopped.
   *
   * @param {object|string} message message object (uses .id/.content) or plain text
   * @param {HTMLElement|null} mountEl slot element the player is injected into
   * @param {Function} onError
   * @param {Function} onEnd called when the full clip finishes naturally
   */
  async speak(message, mountEl, onError, onEnd) {
    // Stop whatever is playing/generating now (also bumps stopCounter)
    this.stop()

    const myStop = this.stopCounter
    const msgId = typeof message === 'object' ? message.id : null
    const textContent = typeof message === 'object' ? (message.content || '') : message
    if (!textContent?.trim()) {
      onError?.('No text to speak')
      return false
    }

    this.abortController = new AbortController()
    this.segments = []
    this.segmentIndex = -1
    this.streamComplete = false
    this.waiting = false
    this._finishCb = onEnd
    this.playerSlot = mountEl || null
    this.currentAudioMessageId = msgId
    this.isPlaying = true // optimistic — pause icon shows immediately
    this.#emit()

    let res
    try {
      res = await fetch('/api/tts/generate/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textContent }),
        signal: this.abortController.signal,
      })
      if (this.stopCounter !== myStop) return false
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'TTS generation failed' }))
        throw new Error(err.detail || 'TTS generation failed')
      }
    } catch (e) {
      if (e?.name === 'AbortError' || this.stopCounter !== myStop) return false
      this.#fail(onError, e.message || 'TTS error')
      return false
    }

    // Read the NDJSON stream, playing the first segment as soon as it arrives
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    let firstSegmentPlayed = false
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        if (this.stopCounter !== myStop) return false
        buf += decoder.decode(value, { stream: true })
        let nl
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          let evt
          try { evt = JSON.parse(line) } catch { continue }
          if (evt.error) { this.#fail(onError, evt.error); return false }
          if (evt.url) {
            this.segments.push({ url: evt.url })
            if (!firstSegmentPlayed) {
              firstSegmentPlayed = true
              this.#showPlayer()
              const el = this.#getPlayer()
              el.src = evt.url
              this.segmentIndex = 0
              this.#ensureVolume()
              try {
                await el.play()
              } catch (e) {
                if (e?.name === 'NotAllowedError') {
                  // Autoplay blocked — the visible player's native button rescues it
                  onError?.('Press play to listen')
                } else if (this.stopCounter !== myStop) {
                  this.#fail(onError, 'Playback failed: ' + e.message)
                  return false
                }
              }
            } else if (this.waiting) {
              this.#advance() // player ended while this segment was generating
            }
          }
        }
      }
    } catch (e) {
      if (e?.name === 'AbortError' || this.stopCounter !== myStop) return false
      this.#fail(onError, 'Stream interrupted: ' + e.message)
      return false
    }

    if (this.stopCounter !== myStop) return false
    if (!this.segments.length) { this.#fail(onError, 'No audio generated'); return false }

    this.streamComplete = true
    if (this.waiting) this.#advance()
    return true
  }

  stop() {
    this._stopping = true
    this.stopCounter++
    this.abortController?.abort()
    this.abortController = null
    if (this.player) {
      try { this.player.pause() } catch { /* noop */ }
      // keep src so the native control can replay the current segment
    }
    this.segments = []
    this.segmentIndex = -1
    this.streamComplete = false
    this.waiting = false
    this._finishCb = null
    this.isPlaying = false
    this.isPaused = false
    this.currentAudioMessageId = null
    this._stopping = false
    this.#emit()
  }

  // Pause mid-stream: element keeps currentTime + segment queue + the open
  // NDJSON fetch, so resume() continues exactly where it stopped.
  pause() {
    if (!this.player || !this.isPlaying) return
    this.player.pause() // untrusted event — listener won't re-enter
    this.isPlaying = false
    this.isPaused = true
    this.#emit()
  }

  resume() {
    if (!this.player || !this.isPaused) return
    this.isPaused = false
    this.isPlaying = true
    this.#emit()
    this.#ensureVolume()
    this.player.play().catch(() => {})
  }
}

export const ttsService = new TTSService()
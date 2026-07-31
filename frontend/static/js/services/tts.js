/**
 * TTS Service — Text-to-Speech playback
 */

export class TTSService {
  constructor() {
    this.ttsAvailable = false
    this.currentAudio = null
    this.currentAudioMessageId = null
    this.isPlaying = false
    // Per-call cancellation token: stop() bumps it, which invalidates EVERY
    // in-flight speak() regardless of how many new speaks start afterwards.
    this.stopCounter = 0
    // Every element this service ever created — stop() pauses ALL of them,
    // so a stale/null currentAudio can never leave audio playing.
    this.audioElements = new Set()
    this.abortController = null
    // Single writer: the chat component registers this to mirror real audio state into the store
    this.onStateChange = null
  }

  // Emit the authoritative (drift-proof) state derived from the audio element
  #emit() {
    this.onStateChange?.({
      playing: this.isPlaying,
      msgId: this.currentAudioMessageId,
    })
  }

  #reset(el) {
    this.audioElements.delete(el)
    if (this.currentAudio === el) this.currentAudio = null
    this.isPlaying = false
    this.currentAudioMessageId = null
    this.#emit()
  }

  #attachEvents(audio, onEnd) {
    audio.addEventListener('play', () => {
      this.isPlaying = true
      this.#emit()
    })
    audio.addEventListener('pause', () => {
      if (this.currentAudio === audio) {
        this.isPlaying = false
        this.#emit()
      }
    })
    audio.addEventListener('ended', () => {
      if (this.currentAudio === audio) {
        this.#reset(audio)
        onEnd?.()
      }
    })
    audio.addEventListener('error', () => {
      if (this.currentAudio === audio) {
        this.#reset(audio)
        onEnd?.()
      }
    })
  }

  async checkAvailability() {
    try {
      const res = await fetch('/api/tts/status')
      const data = await res.json()
      this.ttsAvailable = data.available || false
    } catch { this.ttsAvailable = false }
  }

  async speak(message, onError, onEnd) {
    const myStop = this.stopCounter
    try {
      const textContent = typeof message === 'object' ? (message.content || '') : message

      // Cancel any previous in-flight generation (old responses never play)
      this.abortController?.abort()
      this.abortController = new AbortController()

      const res = await fetch('/api/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textContent }),
        signal: this.abortController.signal,
      })
      if (this.stopCounter !== myStop) return false

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'TTS failed' }))
        throw new Error(err.detail || 'TTS generation failed')
      }
      const data = await res.json()

      if (this.stopCounter !== myStop) return false

      if (!data.success) {
        throw new Error(data.error || 'TTS generation failed')
      }

      const audioUrl = data.audio_url || `/api/audio/${data.filename}`

      if (this.currentAudio) { this.currentAudio.pause() }

      const audio = new Audio(audioUrl)
      this.currentAudio = audio
      this.currentAudioMessageId = typeof message === 'object' ? message.id : null
      this.isPlaying = true
      this.audioElements.add(audio)
      this.#emit()

      // State follows the audio element's real events (play/pause/ended/error)
      this.#attachEvents(audio, onEnd)

      try {
        await audio.play()
      } catch (e) {
        if (this.stopCounter !== myStop) return false
        if (e && e.name === 'NotAllowedError') {
          // Autoplay blocked (play happened long after the click). Drop back to
          // idle — the next click re-speaks from cache with a fresh gesture.
          this.#reset(audio)
          onError?.('Click the button again to play')
          return false
        }
        this.#reset(audio)
        onError?.('Playback failed: ' + e.message)
        onEnd?.()
        return false
      }

      return true
    } catch (e) {
      // A deliberate stop aborts the fetch — that is not an error
      if (e?.name === 'AbortError' || this.stopCounter !== myStop) return false
      onError?.(e.message || 'TTS error')
      return false
    }
  }

  stop() {
    this.stopCounter++
    this.abortController?.abort()
    this.abortController = null
    console.log('[tts] stop — pausing', this.audioElements.size, 'audio element(s)')
    this.audioElements.forEach(el => { try { el.pause() } catch { /* noop */ } })
    this.audioElements.clear()
    this.currentAudio = null
    this.currentAudioMessageId = null
    this.isPlaying = false
    this.#emit()
  }
}

export const ttsService = new TTSService()

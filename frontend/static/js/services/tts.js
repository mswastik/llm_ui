/**
 * TTS Service — Text-to-Speech playback
 */

export class TTSService {
  constructor() {
    this.ttsAvailable = false
    this.currentAudio = null
    this.currentAudioMessageId = null
    this.isPlaying = false
    this.stopped = false
    // Every element this service ever created — stop() pauses ALL of them,
    // so a stale/null currentAudio can never leave audio playing
    this.audioElements = new Set()
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
        this.isPlaying = false
        this.currentAudio = null
        this.audioElements.delete(audio)
        this.#emit()
        onEnd?.()
      }
    })
    audio.addEventListener('error', () => {
      if (this.currentAudio === audio && !this.stopped) {
        this.isPlaying = false
        this.currentAudio = null
        this.audioElements.delete(audio)
        this.#emit()
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
    this.stopped = false
    try {
      const textContent = typeof message === 'object' ? (message.content || '') : message
      const res = await fetch('/api/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textContent })
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'TTS failed' }))
        throw new Error(err.detail || 'TTS generation failed')
      }
      const data = await res.json()

      if (this.stopped) return false

      if (!data.success) {
        throw new Error(data.error || 'TTS generation failed')
      }

      const audioUrl = data.audio_url || `/api/audio/${data.filename}`

      if (this.currentAudio) { this.currentAudio.pause() }

      this.currentAudio = new Audio(audioUrl)
      this.currentAudioMessageId = typeof message === 'object' ? message.id : null
      this.isPlaying = true
      this.audioElements.add(this.currentAudio)
      this.#emit()

      // State follows the audio element's real events (play/pause/ended/error)
      this.#attachEvents(this.currentAudio, onEnd)

      try {
        await this.currentAudio.play()
      } catch (e) {
        // Autoplay blocked: keep the audio loaded so the next click starts it
        if (e && e.name === 'NotAllowedError') {
          onError?.('Audio ready — click the button to play')
          return true
        }
        const el = this.currentAudio
        this.isPlaying = false
        this.currentAudio = null
        this.audioElements.delete(el)
        this.#emit()
        onError?.('Playback failed: ' + e.message)
        onEnd?.()
        return false
      }

      return true
    } catch (e) {
      onError?.(e.message || 'TTS error')
      return false
    }
  }

  stop() {
    this.stopped = true
    // Pause every element we ever created — even if currentAudio drifted null,
    // the playing element is still in the set and gets silenced
    this.audioElements.forEach(el => { try { el.pause() } catch { /* noop */ } })
    this.audioElements.clear()
    this.currentAudio = null
    this.isPlaying = false
    this.#emit()
  }
}

export const ttsService = new TTSService()

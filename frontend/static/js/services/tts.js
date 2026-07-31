/**
 * TTS Service — Text-to-Speech playback
 */

export class TTSService {
  constructor() {
    this.ttsAvailable = false
    this.currentAudio = null
    this.currentAudioMessageId = null
    this.isPlaying = false
  }

  async checkAvailability() {
    try {
      const res = await fetch('/api/tts/status')
      const data = await res.json()
      this.ttsAvailable = data.available || false
    } catch { this.ttsAvailable = false }
  }

  async speak(message, onError, onEnd) {
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

      if (!data.success) {
        throw new Error(data.error || 'TTS generation failed')
      }

      const audioUrl = data.audio_url || `/api/audio/${data.filename}`

      if (this.currentAudio) { this.currentAudio.pause() }

      this.currentAudio = new Audio(audioUrl)
      this.currentAudioMessageId = typeof message === 'object' ? message.id : null
      this.isPlaying = true

      // Wire ended BEFORE play() so even a short clip flips the icon back only on completion
      this.currentAudio.onended = () => {
        this.isPlaying = false
        this.currentAudio = null
        onEnd?.()
      }

      try {
        await this.currentAudio.play()
      } catch (e) {
        this.isPlaying = false
        this.currentAudio = null
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
    this.currentAudio?.pause()
    this.currentAudio = null
    this.isPlaying = false
  }
}

export const ttsService = new TTSService()

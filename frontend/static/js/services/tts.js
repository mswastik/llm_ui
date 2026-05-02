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

  async speak(message, onProgress, onError) {
    try {
      const res = await fetch('/api/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message.content || message })
      })
      if (!res.ok) throw new Error('TTS generation failed')
      const data = await res.json()
      const audioUrl = `/api/audio/${data.filename}`

      if (this.currentAudio) { this.currentAudio.pause() }

      this.currentAudio = new Audio(audioUrl)
      this.currentAudioMessageId = typeof message === 'object' ? message.id : null
      this.isPlaying = true

      this.currentAudio.play().catch(e => {
        this.isPlaying = false
        onError?.('Playback failed')
      })

      this.currentAudio.onended = () => {
        this.isPlaying = false
        this.currentAudio = null
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

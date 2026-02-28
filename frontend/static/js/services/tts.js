/**
 * TTS Service - Text-to-Speech functionality
 */
import { api } from '../utils.js'

export class TTSService {
  constructor() {
    this.currentAudio = null
    this.currentAudioMessageId = null
    this.isPlaying = false
    this.ttsAvailable = false
  }

  async checkAvailability() {
    try {
      const data = await api.get('/api/tts/status')
      this.ttsAvailable = data.available
    } catch {
      this.ttsAvailable = false
    }
    return this.ttsAvailable
  }

  async speak(message, onError) {
    if (!this.ttsAvailable) {
      onError?.('TTS not available')
      return false
    }

    if (this.currentAudioMessageId === message.id && this.currentAudio) {
      return this.togglePlayPause()
    }

    this.stop()

    let text = message.content.replace(/<[^>]*>/g, '').trim()
    text = text.replace(/[*#`]/g, '') // Strip markdown
    if (!text) {
      onError?.('No text to speak')
      return false
    }

    try {
      const data = await api.post('/api/tts/generate', { text })
      if (data.success && data.audio_url) {
        await this.playAudio(data.audio_url, message.id)
        return true
      }
    } catch {
      onError?.('TTS failed')
    }
    return false
  }

  async playAudio(audioUrl, messageId) {
    return new Promise((resolve, reject) => {
      this.currentAudio = new Audio(audioUrl)
      this.currentAudioMessageId = messageId
      this.isPlaying = true
      
      this.currentAudio.onended = () => this.cleanup()
      this.currentAudio.onpause = () => { this.isPlaying = false }
      this.currentAudio.onplay = () => { this.isPlaying = true }
      this.currentAudio.onerror = () => { this.cleanup(); reject(new Error('Audio error')) }
      
      this.currentAudio.play().catch(reject)
    })
  }

  togglePlayPause() {
    if (!this.currentAudio) return false
    if (this.isPlaying) {
      this.currentAudio.pause()
      this.isPlaying = false
    } else {
      this.currentAudio.play()
      this.isPlaying = true
    }
    return this.isPlaying
  }

  stop() {
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio.currentTime = 0
      this.cleanup()
    }
  }

  cleanup() {
    this.currentAudio = null
    this.currentAudioMessageId = null
    this.isPlaying = false
  }
}

export const ttsService = new TTSService()

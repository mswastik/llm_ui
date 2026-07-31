/**
 * STT Service — Speech-to-Text (Voice Input)
 *
 * Uses the browser MediaRecorder API to record audio from the microphone,
 * then sends the recorded blob to the backend for transcription.
 */

export class STTService {
  constructor() {
    this.mediaRecorder = null
    this.audioChunks = []
    this.isRecording = false
    this.stream = null
    this.supported = false
    this._checkSupport()
  }

  _checkSupport() {
    this.supported = !!(
      navigator.mediaDevices?.getUserMedia &&
      window.MediaRecorder
    )
  }

  isSupported() {
    return this.supported
  }

  async startRecording() {
    if (this.isRecording) return
    if (!this.supported) throw new Error('Recording not supported in this browser')

    this.audioChunks = []

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (e) {
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        throw new Error('Microphone permission denied')
      }
      throw new Error('Could not access microphone: ' + e.message)
    }

    // Prefer WebM Opus (small, good quality), fall back to browser default
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
          ? 'audio/ogg;codecs=opus'
          : ''

    this.mediaRecorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : {})

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.audioChunks.push(e.data)
    }

    this.mediaRecorder.start(250) // collect data every 250ms
    this.isRecording = true
  }

  async stopRecording() {
    if (!this.isRecording || !this.mediaRecorder) return null

    return new Promise((resolve, reject) => {
      this.mediaRecorder.onstop = async () => {
        this.isRecording = false
        this._stopTracks()

        const blob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType || 'audio/webm' })
        this.audioChunks = []

        if (blob.size < 100) {
          resolve(null) // too short, probably silence
          return
        }

        try {
          const text = await this._transcribe(blob)
          resolve(text)
        } catch (e) {
          reject(e)
        }
      }

      this.mediaRecorder.stop()
    })
  }

  cancelRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.onstop = null
      this.mediaRecorder.stop()
    }
    this.isRecording = false
    this.audioChunks = []
    this._stopTracks()
  }

  _stopTracks() {
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop())
      this.stream = null
    }
  }

  async _transcribe(blob) {
    const formData = new FormData()
    formData.append('audio', blob, 'recording.webm')

    const res = await fetch('/api/stt/transcribe', {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Transcription failed' }))
      throw new Error(err.detail || 'Transcription failed')
    }

    const data = await res.json()
    return data.text || ''
  }

  async checkAvailability() {
    try {
      const res = await fetch('/api/stt/status')
      const data = await res.json()
      return data
    } catch {
      return { available: false, engines: [] }
    }
  }
}

export const sttService = new STTService()

---
name: piper-tts-speechd-setup
description: Diagnose and fix Piper TTS + Speech Dispatcher voice configuration issues on Linux
---

<!-- reflection reason: Recurring multi-step diagnostic task: Piper TTS + Speech Dispatcher generic module misconfiguration is a common issue on Linux. The pattern of multiple GenericExecuteSynth lines overriding each other keeps appearing, and the fix requires checking config, verifying voice files exist, providing corrected config, and restarting the service. -->

1. Read the config file at /etc/speech-dispatcher/modules/piper-tts-generic.conf (or user-provided path). 2. Check installed Piper voices: `ls /usr/share/piper-voices/en/en_US/` and verify subdirectories (high/medium/low) exist for each voice. 3. Identify the root cause — most common issue is multiple GenericExecuteSynth lines where only the last one applies, causing all voices to fall back to a hardcoded model. 4. Provide a corrected config with exactly ONE GenericExecuteSynth line that dynamically resolves $VOICE to the correct .onnx model path using find or direct path construction. 5. Instruct user to restart speech-dispatcher: `systemctl --user restart speech-dispatcher`. 6. If no wrapper script exists, provide a bash wrapper that maps $VOICE (e.g. en_US-ryan-high) to the correct model file and pipes piper-tts output to aplay/pw-play at the right sample rate (16kHz for -low voices, 22050Hz otherwise).

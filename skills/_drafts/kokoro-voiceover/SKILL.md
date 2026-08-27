---
name: kokoro-voiceover
description: Generate slowed-down narration audio for explainer videos using Kokoro TTS (speed=0.9) in the ank-content chatterbox venv
---

<!-- reflection reason: Repeatable multi-step procedure (venv selection, CPU-only init, speed-adjusted generation, 24kHz wav export, duration reporting for the video timeline) that will recur for every future explainer video; the user explicitly approved Kokoro and requested slower narration, so the exact recipe should be reusable. -->

1. Use the venv python at /home/swastik/Downloads/repos/ank/ank-content/.venv-chatterbox/bin/python (kokoro + soundfile already installed; no espeak-ng needed).
2. Always init the pipeline on CPU — the GPU is often fully occupied by other processes: KPipeline(lang_code='a', device='cpu', repo_id='hexgrad/Kokoro-82M').
3. Generate per-scene narration with voice='af_heart' and speed=0.9 (user preference: default voice pace is too fast; 0.9 ≈ 7% slower and natural). Adjust only if the user asks.
4. The pipeline yields chunks (i, graphemes, aud); concatenate chunk tensors and write with soundfile at 24000 Hz to a .wav (24kHz mono). Print each clip's duration in seconds.
5. Report printed durations so data-start/data-duration can be set on <audio> tags in the HyperFrames index.html (cumulative offset per scene).
6. Cache outputs: skip regeneration if the target .wav already exists unless text changed.
7. For auditioning, drop samples in ank-content/tts_samples/. Keep the speed=0.9 preference noted in memory unless the user updates it.

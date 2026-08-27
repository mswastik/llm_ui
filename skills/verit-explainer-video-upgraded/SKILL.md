---
name: verit-explainer-video-upgraded
description: Produce professional Verit Analytics explainer videos with 3D effects, cinematic camera movement, light-sweep transitions, micro-effects — no Blender required. Uses HyperFrames + HTML/CSS/GSAP.
---

---
name: verit-explainer-video-upgraded
description: "Professional Verit explainer videos with 3D effects, cinematic camera movement, light-sweep transitions, micro-effects — no Blender required."
---

# Verit Analytics Explainer Videos — Upgraded Quality

Produce short (30-60s) hand-authored educational explainer videos for Verit Analytics (veritanalytics.com — ML supply-chain / demand-forecasting SaaS for small businesses) with **professional-grade visual effects** including 3D perspective transforms, cinematic camera movement on screenshots, light-sweep transitions, parallax depth, and count-up/stagger micro-effects — all within the HyperFrames/HTML/CSS ecosystem. **No Blender installation required.**

## When to use
Any request to make/continue Verit explainer or supply-chain concept videos (bullwhip, safety stock, EOQ, ABC analysis, lead time, dead stock, forecasting, inventory turnover, brand case-stories), OR any SaaS product demo video that needs professional visual polish (camera movement on screenshots, 3D card effects, parallax text, light-sweep transitions).

## Critical decisions (locked by user — do NOT re-litigate)
- **Hand-author each composition** from `verit-video-kit/`. Do NOT use the old `ank-content/generate_video.py` template pipeline.
- **Theory/concept scenes: NO b-roll.** Explain with custom animated objects (growing bars, connecting lines, hub diagrams). The animation IS the explanation.
- Deliberate pauses between scenes; entrances 0.6–0.8s varied eases; hold then 0.5–0.6s fade.
- Chatterbox natural-US voice (one WAV per scene) + warm ambient BGM at data-volume 0.07.
- **NO shadows on or around screenshots** (user-locked): kills visibility on the dark theme. See Screenshot Reveal rules below.

## Key paths
- **Kit:** `/home/swastik/Downloads/repos/ank/ank-content/verit-video-kit/` (README.md, BRAND.md, template.html, gen_voice.py, gen_bgm.py, snippets/, HANDOFF.md)
- **Chatterbox venv:** `/home/swastik/Downloads/repos/ank/ank-content/.venv-chatterbox/bin/python` (Python 3.11, chatterbox-tts 0.1.7)
- **Reference working examples:** `~/workspace/bullwhip-explainer/`, `~/workspace/leadtime-explainer/`, `~/workspace/deadstock-explainer/`
- **CLI:** `npx --yes hyperframes@0.7.37 lint|render`; render `--fps 30 --quality high`
- **AUDIO RENDER (MANDATORY):** this machine's ffmpeg 9.x removed `-filter_complex_script`, so plain renders produce VIDEO-ONLY output. ALWAYS render with:
  `HYPERFRAMES_FFMPEG_PATH=/home/swastik/tools/ffmpeg-wrap npx --yes hyperframes@0.7.37 render --fps 30 --quality high`
  The wrapper rewrites the filter graph into inline form. Verify afterwards: `ffprobe -show_entries stream=codec_type <mp4>` must list an `aac` audio stream. Run renders FROM the project directory — the `--cwd` flag does not reliably relocate output.

## Upgrade Components (Tier 1 Visual Effects)

### A. Cinematic Screenshot Reveal (`snippets/screenshot-reveal.html`)
3D perspective entrance on product screenshots: card starts tilted back and zoomed in, settles flat with subtle floating motion. **Visibility rules (user-locked):**
- NO `box-shadow` on the card — not even soft/subtle ones. Edge definition = solid `border:1px solid rgba(20,184,166,0.35)` only.
- NO blurred `.glow` elements near screenshot scenes and NO dark vignette behind the card.
- Place a FLAT backdrop patch (`background:#0e1524`) behind the card to cancel the global radial-gradient background, which otherwise darkens corners around bright screenshots and reads as shadow.
- Bottom-only gradient overlay for labels is allowed (never touches corners).

### B. 3D Parallax Text (`snippets/parallax-text.html`)
Depth between text layers — headline floats above dimmer giant ghost keyword with shadow gap and parallax drift. Use for hooks and key message scenes.

### C. Zoom-Pan / Ken Burns (`snippets/zoom-pan.html`)
Slow cinematic zoom + pan on static images/screenshots.

### D. Floating Callout Badge (`snippets/floating-badge.html`)
Animated badge with pulsing dot for feature highlights and metric callouts.

### E. Light-Sweep Transition (`snippets/light-sweep.html`) — USE THIS, not registry shaders
Brand-native transition between scene groups: a skewed teal-white gradient bar wipes across the full frame in ~1s, covering the cut. Start it ~0.1s before the old scene's fade ends; next scene enters ~0.15s after sweep start. Each sweep element needs its own unique track index (8, 9, ...).
**Do NOT install/embed registry transitions** (`light-leak`, `flash-through-white`, etc.) — they are full-frame 4-second DEMO compositions with their own backgrounds/fonts and clash with Verit branding mid-video.

### F. Micro-Effects (`snippets/micro-effects.html`)
- **Count-up number:** `tl.fromTo(el,{innerText:0},{innerText:200,duration:2,snap:{innerText:1},ease:"power1.inOut"},t)` — great for KPI/stat reveals.
- **Staggered grid pop-in:** children appear in RANDOM order with springy back-out: `stagger:{each:0.035,from:"random"}` — great for warehouses of units, product grids.
- **Highlight pulse:** after grid lands, pulse only highlighted items (`scale:1.12, yoyo, repeat:3, stagger:0.08`).

## Workflow (per video)

1. `mkdir ~/workspace/<name> && cd`; `npx --yes hyperframes@0.7.37 init . --non-interactive`
2. `cp /home/swastik/Downloads/repos/ank/ank-content/verit-video-kit/template.html index.html`; copy gen_voice.py + gen_bgm.py.
3. Write tight script → edit SCENES dict in gen_voice.py (~2.2 words/sec; 50s ≈ 110 words).
4. Run gen_voice.py with the venv python; then `gen_bgm.py <total_sec>` (needs scipy → venv python).
5. Assemble index.html: set #root data-duration; one `<audio id=vN>` per scene at computed offsets; paste snippet HTML before `<!-- SCENES_INJECT -->` and GSAP after `// ANIM_INJECT`; retime to offsets.
   - `parallax-text.html` for hook/key-message scenes; `screenshot-reveal.html` for product demos; `zoom-pan.html` for Ken Burns; `floating-badge.html` for callouts; `micro-effects.html` counters/grids; `light-sweep.html` between major acts.
6. Lint until 0 errors → render WITH the ffmpeg wrapper env var (see Key paths) → ffprobe audio stream exists.
7. Deliver each MP4 as `MEDIA:/abs/path`.

## Snippet Library (Complete)

| Snippet | Use for | Quality Level |
|---------|---------|---------------|
| `title.html` | Opening hook (bookend) | Standard |
| `cta.html` | Closing logo + tagline | Standard |
| `chain.html` | Amplifying stage-by-stage flow | Standard |
| `badges.html` | Consequences/benefits cards | Standard |
| `hub.html` | Connect/centralize/share diagram | Standard |
| `stat.html` | Big number count-up | Standard |
| `split.html` | Before/after comparison | Standard |
| `timeline.html` | Evolution/steps process | Standard |
| **`screenshot-reveal.html`** | Cinematic 3D camera on screenshots (NO shadows) | **UPGRADED** |
| **`parallax-text.html`** | 3D depth between text layers | **UPGRADED** |
| **`zoom-pan.html`** | Ken Burns camera movement | **UPGRADED** |
| **`floating-badge.html`** | Animated callout badges with pulse | **UPGRADED** |
| **`light-sweep.html`** | Brand-native scene transition | **UPGRADED** |
| **`micro-effects.html`** | Count-up numbers + staggered grid pop-in | **UPGRADED** |

## Pitfalls (learned — read before assembling)
- Large single write_file of a full composition times out — build HTML in chunks/parts and merge.
- Chatterbox venv is `.venv-chatterbox` INSIDE ank-content, NOT `~/workspace/tts-env`.
- Track-index convention: 0 voice, 1 bgm, 2–5 background, 8–9 sweeps, 10+ scene content. **Every clip element needs a UNIQUE track index — two elements sharing one index is a lint error.**
- Old scripts at ank-content/scripts/*.txt are narration source only — rewrite tighter.
- CSS `perspective` must be on the parent container, not the element being transformed.
- **Clip-Safe Exit Pattern:** NEVER animate/tween `visibility` or rely on fade-out alone on a `.clip` element — the framework manages clip visibility itself, and non-linear seeking can land after the exit tween leaving stale visible state (lint: `gsap_animates_clip_element` / `gsap_exit_missing_hard_kill`). Fix: wrap the scene's content in an inner NON-clip `<div>` inside the stage; target all enter/exit tweens AND a hard kill (`tl.set("<inner>",{opacity:0},<just-before-clip-end>)`) at that wrapper.
- **Audio renders fail silently without the wrapper** — always set `HYPERFRAMES_FFMPEG_PATH=/home/swastik/tools/ffmpeg-wrap` and verify the AAC stream with ffprobe.
- Registry transition compositions are standalone demos — never embed them mid-video; use `light-sweep.html`.
- gen_bgm.py needs scipy — run with the ank-content venv python, not system python.
- **NO drop-shadows on screenshot/product cards, ever** (user-locked). No `box-shadow` blur rings, no glow blobs near cards, no vignette behind cards. Solid 1px teal border + flat backdrop patch only.

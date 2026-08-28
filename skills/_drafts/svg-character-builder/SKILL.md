---
name: svg-character-builder
description: Design and build a reusable GSAP-rigged SVG character (named parts, documented pivots) with a live animated preview page and browser screenshot QA for video series kits.
---

<!-- reflection reason: Repeatable multi-step procedure (design rig -> write SVG with named parts/pivots -> generate GSAP preview -> browser QA screenshot -> iterate) that will recur for every supporting character and future series; complements but does not duplicate the existing svg-character-animation skill, which covers animating rigs rather than building them. -->

1. Agree on character concept with the user: name, age/vibe, signature prop, locked color palette (3-5 colors), and key outfit details before drawing.
2. Create the project kit directory, e.g. ~/workspace/<series>/kit/characters/.
3. Write the character as a single SVG (viewBox ~400x600) with a layered structure: back hair -> legs -> body -> arms -> accessories -> head group. Give every animatable part a stable id (#head, #arm_l, #arm_r, #lid_l, #lid_r, #mouth_smile, #mouth_open, plus props like #scarf_tail).
4. Document each part's GSAP pivot point (svgOrigin coordinates) in an SVG comment header so future compositions can animate without guessing.
5. Build blink capability as skin-colored eyelid rects (opacity 0) covering the eyes, and a hidden open-mouth group for talk/viseme swaps.
6. Generate a preview.html that inlines the SVG, loads GSAP from CDN, and runs an idle loop: breathing bob (y yoyo), head tilt (rotation with svgOrigin at neck), random blinking (1.8-4s delay), accessory sway, and one signature gesture (e.g., wave with back.out ease). Add a simple themed background for context.
7. Open the preview with browser_open (use a configured account, not 'default') and take a browser_screenshot; verify via browser_html that GSAP transforms are applied (data-svg-origin / matrix transforms present).
8. Show the user the screenshot and file paths; iterate on design feedback by editing the SVG (2-minute changes) and regenerating the preview.
9. Keep the final SVG as the single source of truth for the character so every episode reuses the identical file (guaranteed consistency).

---
name: video-metadata
description: Generate YouTube-style titles and descriptions for hyperframes explainer/story videos by extracting the actual scene copy from the project so metadata matches the video content exactly.
---

<!-- reflection reason: Repeatable multi-step procedure: extract scene copy from any hyperframes project, then apply a fixed title/description template with brand CTA and sources. Directly reusable for every future video the user renders. -->

1. Identify the target video project(s) under ~/workspace/<project-name> (e.g. stories-target, stories-kfc).
2. Extract the on-screen text from each project by parsing ~/workspace/<project>/index.html (regex for text nodes >25 chars, dedupe, HTML-unescape) to get the real hook line, key facts, lesson callout, Verit Analytics CTA line, and source attribution.
3. Confirm duration and render path from ~/workspace/<project>/renders/ if needed for description context.
4. For each video produce: 3 title options — option 1 must reuse the video's own opening hook line verbatim; options 2-3 are alternative angles (outcome-focused, curiosity-driven).
5. Write a description following this fixed structure: (a) 1-paragraph dramatic setup restating the story premise from scene copy, (b) paragraph detailing the failure mechanics using exact facts from the scenes, (c) '⚠️ The lesson:' line quoting the video's lesson card verbatim, (d) '📊 Verit Analytics' CTA paragraph matching the product feature shown in that specific video's closing card + 'Start free → veritanalytics.com', (e) 'Source:' line copying the video's source attribution, (f) relevant hashtags (#SupplyChain plus topic-specific tags).
6. Present all videos in one response with clear per-video sections; offer follow-ups: tags, thumbnail text overlays, or saving output as youtube.md in each project folder.

---
name: html-infographic-render
description: Render HTML/CSS infographics to tight PNGs using Playwright viewport resize
---

<!-- reflection reason: Multi-step workflow discovered through trial-and-error for creating social media infographics from HTML/CSS. The key insight — render at full height then resize viewport to actual content height before screenshotting — avoids both the flexbox footer-gap problem and the fixed-height empty-space problem. This pattern will recur whenever generating PNGs from HTML templates. -->

1. Write the infographic HTML with all content, ensuring body has no fixed height constraint (use min-height or none).
2. Render with Playwright: open page at viewport 1200x1600, wait for networkidle + 2s for fonts.
3. Measure actual content height: page.evaluate('document.body.scrollHeight').
4. Resize viewport to match: page.set_viewport_size({width: 1200, height: actual_content_height}).
5. Screenshot at the resized viewport: page.screenshot(path=output_path, full_page=False).
6. Verify output dimensions with `identify <file>` — height should match content height, no empty space below footer.
7. If using Buffer API, replace em-dash '—' with '--' (double hyphen) in all text to avoid rendering issues.

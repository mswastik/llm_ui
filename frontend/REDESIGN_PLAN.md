# Frontend Redesign Plan

## Problem Statement

The current frontend has ~4,640 lines of code for a relatively simple chat UI with basic styling.
The complexity is disproportionate to the visual output.

## Goals

1. **Reduce code by ~40%** while adding more features
2. **Modern design language** with dark/light mode, glassmorphism, smooth animations
3. **Better UX** with micro-interactions, keyboard shortcuts, smart input
4. **Simpler architecture** — fewer components, cleaner separation of concerns
5. **Zero backend changes** — reuse all existing API endpoints

## Design System

### Colors (CSS Custom Properties)
- Primary: indigo/violet gradient
- Background: slate gray (dark mode) / white (light mode)
- Surfaces: slightly elevated with subtle shadows
- Accents: green (success), red (error/warning), amber (tools)

### Typography
- System font stack (Inter-like)
- Monospace for code blocks
- Proper hierarchy: 12px → 14px → 16px → 20px → 24px → 32px

### Spacing
- 4px base unit (4, 8, 12, 16, 24, 32, 48, 64)

### Components
- Cards with rounded corners (12px)
- Subtle borders (1px, semi-transparent)
- Soft shadows (0 1px 3px rgba(0,0,0,0.1))
- Smooth transitions (150-300ms)

## File Structure (After)

```
frontend/
├── static/
│   ├── css/
│   │   ├── theme.css          # New unified theme (dark/light)
│   │   └── animations.css     # Keyframes and transitions
│   └── js/
│       ├── main.js            # Single entry point
│       ├── store.js           # Simplified stores (chat, ui)
│       ├── sse.js             # SSE service (keep)
│       ├── tts.js             # TTS service (keep)
│       └── components/
│           ├── chat.js        # Chat component (simplified)
│           ├── sidebar.js     # Sidebar component
│           └── settings.js    # Settings component (unified)
├── templates/
│   ├── base.html              # Updated base
│   ├── index.html             # Main layout (single page)
│   ├── agents.html            # Agents page (reuse layout)
│   └── knowledge.html         # Knowledge page (reuse layout)
```

## Phase Plan

### Phase 1: Foundation (Day 1)
- [x] Create new theme.css with dark/light mode
- [x] Create new base.html with Phosphor Icons CDN
- [x] Simplify store.js (merge 4 stores → 2)
- [x] Create new main.js entry point

### Phase 2: Layout & Sidebar (Day 1-2)
- [x] Create new index.html layout
- [x] Rewrite sidebar.html with search, animations
- [x] Implement compact/expanded sidebar with smooth transition

### Phase 3: Chat Area (Day 2-3)
- [x] Rewrite chat.html with modern message cards
- [x] Implement block system (content/thinking/tool) elegantly
- [x] Smart input area with pill toggles
- [x] Streaming indicator with typing animation

### Phase 4: Settings & Modals (Day 3)
- [x] Unified settings modal
- [x] Knowledge base modal
- [x] Toast notification system

### Phase 5: MCP Panel (Day 3-4)
- [x] Slide-in MCP panel
- [x] Server status indicators
- [x] Tool discovery display

### Phase 6: Polish (Day 4)
- [x] Keyboard shortcuts
- [x] Empty states
- [x] Responsive design
- [x] Performance optimizations

### Phase 7: Standalone Pages (Day 4-5)
- [x] Agents page (agents.html)
- [x] Knowledge page (knowledge.html)
- [x] Update backend routes if needed

## Migration Strategy

1. **Create new files alongside old ones** — no deletions initially
2. **Update base.html to load new assets** — keep old files as fallback
3. **Test all functionality** — send messages, settings, MCP, RAG, TTS
4. **Remove old files** once verified
5. **Backup before starting**

## Risk Mitigation

- Keep all backend API calls identical
- Keep Alpine.js (no framework change)
- Keep SSE streaming (no protocol change)
- Keep message block system (but render it better)
- All changes are frontend-only

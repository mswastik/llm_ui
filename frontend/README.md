# Frontend Modular Architecture

This document describes the refactored modular architecture for the Alpine.js frontend.

## Structure

```
frontend/
├── static/
│   ├── css/
│   │   └── styles.css              # Custom styles (Tailwind supplements)
│   └── js/
│       ├── app.js                  # Main entry point (ES module)
│       ├── store.js                # Alpine.store() definitions
│       ├── utils.js                # Shared utilities (formatters, API, helpers)
│       ├── components/
│       │   ├── chat.js             # Chat component (messaging, streaming, tool calls)
│       │   ├── sidebar.js          # Sidebar component (conversations)
│       │   ├── settings.js         # Settings component (app config, MCP servers)
│       │   └── documents.js        # Documents component (RAG knowledge base)
│       └── services/
│           ├── sse.js              # Server-Sent Events service
│           └── tts.js              # Text-to-Speech service
└── templates/
    ├── base.html                   # Base template (head, scripts, layout)
    ├── index.html                  # Main page (assembles partials)
    └── partials/
        ├── chat.html               # Chat area with messages and input
        ├── sidebar.html            # Sidebar with conversations
        ├── settings.html           # Settings modal
        └── documents.html          # Documents modal
```

## Key Concepts

### 1. Alpine.store() for Shared State

Centralized state management accessible across all components via `$store`:

```javascript
// Access in templates
<span x-text="$store.chat.messages.length"></span>
<button @click="$store.documents.show = true">Documents</button>

// Access in components
this.$store.chat.addMessage(msg)
this.$store.chat.showToast('Success!', 'success')
```

**Stores:**
- `$store.chat` - Conversations, messages, models, UI state
- `$store.settings` - App settings, MCP servers
- `$store.documents` - Document list
- `$store.tts` - TTS state

### 2. Modular Components (Alpine.data)

Each component is defined in its own file using `Alpine.data()`:

```javascript
// components/chat.js
document.addEventListener('alpine:init', () => {
  Alpine.data('chat', () => ({
    // State
    get messages() { return this.$store.chat.messages },
    
    // Methods
    async sendMessage(inputMessage) { ... },
    processStreamEvent(data, msgIndex) { ... }
  }))
})
```

**Components:**
- `chat` - Core chat functionality, streaming, tool call display
- `sidebar` - Conversation management
- `settings` - App configuration, MCP server management
- `documents` - Document upload/management

### 3. ES Modules

All JavaScript uses ES modules for proper dependency management:

```javascript
// Import utilities
import { api, helpers } from '../utils.js'
import { sseService } from '../services/sse.js'

// Export for external use
export { formatters, markdownUtils }
```

### 4. Jinja Partials

HTML is split into logical partials for maintainability:

```html
<!-- index.html -->
{% include "partials/sidebar.html" %}
{% include "partials/chat.html" %}
{% include "partials/settings.html" %}
```

### 5. Service Layer

Reusable services for cross-cutting concerns:

- **SSE Service** - Server-Sent Events streaming
- **TTS Service** - Text-to-Speech functionality

## Component Communication

### Via Store (Preferred)

```javascript
// In any component
this.$store.chat.addMessage(msg)
this.$store.chat.showToast('Done!', 'success')
```

### Via Computed Getters

```javascript
// Access store state as local properties
get messages() { return this.$store.chat.messages }
get isLoading() { return this.$store.chat.isLoading }
```

## Tool Call Display

The chat component handles tool calls with a clean block-based architecture:

```javascript
// Message structure with blocks
{
  role: 'assistant',
  content: 'Response text',
  tool_calls: [
    { type: 'thinking', content: '...' },
    { type: 'tool_call', name: 'search_web', arguments: {...}, result: {...} }
  ]
}
```

Each block type is rendered separately with expandable sections.

## Benefits

### Before Refactoring
- ❌ 818 lines of monolithic JS
- ❌ 853 lines of HTML with inline logic
- ❌ Difficult to debug tool calls
- ❌ No separation of concerns
- ❌ Hard to add new features

### After Refactoring
- ✅ Modular components (100-200 lines each)
- ✅ Clean HTML partials (200-300 lines each)
- ✅ Centralized state management
- ✅ Clear separation of concerns
- ✅ Easy to debug (console.log in specific components)
- ✅ Easy to extend (add new components)

## Adding New Features

### 1. New Component

```bash
# Create component file
touch frontend/static/js/components/myfeature.js
```

```javascript
// components/myfeature.js
document.addEventListener('alpine:init', () => {
  Alpine.data('myFeature', () => ({
    async init() { ... },
    async doSomething() { ... }
  }))
})
```

```javascript
// app.js - import it
import './components/myfeature.js'
```

### 2. New Store

```javascript
// store.js
Alpine.store('myFeature', {
  data: [],
  async loadData() { ... }
})
```

### 3. New Partial

```bash
# Create partial
touch frontend/templates/partials/myfeature.html
```

```html
<!-- myfeature.html -->
<div x-data="myFeature" x-init="init()">
  <!-- Feature UI -->
</div>
```

## Debugging Tips

### Component State

```javascript
// In component methods
console.log('[chat] Messages:', this.$store.chat.messages)
```

### Template Debugging

```html
<!-- Add temporary debug output -->
<pre x-text="JSON.stringify($store.chat.messages, null, 2)"></pre>
```

### SSE Events

```javascript
// In chat.js processStreamEvent
console.log('[DEBUG] processStreamEvent:', data.type, data)
```

## Migration Notes

### Breaking Changes
- All `x-data="chatApp()"` replaced with component-specific `x-data`
- Direct state access via `$store` instead of `this.`
- ES module imports instead of global script tags

### Script Loading

Old:
```html
<script src="/static/js/app.js"></script>
```

New:
```html
<script type="module" src="/static/js/app.js"></script>
```

## Performance

- **Bundle Size**: No increase (still using CDN for Tailwind/Alpine)
- **Load Time**: Same (modules load in parallel)
- **Runtime**: Improved (reactivity scoped to components)

## Future Improvements

1. **TypeScript** - Add type safety to components
2. **Build Step** - Use Vite for bundling and HMR
3. **Component Library** - Consider Headless UI for modals/dropdowns
4. **Testing** - Add Vitest + Testing Library for component tests

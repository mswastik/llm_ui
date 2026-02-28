# LLM UI - Frontend Architecture

A modern, modular frontend architecture for the LLM UI application.

## Overview

The frontend has been refactored from a monolithic structure into a clean, modular architecture using ES6 modules. This improves maintainability, testability, and code organization.

## Directory Structure

```
frontend/
├── static/
│   ├── css/
│   │   └── styles.css           # Custom Tailwind CSS styles
│   └── js/
│       ├── app.js               # Main entry point (Alpine.js application)
│       ├── state/               # State management modules
│       │   ├── chatState.js     # Chat-related state (conversations, messages)
│       │   ├── settingsState.js # Settings, MCP, documents state
│       │   └── uiState.js       # UI toggle states (sidebar, modals, expansions)
│       ├── services/            # Service layer (API calls, external services)
│       │   ├── api.js           # REST API abstraction
│       │   ├── sse.js           # Server-Sent Events streaming
│       │   └── tts.js           # Text-to-speech service
│       ├── utils/               # Utility functions
│       │   ├── markdown.js      # Markdown rendering with citations
│       │   ├── formatters.js    # Date, file size, text formatters
│       │   └── helpers.js       # Common helpers (copy, scroll, etc.)
│       └── components/          # UI component logic
│           └── chat.js          # Chat component streaming logic
└── templates/
    └── index.html               # Main HTML template (853 lines, down from 1334)
```

## Module Responsibilities

### State Management (`state/`)

**chatState.js** - Manages chat-related state:
- Conversations list and current conversation
- Messages array
- Loading states
- Tool execution status
- Model selection
- Web search and RAG toggles

**settingsState.js** - Manages application settings:
- General application settings
- MCP server configurations
- Document management
- TTS and SearXNG settings

**uiState.js** - Manages UI toggle states:
- Sidebar collapse/expand
- Modal visibility (settings, documents)
- Expansion states for collapsible sections (tool calls, thinking, sources)

### Services (`services/`)

**api.js** - REST API abstraction layer:
- `conversationsApi` - Conversation CRUD operations
- `messagesApi` - Message operations
- `modelsApi` - Model listing
- `documentsApi` - Document upload and management
- `settingsApi` - Settings management
- `mcpApi` - MCP server management
- `ttsApi` - Text-to-speech operations

**sse.js** - Server-Sent Events streaming:
- `SSEService` class for managing SSE connections
- Event handlers for streaming responses
- Support for both regular and regenerated responses

**tts.js** - Text-to-speech service:
- `TTSService` class for TTS operations
- Audio playback management
- Pause/resume functionality
- Loading state tracking

### Utilities (`utils/`)

**markdown.js** - Markdown rendering:
- `renderMarkdown()` - Basic markdown parsing
- `renderWithCitations()` - Markdown with citation links
- `getMessageSources()` - Extract sources from tool calls

**formatters.js** - Data formatting:
- `formatDate()` - Relative time formatting
- `formatFileSize()` - Human-readable file sizes
- `formatSources()` - Source formatting for display
- `stripMarkdown()` - Remove markdown syntax

**helpers.js** - Common utilities:
- `scrollToBottom()` - Scroll messages container
- `copyToClipboard()` - Cross-browser clipboard copy
- `generateId()` - Unique ID generation
- `toggleExpansion()` / `isExpanded()` - Expansion state management

### Components (`components/`)

**chat.js** - Chat component logic:
- `streamResponse()` - Initiate response streaming
- `processStreamEvent()` - Process SSE events
- Event handlers for different event types (content, thinking, tool calls)

## Main Application (`app.js`)

The main `app.js` file wires all modules together into a single Alpine.js application:

```javascript
function chatApp() {
    return {
        // State from modules
        conversations: chatState.conversations,
        messages: chatState.messages,
        // ... etc
        
        // Methods delegate to modules
        async sendMessage() {
            await chatState.sendMessage.call(chatState, ...);
        },
        // ... etc
    };
}
```

## Key Improvements

### Before Refactoring
- **index.html**: 1,334 lines (monolithic)
- **app.js**: 1,576 lines (single file)
- **Total**: ~2,910 lines in 2 files

### After Refactoring
- **index.html**: 853 lines (36% reduction)
- **JavaScript**: ~1,800 lines across 11 modules
- **Better organization**: Clear separation of concerns
- **Reduced duplication**: ~40% less redundant code

## Benefits

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Isolated modules are easier to test
3. **Reusability**: Utility functions can be reused across modules
4. **Scalability**: New features can be added as new modules
5. **Debugging**: Easier to locate and fix issues
6. **Onboarding**: Clear structure helps new developers understand the codebase

## Usage

### Adding a New API Endpoint

1. Add to `services/api.js`:
```javascript
export const newApi = {
    ...apiBase,
    async getSomething(id) {
        return this.get(`/api/something/${id}`);
    }
};
```

2. Use in state management:
```javascript
import { newApi } from '../services/api.js';

async loadSomething(id) {
    const data = await newApi.getSomething(id);
    // ... update state
}
```

### Adding a New UI Toggle

1. Add state to `state/uiState.js`:
```javascript
export const uiState = {
    expandedItems: {},
    
    toggleItem(id) {
        helpers.toggleExpansion(this.expandedItems, id);
    },
    
    isItemExpanded(id) {
        return helpers.isExpanded(this.expandedItems, id);
    }
};
```

2. Wire in `app.js`:
```javascript
toggleItem(id) {
    uiState.toggleItem.call(uiState, id);
}
```

## Browser Compatibility

The modular structure uses ES6 modules which are supported in all modern browsers:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+

The `type="module"` script tag is used in `index.html`:
```html
<script type="module" src="/static/js/app.js"></script>
```

## Development Tips

1. **Import/Export**: Always use named exports for better tree-shaking
2. **State Management**: Keep state in `state/` modules, not components
3. **API Calls**: Always go through `services/api.js` for consistency
4. **Utilities**: Put reusable functions in `utils/`
5. **Testing**: Write unit tests for utility functions first

## Migration Notes

If you're updating from the old monolithic structure:

1. All function names remain the same in the public API
2. State properties are accessed the same way in templates
3. Internal implementation is now modular
4. No changes required to backend code

## Future Enhancements

Potential improvements for the future:

1. **TypeScript**: Add type safety with TypeScript
2. **Build Process**: Add bundling with Vite or Webpack
3. **Component Library**: Extract reusable UI components
4. **State Persistence**: Add localStorage for UI preferences
5. **Testing Framework**: Add Jest or Vitest for unit tests

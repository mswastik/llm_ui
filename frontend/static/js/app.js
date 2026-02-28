/**
 * Main Alpine.js Application
 * All functionality in a single file for reliable Alpine.js initialization
 */

// Utility functions
const formatters = {
    formatDate(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
        return date.toLocaleDateString();
    },
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },
    stripMarkdown(text) {
        if (!text) return '';
        text = text.replace(/^#{1,6}\s+/gm, '');
        text = text.replace(/\*\*(.*?)\*\*/g, '$1');
        text = text.replace(/\*(.*?)\*/g, '$1');
        text = text.replace(/`(.*?)`/g, '$1');
        text = text.replace(/```[\s\S]*?```/g, '');
        text = text.replace(/\[\d+\]/g, '');
        text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
        text = text.replace(/\n{3,}/g, '\n\n');
        return text.trim();
    }
};

const markdownUtils = {
    render(text) {
        if (!text) return '';
        return marked.parse(text);
    },
    renderWithCitations(text, sources = []) {
        if (!text) return '';
        let html = this.render(text);
        if (sources && sources.length > 0) {
            html = html.replace(/\[(\d+)\]/g, (match, num) => {
                const index = parseInt(num) - 1;
                if (index >= 0 && index < sources.length) {
                    const source = sources[index];
                    const url = source.url || '#';
                    const title = source.title || 'Source';
                    return `<sup><a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-link" title="${title}">[${num}]</a></sup>`;
                }
                return match;
            });
        }
        return html;
    },
    getMessageSources(message) {
        if (!message?.tool_calls || message.tool_calls.length === 0) return [];
        const allSources = [];
        message.tool_calls.forEach(toolCall => {
            if (toolCall.result?.sources) {
                toolCall.result.sources.forEach(source => {
                    if (!allSources.some(s => s.url === source.url)) {
                        allSources.push(source);
                    }
                });
            }
        });
        return allSources;
    }
};

const helpers = {
    scrollToBottom(container) {
        if (container) container.scrollTop = container.scrollHeight;
    },
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (error) {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                document.body.removeChild(textArea);
                return true;
            } catch (err) {
                document.body.removeChild(textArea);
                return false;
            }
        }
    },
    generateId() { return Date.now(); },
    toggleExpansion(state, key) { state[key] = !state[key]; return state[key]; },
    isExpanded(state, key) { return state[key] === true; },
    createExpansionKey(id, index) { return `${id}-${index}`; }
};

// API helpers
const api = {
    async get(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error('API Error');
        return response.json();
    },
    async post(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'API Error');
        }
        return response.json();
    },
    async put(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error('API Error');
        return response.json();
    },
    async delete(endpoint) {
        const response = await fetch(endpoint, { method: 'DELETE' });
        if (!response.ok) throw new Error('API Error');
        return response.json();
    }
};

// SSE Service
class SSEService {
    constructor() {
        this.eventSource = null;
        this.streamingConversationId = null;
    }
    stream(requestId, conversationId, options = {}) {
        let url = `/api/stream/${requestId}?conversation_id=${conversationId}`;
        if (options.enableWebSearch) url += '&enable_web_search=true';
        if (options.enableRag) url += '&enable_rag=true';
        if (options.model) url += `&model=${encodeURIComponent(options.model)}`;
        this.streamingConversationId = conversationId;
        this.eventSource = new EventSource(url);
        return this.setupListeners();
    }
    setupListeners() {
        let streamCompleted = false;
        return {
            onData: (handler) => {
                this.eventSource.onmessage = (event) => {
                    try {
                        handler(JSON.parse(event.data));
                    } catch (error) {
                        console.error('Error parsing SSE:', error);
                    }
                };
            },
            onError: (handler) => {
                this.eventSource.onerror = () => {
                    if (!streamCompleted) {
                        streamCompleted = true;
                        this.close();
                    }
                };
            }
        };
    }
    close() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
}

const sseService = new SSEService();

// TTS Service
class TTSService {
    constructor() {
        this.currentAudio = null;
        this.currentAudioMessageId = null;
        this.isPlaying = false;
        this.ttsAvailable = false;
    }
    async checkAvailability() {
        try {
            const data = await api.get('/api/tts/status');
            this.ttsAvailable = data.available;
        } catch (error) {
            this.ttsAvailable = false;
        }
        return this.ttsAvailable;
    }
    async speak(message, onError) {
        if (!this.ttsAvailable) {
            onError?.('TTS not available');
            return false;
        }
        if (this.currentAudioMessageId === message.id && this.currentAudio) {
            return this.togglePlayPause();
        }
        this.stop();
        let text = message.content.replace(/<[^>]*>/g, '').trim();
        text = formatters.stripMarkdown(text);
        if (!text) {
            onError?.('No text to speak');
            return false;
        }
        try {
            const data = await api.post('/api/tts/generate', { text });
            if (data.success && data.audio_url) {
                await this.playAudio(data.audio_url, message.id);
                return true;
            }
        } catch (error) {
            onError?.('TTS failed');
        }
        return false;
    }
    async playAudio(audioUrl, messageId) {
        return new Promise((resolve, reject) => {
            this.currentAudio = new Audio(audioUrl);
            this.currentAudioMessageId = messageId;
            this.isPlaying = true;
            this.currentAudio.onended = () => this.cleanup();
            this.currentAudio.onpause = () => { this.isPlaying = false; };
            this.currentAudio.onplay = () => { this.isPlaying = true; };
            this.currentAudio.onerror = () => { this.cleanup(); reject(new Error('Audio error')); };
            this.currentAudio.play().catch(reject);
        });
    }
    togglePlayPause() {
        if (!this.currentAudio) return false;
        if (this.isPlaying) {
            this.currentAudio.pause();
            this.isPlaying = false;
        } else {
            this.currentAudio.play();
            this.isPlaying = true;
        }
        return this.isPlaying;
    }
    stop() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.cleanup();
        }
    }
    cleanup() {
        this.currentAudio = null;
        this.currentAudioMessageId = null;
        this.isPlaying = false;
    }
}

const ttsService = new TTSService();

// Main Alpine.js Application
function chatApp() {
    return {
        // State
        conversations: [],
        currentConversationId: null,
        currentConversationTitle: 'New Chat',
        messages: [],
        inputMessage: '',
        isLoading: false,
        toolStatus: { active: false, tool: '', status: '', progress: null, data: null },
        enableWebSearch: false,
        enableRAG: false,
        availableModels: [],
        selectedModel: '',
        editingMessageId: null,
        editContent: '',
        sidebarCollapsed: false,
        showSettings: false,
        showDocuments: false,
        activeSettingsTab: 'general',
        expandedToolCalls: {},
        expandedToolCallBlocks: {},
        expandedThinkingBlocks: {},
        expandedThinking: {},
        expandedContentBlocks: {},
        expandedSources: {},
        settings: {},
        documents: [],
        mcpServers: [],
        mcpTools: [],
        newServer: { name: '', transport_type: 'stdio', command: '', args: '[]', url: '' },
        ttsLoading: {},
        currentAudio: null,
        currentAudioMessageId: null,
        ttsAvailable: false,
        isPlaying: false,
        toast: { show: false, message: '', type: 'success' },

        // Initialization
        async init() {
            await this.loadConversations();
            await this.loadSettings();
            await this.loadMCPServers();
            await this.loadModels();
            await this.loadDocuments();
            await ttsService.checkAvailability();
            this.ttsAvailable = ttsService.ttsAvailable;
            if (this.conversations.length === 0) {
                await this.createNewConversation();
            } else {
                await this.loadConversation(this.conversations[0].id);
            }
        },

        // Conversations
        async loadConversations() {
            try {
                const data = await api.get('/api/conversations');
                this.conversations = data.conversations;
            } catch (error) { console.error('Error loading conversations:', error); }
        },
        async createNewConversation() {
            try {
                const data = await api.post('/api/conversations', { title: 'New Chat' });
                this.conversations.unshift(data.conversation);
                await this.loadConversation(data.conversation.id);
            } catch (error) { console.error('Error creating conversation:', error); }
        },
        async loadConversation(conversationId) {
            sseService.close();
            this.isLoading = false;
            this.toolStatus.active = false;
            try {
                const data = await api.get(`/api/conversations/${conversationId}`);
                this.currentConversationId = conversationId;
                this.currentConversationTitle = data.conversation.title;
                this.messages = data.messages;
                this.$nextTick(() => helpers.scrollToBottom(this.$refs?.messagesContainer));
            } catch (error) { console.error('Error loading conversation:', error); }
        },
        async deleteConversation(conversationId, event) {
            event?.stopPropagation();
            if (!confirm('Delete this conversation?')) return;
            try {
                await api.delete(`/api/conversations/${conversationId}`);
                this.conversations = this.conversations.filter(c => c.id !== conversationId);
                if (conversationId === this.currentConversationId) {
                    if (this.conversations.length > 0) {
                        await this.loadConversation(this.conversations[0].id);
                    } else {
                        await this.createNewConversation();
                    }
                }
            } catch (error) { console.error('Error deleting conversation:', error); }
        },
        async updateConversationTitle(conversationId, event) {
            const newTitle = event.target.value.trim();
            if (!newTitle) return;
            try {
                await api.put(`/api/conversations/${conversationId}`, { title: newTitle });
                const conv = this.conversations.find(c => c.id === conversationId);
                if (conv) conv.title = newTitle;
                if (conversationId === this.currentConversationId) {
                    this.currentConversationTitle = newTitle;
                }
            } catch (error) { console.error('Error updating title:', error); }
        },

        // Messages
        async sendMessage() {
            if (!this.inputMessage?.trim() || this.isLoading) return;
            const messageText = this.inputMessage.trim();
            this.inputMessage = '';
            this.isLoading = true;
            const userMessage = {
                id: helpers.generateId(),
                role: 'user',
                content: messageText,
                created_at: new Date().toISOString()
            };
            this.messages.push(userMessage);
            try {
                const data = await api.post(`/api/conversations/${this.currentConversationId}/messages`, {
                    message: messageText,
                    enable_web_search: this.enableWebSearch,
                    enable_rag: this.enableRAG
                });
                await this.streamResponse(data.request_id);
            } catch (error) {
                console.error('Error sending message:', error);
                this.isLoading = false;
            }
        },
        async streamResponse(requestId) {
            const options = {
                enableWebSearch: this.enableWebSearch,
                enableRag: this.enableRAG,
                model: this.selectedModel
            };
            const handlers = sseService.stream(requestId, this.currentConversationId, options);
            const assistantMessage = {
                id: helpers.generateId() + 1,
                role: 'assistant',
                content: '',
                thinking: '',
                tool_calls: [],
                created_at: new Date().toISOString()
            };
            this.messages.push(assistantMessage);
            const msgIndex = this.messages.length - 1;
            handlers.onData((data) => {
                this.processStreamEvent(data, assistantMessage, msgIndex);
            });
        },
        processStreamEvent(data, assistantMessage, msgIndex) {
            console.log('[DEBUG] processStreamEvent received:', data.type, data);
            console.log('[DEBUG] Current tool_calls array:', this.messages[msgIndex].tool_calls);
            console.log('[DEBUG] editingMessageId:', this.editingMessageId, 'message.id:', this.messages[msgIndex].id);
            
            switch (data.type) {
                case 'content':
                    console.log('[DEBUG] Content received:', data.content.substring(0, 50));
                    this.messages[msgIndex].content += data.content;
                    break;
                case 'thinking':
                    console.log('[DEBUG] Thinking received:', data.content.substring(0, 50));
                    const toolCalls = this.messages[msgIndex].tool_calls;
                    if (toolCalls.length > 0 && toolCalls[toolCalls.length - 1].type === 'thinking') {
                        toolCalls[toolCalls.length - 1].content += data.content;
                        console.log('[DEBUG] Appended to existing thinking block, length:', toolCalls[toolCalls.length - 1].content.length);
                    } else {
                        toolCalls.push({ type: 'thinking', content: data.content });
                        console.log('[DEBUG] Created new thinking block, length:', data.content.length);
                    }
                    // Replace entire message object to trigger Alpine reactivity
                    this.messages[msgIndex] = { ...this.messages[msgIndex], tool_calls: [...toolCalls] };
                    console.log('[DEBUG] Updated tool_calls after thinking:', this.messages[msgIndex].tool_calls);
                    console.log('[DEBUG] tool_calls.length:', this.messages[msgIndex].tool_calls.length);
                    break;
                case 'tool_call_start':
                    console.log('[DEBUG] Tool call start:', data.tool, data.args);
                    this.toolStatus.active = true;
                    this.toolStatus.tool = data.tool;
                    this.toolStatus.status = 'Starting...';
                    const newToolCall = {
                        type: 'tool_call',
                        name: data.tool,
                        arguments: data.args,
                        status: 'starting',
                        progress: 0,
                        result: null,
                        progress_history: [{ status: 'starting', progress: 0, timestamp: new Date().toISOString() }]
                    };
                    const currentToolCalls = this.messages[msgIndex].tool_calls;
                    currentToolCalls.push(newToolCall);
                    // Replace entire message object to trigger Alpine reactivity
                    this.messages[msgIndex] = { ...this.messages[msgIndex], tool_calls: [...currentToolCalls] };
                    console.log('[DEBUG] Added tool call, tool_calls now:', this.messages[msgIndex].tool_calls);
                    console.log('[DEBUG] tool_calls.length after add:', this.messages[msgIndex].tool_calls.length);
                    break;
                case 'tool_progress':
                    console.log('[DEBUG] Tool progress:', data.tool, data.status, data.progress);
                    this.toolStatus.status = data.status;
                    this.toolStatus.progress = data.progress || null;
                    const toolCallsForProgress = this.messages[msgIndex].tool_calls;
                    const currentToolCall = toolCallsForProgress.find(tc => tc.type === 'tool_call' && tc.status !== 'completed' && tc.status !== 'error');
                    console.log('[DEBUG] Found currentToolCall:', currentToolCall);
                    if (currentToolCall) {
                        currentToolCall.status = data.status;
                        currentToolCall.progress = data.progress || 0;
                        if (data.result) {
                            currentToolCall.result = data.result;
                            currentToolCall.status = 'completed';
                            console.log('[DEBUG] Tool completed:', data.tool);
                        }
                        // Replace entire message object to trigger Alpine reactivity
                        this.messages[msgIndex] = { ...this.messages[msgIndex], tool_calls: [...toolCallsForProgress] };
                        console.log('[DEBUG] Updated tool call, tool_calls:', this.messages[msgIndex].tool_calls);
                    }
                    if (data.result) this.toolStatus.active = false;
                    break;
                case 'error':
                    console.log('[DEBUG] Error:', data.error);
                    this.toolStatus.active = false;
                    this.isLoading = false;
                    this.messages[msgIndex].content += `\n\n❌ Error: ${data.error}`;
                    break;
                case 'title_update':
                    console.log('[DEBUG] Title update:', data.title);
                    this.currentConversationTitle = data.title;
                    const convIndex = this.conversations.findIndex(c => c.id === this.currentConversationId);
                    if (convIndex !== -1) this.conversations[convIndex].title = data.title;
                    break;
                case 'done':
                    console.log('[DEBUG] Stream done');
                    console.log('[DEBUG] Final tool_calls:', this.messages[msgIndex].tool_calls);
                    console.log('[DEBUG] tool_calls.length:', this.messages[msgIndex].tool_calls.length);
                    console.log('[DEBUG] Has tool_call block:', this.messages[msgIndex].tool_calls.some(tc => tc.type === 'tool_call'));
                    sseService.close();
                    this.isLoading = false;
                    this.toolStatus.active = false;
                    break;
                default:
                    console.log('[DEBUG] Unknown event type:', data.type);
            }
            // Force Alpine reactivity by replacing the entire messages array
            this.messages = [...this.messages];
            this.$nextTick(() => helpers.scrollToBottom(this.$refs?.messagesContainer));
        },
        async deleteMessage(messageId, event) {
            event?.stopPropagation();
            if (!confirm('Delete this message?')) return;
            try {
                await api.delete(`/api/messages/${messageId}`);
                this.messages = this.messages.filter(m => m.id !== messageId);
            } catch (error) { console.error('Error deleting message:', error); }
        },
        startEditMessage(messageId, content) {
            this.editingMessageId = messageId;
            this.editContent = content;
        },
        cancelEdit() {
            this.editingMessageId = null;
            this.editContent = '';
        },
        async saveEdit(messageId) {
            if (!this.editContent.trim()) { this.cancelEdit(); return; }
            const msg = this.messages.find(m => m.id === this.editingMessageId);
            if (!msg) { this.cancelEdit(); return; }
            if (msg.role === 'user') {
                if (this.editContent.trim() !== msg.content.trim()) {
                    await this.forkConversation(this.editingMessageId, this.editContent.trim());
                } else {
                    this.cancelEdit();
                }
                return;
            }
            try {
                const data = await api.put(`/api/messages/${this.editingMessageId}`, { content: this.editContent });
                msg.content = data.message.content;
            } catch (error) { console.error('Error updating message:', error); }
            this.cancelEdit();
        },
        async forkConversation(originalMessageId, newContent) {
            try {
                const data = await api.post('/api/conversations', { title: 'Forked: ' + newContent.substring(0, 30) + '...' });
                const newConversationId = data.conversation.id;
                this.conversations.unshift(data.conversation);
                await api.post(`/api/conversations/${newConversationId}/messages`, { message: newContent });
                const streamData = await api.post(`/api/conversations/${newConversationId}/messages`, { message: newContent });
                this.currentConversationId = newConversationId;
                this.currentConversationTitle = data.conversation.title;
                this.messages = [{ id: helpers.generateId(), role: 'user', content: newContent, created_at: new Date().toISOString() }];
                await this.streamResponse(streamData.request_id);
            } catch (error) { console.error('Error forking:', error); }
            this.cancelEdit();
        },
        async regenerateResponse(messageId) {
            if (this.isLoading) return;
            this.isLoading = true;
            try {
                const data = await api.post(`/api/conversations/${this.currentConversationId}/regenerate`, { message_id: messageId });
                const msgIndex = this.messages.findIndex(m => m.id === messageId);
                if (msgIndex !== -1) this.messages = this.messages.slice(0, msgIndex + 1);
                const handlers = sseService.stream(data.request_id, this.currentConversationId, { model: this.selectedModel });
                const assistantMessage = { id: helpers.generateId() + 1, role: 'assistant', content: '', thinking: '', tool_calls: [], created_at: new Date().toISOString() };
                this.messages.push(assistantMessage);
                handlers.onData((d) => this.processStreamEvent(d, assistantMessage, this.messages.length - 1));
            } catch (error) {
                console.error('Error regenerating:', error);
                this.isLoading = false;
            }
        },
        cancelRequest() {
            sseService.close();
            this.isLoading = false;
            this.toolStatus.active = false;
            const lastMessage = this.messages[this.messages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.content.trim() === '') {
                lastMessage.content = '⚠️ Request cancelled.';
            }
            this.showToast('Request cancelled', 'info');
        },

        // Models
        async loadModels() {
            try {
                const data = await api.get('/api/models');
                this.availableModels = data.models || [];
                const savedModel = localStorage.getItem('selectedModel');
                if (savedModel && this.availableModels.some(m => m.id === savedModel)) {
                    this.selectedModel = savedModel;
                } else if (this.availableModels.length > 0) {
                    this.selectedModel = this.availableModels[0].id;
                }
            } catch (error) { console.error('Error loading models:', error); }
        },
        updateSelectedModel() {
            if (this.selectedModel) localStorage.setItem('selectedModel', this.selectedModel);
        },

        // Documents
        async loadDocuments() {
            try {
                const data = await api.get('/api/documents');
                this.documents = data.documents;
            } catch (error) { console.error('Error loading documents:', error); }
        },
        async uploadDocument(file) {
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await fetch('/api/documents/upload', { method: 'POST', body: formData });
                if (response.ok) {
                    const data = await response.json();
                    this.documents.unshift(data.document);
                    this.showToast('Document uploaded!', 'success');
                } else {
                    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
                    this.showToast(`Error: ${error.detail}`, 'error');
                }
            } catch (error) {
                console.error('Error uploading:', error);
                this.showToast('Upload failed', 'error');
            }
        },
        async deleteDocument(documentId) {
            if (!confirm('Delete this document?')) return;
            try {
                await api.delete(`/api/documents/${documentId}`);
                this.documents = this.documents.filter(d => d.id !== documentId);
            } catch (error) { console.error('Error deleting document:', error); }
        },

        // Settings
        async loadSettings() {
            try {
                const data = await api.get('/api/settings');
                this.settings = data;
            } catch (error) { console.error('Error loading settings:', error); }
        },
        async saveSettings() {
            try {
                const data = await api.put('/api/settings', this.settings);
                this.settings = data;
                this.showToast('Settings saved!', 'success');
                this.showSettings = false;
            } catch (error) {
                this.showToast('Error saving settings', 'error');
            }
        },
        onTtsEngineChange() {
            if (this.settings.tts_engine === 'kokoro') {
                this.settings.tts_voice = 'af_bella';
                this.settings.kokoro_lang = 'a';
                if (!this.settings.kokoro_device) this.settings.kokoro_device = 'cpu';
            } else if (this.settings.tts_engine === 'edge-tts') {
                this.settings.tts_voice = 'en-IN-NeerjaNeural';
            } else if (this.settings.tts_engine === 'pyttsx3') {
                this.settings.tts_voice = '';
            }
        },

        // MCP
        async loadMCPServers() {
            try {
                const data = await api.get('/api/mcp/servers');
                this.mcpServers = data.servers;
            } catch (error) { console.error('Error loading MCP servers:', error); }
        },
        async loadMCPTools() {
            try {
                const data = await api.get('/api/mcp/tools');
                this.mcpTools = data.tools || [];
            } catch (error) { console.error('Error loading MCP tools:', error); }
        },
        async addMCPServer() {
            try {
                let args = [];
                try { args = JSON.parse(this.newServer.args); } catch (e) {
                    this.showToast('Invalid JSON for args', 'error');
                    return;
                }
                if (this.newServer.transport_type !== 'stdio' && !this.newServer.url) {
                    this.showToast('URL required for SSE/HTTP', 'error');
                    return;
                }
                if (this.newServer.transport_type === 'stdio' && !this.newServer.command) {
                    this.showToast('Command required for stdio', 'error');
                    return;
                }
                await api.post('/api/mcp/servers', {
                    name: this.newServer.name,
                    transport_type: this.newServer.transport_type,
                    command: this.newServer.command,
                    args: args,
                    env: {},
                    url: this.newServer.url || null
                });
                await this.loadMCPServers();
                await this.loadMCPTools();
                this.newServer = { name: '', transport_type: 'stdio', command: '', args: '[]', url: '' };
                this.showToast('Server added!', 'success');
            } catch (error) {
                this.showToast('Error adding server', 'error');
            }
        },
        async removeMCPServer(serverName) {
            if (!confirm(`Remove "${serverName}"?`)) return;
            try {
                await api.delete(`/api/mcp/servers/${serverName}`);
                await this.loadMCPServers();
                await this.loadMCPTools();
                this.showToast('Server removed', 'success');
            } catch (error) { this.showToast('Error removing server', 'error'); }
        },
        async reconnectMCPServer(serverName) {
            try {
                await api.post(`/api/mcp/servers/${serverName}/reconnect`);
                await this.loadMCPServers();
                await this.loadMCPTools();
                this.showToast('Reconnected', 'success');
            } catch (error) { this.showToast('Error reconnecting', 'error'); }
        },
        async refreshMCPServerTools(serverName) {
            try {
                await api.post(`/api/mcp/servers/${serverName}/refresh`);
                await this.loadMCPServers();
                await this.loadMCPTools();
                this.showToast('Tools refreshed', 'success');
            } catch (error) { this.showToast('Error refreshing', 'error'); }
        },

        // Utilities
        renderMarkdown(text) { return markdownUtils.render(text); },
        renderMarkdownWithCitations(text, sources) { return markdownUtils.renderWithCitations(text, sources); },
        getMessageSources(message) { return markdownUtils.getMessageSources(message); },
        formatDate: (isoString) => formatters.formatDate(isoString),
        formatFileSize: (bytes) => formatters.formatFileSize(bytes),
        scrollToBottom() { helpers.scrollToBottom(this.$refs?.messagesContainer); },
        async copyMessage(message) {
            const type = message.role === 'user' ? 'Query' : 'Response';
            const success = await helpers.copyToClipboard(message.content);
            this.showToast(success ? `${type} copied!` : 'Copy failed', success ? 'success' : 'error');
        },
        showToast(message, type = 'success') {
            this.toast.message = message;
            this.toast.type = type;
            this.toast.show = true;
            setTimeout(() => { this.toast.show = false; }, 2500);
        },
        // Check if tool calls should be displayed
        shouldShowToolCalls(message) {
            const hasToolCalls = message?.tool_calls && message.tool_calls.length > 0;
            const notEditing = this.editingMessageId !== message.id;
            console.log('[DEBUG] shouldShowToolCalls:', { 
                messageId: message.id, 
                hasToolCalls, 
                toolCallsLength: message.tool_calls?.length,
                notEditing,
                editingMessageId: this.editingMessageId,
                result: hasToolCalls && notEditing
            });
            return hasToolCalls && notEditing;
        },

        // UI Toggles
        toggleSidebar() { this.sidebarCollapsed = !this.sidebarCollapsed; },
        toggleToolCallBlock(messageId, blockIndex) {
            helpers.toggleExpansion(this.expandedToolCallBlocks, helpers.createExpansionKey(messageId, blockIndex));
        },
        isToolCallBlockExpanded(messageId, blockIndex) {
            return helpers.isExpanded(this.expandedToolCallBlocks, helpers.createExpansionKey(messageId, blockIndex));
        },
        toggleThinkingBlock(messageId, blockIndex) {
            helpers.toggleExpansion(this.expandedThinkingBlocks, helpers.createExpansionKey(messageId, blockIndex));
        },
        isThinkingBlockExpanded(messageId, blockIndex) {
            return helpers.isExpanded(this.expandedThinkingBlocks, helpers.createExpansionKey(messageId, blockIndex));
        },
        toggleThinking(messageId) { helpers.toggleExpansion(this.expandedThinking, messageId); },
        isThinkingExpanded(messageId) { return helpers.isExpanded(this.expandedThinking, messageId); },
        toggleSources(messageId) { helpers.toggleExpansion(this.expandedSources, messageId); },
        isSourcesExpanded(messageId) { return helpers.isExpanded(this.expandedSources, messageId); },

        // TTS
        async speakMessage(message) {
            const success = await ttsService.speak(message, (error) => this.showToast(error, 'error'));
            if (success) {
                this.currentAudio = ttsService.currentAudio;
                this.currentAudioMessageId = ttsService.currentAudioMessageId;
                this.isPlaying = ttsService.isPlaying;
            }
        },
        stopAudio() {
            ttsService.stop();
            this.currentAudio = null;
            this.currentAudioMessageId = null;
            this.isPlaying = false;
        },
        stripMarkdown: (text) => formatters.stripMarkdown(text)
    };
}

// Make globally available for Alpine.js
window.chatApp = chatApp;

function chatApp() {
    return {
        // State
        conversations: [],
        currentConversationId: null,
        currentConversationTitle: 'New Chat',
        messages: [],
        inputMessage: '',
        isLoading: false,
        showSettings: false,
        
        // Sidebar state
        sidebarCollapsed: false,
        
        // Tool toggles
        enableWebSearch: false,
        enableRAG: false,
        
        // Model selection
        availableModels: [],
        selectedModel: '',
        
        // Editing state
        editingMessageId: null,
        editContent: '',
        
        // Tool execution status
        toolStatus: {
            active: false,
            tool: '',
            status: '',
            progress: null,
            data: null
        },
        
        // MCP Servers
        mcpServers: [],
        newServer: {
            name: '',
            command: '',
            args: '[]'
        },
        
        // Documents
        documents: [],
        showDocuments: false,
        
        // Event Source for SSE
        eventSource: null,
        
        // Toast notification
        toast: {
            show: false,
            message: '',
            type: 'success'
        },
        
        // Expanded tool calls tracking (by message id)
        expandedToolCalls: {},
        
        // Initialize
        async init() {
            await this.loadConversations();
            await this.loadMCPServers();
            await this.loadModels();
            await this.loadDocuments();
            
            // Create a new conversation if none exist
            if (this.conversations.length === 0) {
                await this.createNewConversation();
            } else {
                // Load the most recent conversation
                await this.loadConversation(this.conversations[0].id);
            }
        },
        
        // Conversation Management
        async loadConversations() {
            try {
                const response = await fetch('/api/conversations');
                const data = await response.json();
                this.conversations = data.conversations;
            } catch (error) {
                console.error('Error loading conversations:', error);
            }
        },
        
        async createNewConversation() {
            try {
                const response = await fetch('/api/conversations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: 'New Chat' })
                });
                const data = await response.json();
                
                this.conversations.unshift(data.conversation);
                await this.loadConversation(data.conversation.id);
            } catch (error) {
                console.error('Error creating conversation:', error);
            }
        },
        
        async loadConversation(conversationId) {
            try {
                const response = await fetch(`/api/conversations/${conversationId}`);
                const data = await response.json();
                
                this.currentConversationId = conversationId;
                this.currentConversationTitle = data.conversation.title;
                this.messages = data.messages;
                
                // Scroll to bottom
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                console.error('Error loading conversation:', error);
            }
        },
        
        async deleteConversation(conversationId, event) {
            event.stopPropagation();
            if (!confirm('Are you sure you want to delete this conversation?')) return;
            
            try {
                const response = await fetch(`/api/conversations/${conversationId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    // Remove from list
                    this.conversations = this.conversations.filter(c => c.id !== conversationId);
                    
                    // If deleted current conversation, load another or create new
                    if (conversationId === this.currentConversationId) {
                        if (this.conversations.length > 0) {
                            await this.loadConversation(this.conversations[0].id);
                        } else {
                            await this.createNewConversation();
                        }
                    }
                }
            } catch (error) {
                console.error('Error deleting conversation:', error);
            }
        },
        
        async updateConversationTitle(conversationId, event) {
            const newTitle = event.target.value.trim();
            if (!newTitle) return;
            
            try {
                await fetch(`/api/conversations/${conversationId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: newTitle })
                });
                
                // Update local state
                const conv = this.conversations.find(c => c.id === conversationId);
                if (conv) {
                    conv.title = newTitle;
                }
                if (conversationId === this.currentConversationId) {
                    this.currentConversationTitle = newTitle;
                }
            } catch (error) {
                console.error('Error updating conversation title:', error);
            }
        },
        
        // Message Handling
        async sendMessage() {
            if (!this.inputMessage.trim() || this.isLoading) return;
            
            const messageText = this.inputMessage.trim();
            this.inputMessage = '';
            this.isLoading = true;
            
            // Add user message to UI immediately
            const userMessage = {
                id: Date.now(),
                role: 'user',
                content: messageText,
                created_at: new Date().toISOString()
            };
            this.messages.push(userMessage);
            
            this.scrollToBottom();
            
            try {
                // Build request with tool options
                const requestBody = {
                    message: messageText,
                    enable_web_search: this.enableWebSearch,
                    enable_rag: this.enableRAG
                };
                
                // Send message to backend
                const response = await fetch(`/api/conversations/${this.currentConversationId}/messages`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                const requestId = data.request_id;
                
                // Connect to SSE stream for real-time updates
                await this.streamResponse(requestId, data.enable_web_search, data.enable_rag);
                
            } catch (error) {
                console.error('Error sending message:', error);
                this.isLoading = false;
            }
        },
        
        async streamResponse(requestId, enableWebSearch = false, enableRAG = false) {
            let url = `/api/stream/${requestId}?conversation_id=${this.currentConversationId}`;
            if (enableWebSearch) {
                url += '&enable_web_search=true';
            }
            if (enableRAG) {
                url += '&enable_rag=true';
            }
            if (this.selectedModel) {
                url += `&model=${encodeURIComponent(this.selectedModel)}`;
            }
            
            this.eventSource = new EventSource(url);
            
            // Create assistant message placeholder
            const assistantMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: '',
                thinking: '',  // Track thinking content separately
                tool_calls: [],
                created_at: new Date().toISOString()
            };
            this.messages.push(assistantMessage);
            
            // Get the message index
            const msgIndex = this.messages.length - 1;
            
            let streamCompleted = false;
            
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    switch (data.type) {
                        case 'content':
                            // Update message content
                            this.messages[msgIndex].content += data.content;
                            
                            // Direct DOM update for immediate streaming display
                            const msgEl = document.getElementById('message-content-' + assistantMessage.id);
                            if (msgEl) {
                                msgEl.innerHTML = marked.parse(this.messages[msgIndex].content);
                            }
                            this.scrollToBottom();
                            break;
                        
                        case 'thinking':
                            // Thinking content from reasoning models (e.g., DeepSeek)
                            this.messages[msgIndex].thinking += data.content;
                            // Update thinking display if element exists
                            const thinkingEl = document.getElementById('thinking-content-' + assistantMessage.id);
                            if (thinkingEl) {
                                thinkingEl.innerHTML = marked.parse(this.messages[msgIndex].thinking);
                            }
                            this.scrollToBottom();
                            break;
                        
                        case 'tool_call_start':
                            this.toolStatus.active = true;
                            this.toolStatus.tool = data.tool;
                            this.toolStatus.status = 'Starting...';
                            this.toolStatus.progress = 0;
                            this.messages[msgIndex].tool_calls.push({
                                name: data.tool,
                                arguments: data.args,
                                status: 'starting',
                                progress: 0,
                                result: null,
                                progress_history: [{
                                    status: 'starting',
                                    progress: 0,
                                    timestamp: new Date().toISOString()
                                }]
                            });
                            break;
                        
                        case 'tool_progress':
                            this.toolStatus.tool = data.tool;
                            this.toolStatus.status = data.status;
                            this.toolStatus.progress = data.progress || null;
                            this.toolStatus.data = data.data || null;
                            
                            // Update the tool call in the messages array
                            const toolCallIndex = this.messages[msgIndex].tool_calls.findIndex(
                                tc => tc.name === data.tool && tc.status !== 'completed'
                            );
                            if (toolCallIndex !== -1) {
                                this.messages[msgIndex].tool_calls[toolCallIndex].status = data.status;
                                this.messages[msgIndex].tool_calls[toolCallIndex].progress = data.progress || 0;
                                
                                // Add to progress history
                                if (!this.messages[msgIndex].tool_calls[toolCallIndex].progress_history) {
                                    this.messages[msgIndex].tool_calls[toolCallIndex].progress_history = [];
                                }
                                this.messages[msgIndex].tool_calls[toolCallIndex].progress_history.push({
                                    status: data.status,
                                    progress: data.progress || 0,
                                    data: data.data || null,
                                    timestamp: new Date().toISOString()
                                });
                                
                                if (data.result) {
                                    this.messages[msgIndex].tool_calls[toolCallIndex].result = data.result;
                                    this.messages[msgIndex].tool_calls[toolCallIndex].status = 'completed';
                                    // Add final entry to progress history
                                    this.messages[msgIndex].tool_calls[toolCallIndex].progress_history.push({
                                        status: 'completed',
                                        progress: 100,
                                        timestamp: new Date().toISOString()
                                    });
                                }
                            }
                            
                            if (data.result) {
                                this.toolStatus.active = false;
                                // Don't add sources to main content - they're already shown in tool_calls section
                            }
                            this.scrollToBottom();
                            break;
                        
                        case 'tool_error':
                            this.toolStatus.active = false;
                            this.messages[msgIndex].content += `\n\n⚠️ Error executing ${data.tool}: ${data.error}`;
                            const errorEl = document.getElementById('message-content-' + assistantMessage.id);
                            if (errorEl) {
                                errorEl.innerHTML = marked.parse(this.messages[msgIndex].content);
                            }
                            this.scrollToBottom();
                            break;
                        
                        case 'error':
                            this.toolStatus.active = false;
                            this.isLoading = false;
                            this.messages[msgIndex].content += `\n\n❌ Error: ${data.error}`;
                            const errorMsgEl = document.getElementById('message-content-' + assistantMessage.id);
                            if (errorMsgEl) {
                                errorMsgEl.innerHTML = marked.parse(this.messages[msgIndex].content);
                            }
                            this.scrollToBottom();
                            break;
                        
                        case 'title_update':
                            this.currentConversationTitle = data.title;
                            const convIndex = this.conversations.findIndex(
                                c => c.id === this.currentConversationId
                            );
                            if (convIndex !== -1) {
                                this.conversations[convIndex].title = data.title;
                            }
                            break;
                        
                        case 'done':
                            streamCompleted = true;
                            this.eventSource.close();
                            this.isLoading = false;
                            this.toolStatus.active = false;
                            // Final update to ensure content is properly rendered
                            this.messages[msgIndex].content = this.messages[msgIndex].content;
                            break;
                    }
                } catch (error) {
                    console.error('Error parsing SSE message:', error);
                }
            };
            
            this.eventSource.onerror = (error) => {
                console.error('SSE Error:', error, 'streamCompleted:', streamCompleted);
                
                // Don't close on every error - check if stream was completed
                if (!streamCompleted) {
                    // Check if we have any content - if so, the connection might have just ended
                    if (this.messages[msgIndex].content && this.messages[msgIndex].content.trim() !== '') {
                        // We have content, just mark as done
                        streamCompleted = true;
                        this.eventSource.close();
                        this.isLoading = false;
                        this.toolStatus.active = false;
                        return;
                    }
                    
                    this.eventSource.close();
                    this.isLoading = false;
                    this.toolStatus.active = false;
                    
                    if (this.messages[msgIndex].content === '') {
                        this.messages[msgIndex].content = '❌ Connection error. Please check:\n1. Is llama.cpp running on port 8080?\n2. Is the backend server running?\n3. Check browser console for details.';
                        const errEl = document.getElementById('message-content-' + assistantMessage.id);
                        if (errEl) {
                            errEl.innerHTML = marked.parse(this.messages[msgIndex].content);
                        }
                    }
                }
            };
        },
        
        // Message editing
        startEditMessage(messageId, content) {
            this.editingMessageId = messageId;
            this.editContent = content;
        },
        
        cancelEdit() {
            this.editingMessageId = null;
            this.editContent = '';
        },
        
        async saveEdit(messageId) {
            if (!this.editContent.trim()) {
                this.cancelEdit();
                return;
            }
            
            // Find the message being edited
            const msg = this.messages.find(m => m.id === messageId);
            if (!msg) {
                this.cancelEdit();
                return;
            }
            
            // If editing a user message, we need to fork the conversation
            if (msg.role === 'user') {
                // Check if content changed
                if (this.editContent.trim() !== msg.content.trim()) {
                    await this.forkConversation(messageId, this.editContent.trim());
                } else {
                    this.cancelEdit();
                }
                return;
            }
            
            // For assistant messages, just update in place
            try {
                const response = await fetch(`/api/messages/${messageId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: this.editContent })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    // Update local message
                    const updatedMsg = this.messages.find(m => m.id === messageId);
                    if (updatedMsg) {
                        updatedMsg.content = data.message.content;
                    }
                }
            } catch (error) {
                console.error('Error updating message:', error);
            }
            
            this.cancelEdit();
        },
        
        async forkConversation(originalMessageId, newContent) {
            // Find the message index
            const msgIndex = this.messages.findIndex(m => m.id === originalMessageId);
            if (msgIndex === -1) {
                this.cancelEdit();
                return;
            }
            
            // Create a new conversation with the edited message
            try {
                const response = await fetch('/api/conversations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: 'Forked: ' + newContent.substring(0, 30) + '...' })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    const newConversationId = data.conversation.id;
                    
                    // Add to conversations list
                    this.conversations.unshift(data.conversation);
                    
                    // Add the edited user message to the new conversation
                    await fetch(`/api/conversations/${newConversationId}/messages`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: newContent })
                    });
                    
                    // Get the request_id for streaming
                    const streamResponse = await fetch(`/api/conversations/${newConversationId}/messages`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: newContent })
                    });
                    
                    const streamData = await streamResponse.json();
                    
                    // Load the new conversation
                    this.currentConversationId = newConversationId;
                    this.currentConversationTitle = data.conversation.title;
                    this.messages = [];
                    
                    // Add user message to display
                    this.messages.push({
                        id: Date.now(),
                        role: 'user',
                        content: newContent,
                        created_at: new Date().toISOString()
                    });
                    
                    // Stream the response
                    await this.streamResponse(streamData.request_id);
                }
            } catch (error) {
                console.error('Error forking conversation:', error);
            }
            
            this.cancelEdit();
        },
        
        async deleteMessage(messageId, event) {
            event.stopPropagation();
            if (!confirm('Are you sure you want to delete this message?')) return;
            
            try {
                const response = await fetch(`/api/messages/${messageId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    this.messages = this.messages.filter(m => m.id !== messageId);
                }
            } catch (error) {
                console.error('Error deleting message:', error);
            }
        },
        
        // Regenerate response
        async regenerateResponse(messageId) {
            if (this.isLoading) return;
            
            this.isLoading = true;
            
            try {
                const response = await fetch(`/api/conversations/${this.currentConversationId}/regenerate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message_id: messageId })
                });
                
                const data = await response.json();
                
                if (data.request_id) {
                    // Remove the old assistant message
                    const msgIndex = this.messages.findIndex(m => m.id === messageId);
                    if (msgIndex !== -1) {
                        this.messages = this.messages.slice(0, msgIndex + 1);
                    }
                    
                    // Stream the new response
                    await this.streamRegenerateResponse(data.request_id);
                }
            } catch (error) {
                console.error('Error regenerating response:', error);
                this.isLoading = false;
            }
        },
        
        async streamRegenerateResponse(requestId) {
            let url = `/api/stream/regenerate/${requestId}?conversation_id=${this.currentConversationId}`;
            if (this.selectedModel) {
                url += `&model=${encodeURIComponent(this.selectedModel)}`;
            }
            
            this.eventSource = new EventSource(url);
            
            // Create assistant message placeholder
            const assistantMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: '',
                thinking: '',  // Track thinking content separately
                tool_calls: [],
                created_at: new Date().toISOString()
            };
            this.messages.push(assistantMessage);
            
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    switch (data.type) {
                        case 'content':
                            assistantMessage.content += data.content;
                            this.scrollToBottom();
                            break;
                        
                        case 'thinking':
                            // Thinking content from reasoning models (e.g., DeepSeek)
                            assistantMessage.thinking += data.content;
                            this.scrollToBottom();
                            break;
                        
                        case 'tool_call_start':
                            this.toolStatus.active = true;
                            this.toolStatus.tool = data.tool;
                            this.toolStatus.status = 'Starting...';
                            this.toolStatus.progress = 0;
                            assistantMessage.tool_calls.push({
                                name: data.tool,
                                arguments: data.args,
                                status: 'starting',
                                progress: 0,
                                result: null,
                                progress_history: [{
                                    status: 'starting',
                                    progress: 0,
                                    timestamp: new Date().toISOString()
                                }]
                            });
                            break;
                        
                        case 'tool_progress':
                            this.toolStatus.tool = data.tool;
                            this.toolStatus.status = data.status;
                            this.toolStatus.progress = data.progress || null;
                            this.toolStatus.data = data.data || null;
                            
                            // Update the tool call in the assistant message
                            const toolCallIdx = assistantMessage.tool_calls.findIndex(
                                tc => tc.name === data.tool && tc.status !== 'completed'
                            );
                            if (toolCallIdx !== -1) {
                                assistantMessage.tool_calls[toolCallIdx].status = data.status;
                                assistantMessage.tool_calls[toolCallIdx].progress = data.progress || 0;
                                
                                // Add to progress history
                                if (!assistantMessage.tool_calls[toolCallIdx].progress_history) {
                                    assistantMessage.tool_calls[toolCallIdx].progress_history = [];
                                }
                                assistantMessage.tool_calls[toolCallIdx].progress_history.push({
                                    status: data.status,
                                    progress: data.progress || 0,
                                    data: data.data || null,
                                    timestamp: new Date().toISOString()
                                });
                                
                                if (data.result) {
                                    assistantMessage.tool_calls[toolCallIdx].result = data.result;
                                    assistantMessage.tool_calls[toolCallIdx].status = 'completed';
                                    // Add final entry to progress history
                                    assistantMessage.tool_calls[toolCallIdx].progress_history.push({
                                        status: 'completed',
                                        progress: 100,
                                        timestamp: new Date().toISOString()
                                    });
                                }
                            }
                            
                            if (data.result) {
                                this.toolStatus.active = false;
                            }
                            this.scrollToBottom();
                            break;
                        
                        case 'tool_error':
                            this.toolStatus.active = false;
                            assistantMessage.content += `\n\n⚠️ Error: ${data.error}`;
                            this.scrollToBottom();
                            break;
                        
                        case 'error':
                            this.toolStatus.active = false;
                            this.isLoading = false;
                            assistantMessage.content += `\n\n❌ Error: ${data.error}`;
                            this.scrollToBottom();
                            break;
                        
                        case 'done':
                            this.eventSource.close();
                            this.isLoading = false;
                            this.toolStatus.active = false;
                            break;
                    }
                } catch (error) {
                    console.error('Error parsing SSE message:', error);
                }
            };
            
            this.eventSource.onerror = (error) => {
                console.error('SSE Error:', error);
                this.eventSource.close();
                this.isLoading = false;
                this.toolStatus.active = false;
            };
        },
        
        // Model Management
        async loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                this.availableModels = data.models || [];
                
                // Set default model if available and none selected
                if (this.availableModels.length > 0 && !this.selectedModel) {
                    // Try to get saved model from localStorage
                    const savedModel = localStorage.getItem('selectedModel');
                    if (savedModel && this.availableModels.some(m => m.id === savedModel)) {
                        this.selectedModel = savedModel;
                    } else {
                        this.selectedModel = this.availableModels[0].id;
                    }
                    this.updateSelectedModel();
                }
            } catch (error) {
                console.error('Error loading models:', error);
            }
        },
        
        updateSelectedModel() {
            // Save to localStorage
            if (this.selectedModel) {
                localStorage.setItem('selectedModel', this.selectedModel);
            }
        },
        
        // Document Management
        async loadDocuments() {
            try {
                const response = await fetch('/api/documents');
                const data = await response.json();
                this.documents = data.documents;
            } catch (error) {
                console.error('Error loading documents:', error);
            }
        },
        
        async uploadDocument(file) {
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/documents/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    this.documents.unshift(data.document);
                    alert('Document uploaded successfully!');
                } else {
                    const error = await response.json();
                    alert(`Error uploading document: ${error.detail}`);
                }
            } catch (error) {
                console.error('Error uploading document:', error);
                alert('Error uploading document');
            }
        },
        
        async deleteDocument(documentId) {
            if (!confirm('Are you sure you want to delete this document?')) return;
            
            try {
                const response = await fetch(`/api/documents/${documentId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    this.documents = this.documents.filter(d => d.id !== documentId);
                }
            } catch (error) {
                console.error('Error deleting document:', error);
            }
        },
        
        // MCP Server Management
        async loadMCPServers() {
            try {
                const response = await fetch('/api/mcp/servers');
                const data = await response.json();
                this.mcpServers = data.servers;
            } catch (error) {
                console.error('Error loading MCP servers:', error);
            }
        },
        
        async addMCPServer() {
            try {
                let args = [];
                try {
                    args = JSON.parse(this.newServer.args);
                } catch (e) {
                    alert('Invalid JSON for arguments');
                    return;
                }
                
                const response = await fetch('/api/mcp/servers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: this.newServer.name,
                        command: this.newServer.command,
                        args: args,
                        env: {}
                    })
                });
                
                if (response.ok) {
                    await this.loadMCPServers();
                    this.newServer = { name: '', command: '', args: '[]' };
                    alert('MCP Server added successfully!');
                } else {
                    alert('Failed to add MCP server');
                }
            } catch (error) {
                console.error('Error adding MCP server:', error);
                alert('Error adding MCP server');
            }
        },
        
        async removeMCPServer(serverName) {
            if (!confirm(`Remove MCP server "${serverName}"?`)) return;
            
            try {
                const response = await fetch(`/api/mcp/servers/${serverName}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    await this.loadMCPServers();
                } else {
                    alert('Failed to remove MCP server');
                }
            } catch (error) {
                console.error('Error removing MCP server:', error);
            }
        },
        
        // Utility Functions
        renderMarkdown(text) {
            if (!text) return '';
            return marked.parse(text);
        },
        
        formatSources(sources) {
            let formatted = '\n\n**Sources:**\n';
            sources.forEach((source, index) => {
                formatted += `${index + 1}. [${source.title}](${source.url})\n`;
                if (source.snippet) {
                    formatted += `   > ${source.snippet}\n\n`;
                }
            });
            return formatted;
        },
        
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
        
        scrollToBottom() {
            this.$nextTick(() => {
                const container = this.$refs.messagesContainer;
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },
        
        // Cancel ongoing request
        cancelRequest() {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
            this.isLoading = false;
            this.toolStatus.active = false;
            
            // If there's an empty assistant message, remove it or mark as cancelled
            const lastMessage = this.messages[this.messages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.content.trim() === '') {
                lastMessage.content = '⚠️ Request cancelled by user.';
            }
            
            this.showToast('Request cancelled', 'info');
        },
        
        // Copy text to clipboard
        async copyToClipboard(text, type = 'text') {
            try {
                await navigator.clipboard.writeText(text);
                this.showToast(`${type} copied to clipboard!`, 'success');
            } catch (error) {
                console.error('Failed to copy:', error);
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                document.body.appendChild(textArea);
                textArea.select();
                try {
                    document.execCommand('copy');
                    this.showToast(`${type} copied to clipboard!`, 'success');
                } catch (err) {
                    this.showToast('Failed to copy text', 'error');
                }
                document.body.removeChild(textArea);
            }
        },
        
        // Copy message content (strips markdown for clean copy)
        async copyMessage(message) {
            await this.copyToClipboard(message.content, message.role === 'user' ? 'Query' : 'Response');
        },
        
        // Copy message as markdown
        async copyAsMarkdown(message) {
            const prefix = message.role === 'user' ? '**User:**\n' : '**Assistant:**\n';
            await this.copyToClipboard(prefix + message.content, 'Markdown');
        },
        
        // Show toast notification
        showToast(message, type = 'success') {
            this.toast.message = message;
            this.toast.type = type;
            this.toast.show = true;
            
            setTimeout(() => {
                this.toast.show = false;
            }, 2500);
        },
        
        // Toggle sidebar collapse
        toggleSidebar() {
            this.sidebarCollapsed = !this.sidebarCollapsed;
        },
        
        // Toggle tool calls expansion for a message
        toggleToolCalls(messageId) {
            this.expandedToolCalls[messageId] = !this.expandedToolCalls[messageId];
        },
        
        // Check if tool calls are expanded for a message
        isToolCallsExpanded(messageId) {
            return this.expandedToolCalls[messageId] === true;
        },
        
        // Expanded thinking tracking (by message id)
        expandedThinking: {},
        
        // Toggle thinking expansion for a message
        toggleThinking(messageId) {
            this.expandedThinking[messageId] = !this.expandedThinking[messageId];
        },
        
        // Check if thinking is expanded for a message
        isThinkingExpanded(messageId) {
            return this.expandedThinking[messageId] === true;
        },
        
        // Expanded sources tracking (by message id)
        expandedSources: {},
        
        // Toggle sources expansion for a message
        toggleSources(messageId) {
            this.expandedSources[messageId] = !this.expandedSources[messageId];
        },
        
        // Check if sources are expanded for a message
        isSourcesExpanded(messageId) {
            return this.expandedSources[messageId] === true;
        },
        
        // TTS (Text-to-Speech) state
        ttsLoading: {},
        currentAudio: null,
        currentAudioMessageId: null,
        
        // Speak message using TTS
        async speakMessage(message) {
            // Stop any currently playing audio
            if (this.currentAudio) {
                this.stopAudio();
                return;
            }
            
            // Get text content (strip HTML if any)
            const text = message.content.replace(/<[^>]*>/g, '').trim();
            if (!text) {
                this.showToast('No text to speak', 'error');
                return;
            }
            
            // Set loading state
            this.ttsLoading[message.id] = true;
            
            try {
                const response = await fetch('/api/tts/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                
                if (!response.ok) {
                    throw new Error('TTS generation failed');
                }
                
                const data = await response.json();
                
                if (data.success && data.audio_url) {
                    // Create and play audio
                    this.currentAudio = new Audio(data.audio_url);
                    this.currentAudioMessageId = message.id;
                    
                    this.currentAudio.onended = () => {
                        this.currentAudio = null;
                        this.currentAudioMessageId = null;
                    };
                    
                    this.currentAudio.onerror = () => {
                        this.showToast('Failed to play audio', 'error');
                        this.currentAudio = null;
                        this.currentAudioMessageId = null;
                    };
                    
                    await this.currentAudio.play();
                } else {
                    throw new Error(data.error || 'TTS generation failed');
                }
            } catch (error) {
                console.error('TTS error:', error);
                this.showToast('Failed to generate speech', 'error');
            } finally {
                this.ttsLoading[message.id] = false;
            }
        },
        
        // Stop audio playback
        stopAudio() {
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0;
                this.currentAudio = null;
                this.currentAudioMessageId = null;
            }
        },
        
        // Get sources from message tool calls
        getMessageSources(message) {
            if (!message.tool_calls || message.tool_calls.length === 0) {
                return [];
            }
            
            const allSources = [];
            message.tool_calls.forEach(toolCall => {
                if (toolCall.result && toolCall.result.sources) {
                    toolCall.result.sources.forEach(source => {
                        // Avoid duplicates
                        if (!allSources.some(s => s.url === source.url)) {
                            allSources.push(source);
                        }
                    });
                }
            });
            
            return allSources;
        },
        
        // Render markdown with citation support
        renderMarkdownWithCitations(text, sources) {
            if (!text) return '';
            
            // First parse markdown
            let html = marked.parse(text);
            
            // If we have sources, replace citation references [1], [2], etc. with clickable links
            if (sources && sources.length > 0) {
                // Match [N] patterns where N is a number
                html = html.replace(/\[(\d+)\]/g, (match, num) => {
                    const index = parseInt(num) - 1;
                    if (index >= 0 && index < sources.length) {
                        const source = sources[index];
                        const title = source.title || 'Source';
                        const url = source.url || '#';
                        return `<sup><a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-link" title="${title}">[${num}]</a></sup>`;
                    }
                    return match;
                });
            }
            
            return html;
        }
    }
}

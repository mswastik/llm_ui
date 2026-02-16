# Quick Start Guide

Get up and running with LLM UI in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.10 or higher installed
- [ ] llama.cpp compiled and ready with embeddings support
- [ ] A GGUF model file downloaded
- [ ] Node.js installed (for MCP servers)
- [ ] Optional: SearXNG instance for web search
- [ ] Optional: edge-tts for high-quality TTS (`pip install edge-tts`)

## Step-by-Step Setup

### 1. Start llama.cpp Server (Terminal 1)

```bash
cd /path/to/llama.cpp
./llama-server -m /path/to/your-model.gguf --port 8080 --host 0.0.0.0 --embeddings
```

**Verify it's working:**
```bash
curl http://localhost:8080/health
# Should return: {"status":"ok"}
```

### 2. Install LLM UI (Terminal 2)

```bash
cd llm-ui-app/backend
pip install -r requirements.txt
```

### 3. Install Optional Dependencies (Optional)

For enhanced features:

```bash
# For high-quality text-to-speech
pip install edge-tts

# For offline text-to-speech (alternative)
pip install pyttsx3
```

### 4. Run the Application

```bash
cd llm-ui-app
python run.py
```

You should see:
```
============================================================
🚀 Starting LLM UI Application
============================================================
📍 Server: http://0.0.0.0:8000
...
```

### 5. Open in Browser

Navigate to: **http://localhost:8000**

You should see the chat interface with:
- Left sidebar: Conversations, Knowledge Base, Settings
- Center: Chat area with model selector and tool toggles
- Input box at bottom

### 6. Test Basic Chat

Type a message and hit Enter (or click Send):

```
Hello! Can you help me with Python?
```

The response should stream in real-time!

### 7. Explore Enhanced Features

#### Web Search
Enable "Web Search" toggle and ask:
```
What are the latest developments in quantum computing?
```

#### Document Processing
1. Click "Knowledge Base" in the sidebar
2. Upload a document (PDF, DOCX, TXT, etc.)
3. Enable "Search Documents" and ask questions about the content

#### Text-to-Speech
Click the speaker icon next to any response to hear it spoken aloud.

#### Model Selection
Use the model dropdown to switch between different models if available.

### 8. Add an MCP Server (Optional)

Click "Settings" → "MCP Servers" tab:

**Example - Filesystem Server:**
```
Name: filesystem
Command: npx
Arguments: ["-y", "@modelcontextprotocol/server-filesystem", "/home/youruser"]
```

Click "Add Server"

**Example - Time Server:**
```
Name: time
Command: npx
Arguments: ["-y", "@modelcontextprotocol/server-time"]
```

### 9. Test Tool Usage

Once you've added an MCP server, try:

```
What files are in my home directory?
```

or

```
What time is it?
```

You should see:
1. Tool execution start notification
2. Progress updates (if using custom tools)
3. Tool results in the response
4. Sources cited with clickable links

## Troubleshooting

### Connection to llama.cpp failed

**Problem:** UI shows error connecting to LLM

**Solutions:**
1. Check llama.cpp is running: `curl http://localhost:8080/health`
2. Verify port 8080 isn't blocked by firewall
3. Check logs in llama.cpp terminal
4. Update URL in `backend/config.py` if using different port

### MCP Server won't start

**Problem:** Added MCP server but it shows error

**Solutions:**
1. Verify Node.js is installed: `node --version`
2. Check npm global packages: `npm list -g`
3. Try installing package manually: `npm install -g @modelcontextprotocol/server-time`
4. Check server logs in backend terminal

### No messages appear

**Problem:** Sent message but nothing happens

**Solutions:**
1. Open browser console (F12) and check for errors
2. Verify database file `llm_ui.db` was created
3. Check backend logs for errors
4. Try creating a new conversation

### UI loads but looks broken

**Problem:** Page appears but styling is wrong

**Solutions:**
1. Hard refresh browser: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
2. Check browser console for 404 errors on CSS/JS files
3. Verify `/static` directory exists with css/js folders

### Web Search not working

**Problem:** Web search tool returns errors

**Solutions:**
1. Verify SearXNG instance is running and accessible
2. Check `SEARXNG_URL` in `backend/config.py`
3. Ensure internet connectivity for web requests

### TTS not working

**Problem:** Speaker icon doesn't produce audio

**Solutions:**
1. Install edge-tts: `pip install edge-tts`
2. Or install pyttsx3: `pip install pyttsx3`
3. Check TTS settings in Settings → TTS tab

## Next Steps

### Customize Settings

Edit `backend/config.py` or create `.env` file:

```bash
cp .env.example .env
nano .env
```

### Configure Web Search

Set up SearXNG for privacy-focused web search:
- Install SearXNG locally or use a public instance
- Update `SEARXNG_URL` in configuration

### Add Your Documents

Build a knowledge base by uploading documents:
- PDF, DOCX, TXT, MD, JSON, YAML formats supported
- Documents are automatically processed and indexed
- Ask questions about your documents using the RAG feature

### Explore MCP Servers

Check out available MCP servers:
- https://github.com/modelcontextprotocol/servers
- Filesystem, GitHub, Google Drive, Slack, and more!

## Common Commands

**Start application:**
```bash
python run.py
```

**Start with hot reload (development):**
```bash
# Set DEBUG=true in .env, then:
python run.py
```

**Reset database:**
```bash
rm llm_ui.db
python run.py  # Will recreate on startup
```

**Check logs:**
```bash
# Logs appear in the terminal where you ran run.py
```

## Default Settings

- **Port:** 8000
- **Database:** SQLite (llm_ui.db)
- **llama.cpp URL:** http://localhost:8080
- **Temperature:** 0.7
- **Max Tokens:** 16048
- **Upload Directory:** ./uploads
- **Max Upload Size:** 10MB

## Get Help

- Check `README.md` for detailed documentation
- Review `SEARXNG_INTEGRATION.md` for search tool integration
- Look at example code in `backend/tools/` directory
- Visit Settings modal for configuration options

## Success Checklist

After setup, you should be able to:

- [ ] Send messages and get streaming responses
- [ ] Create new conversations
- [ ] Switch between conversations
- [ ] See conversation history persist after refresh
- [ ] Add/remove MCP servers
- [ ] Use MCP tools in conversations
- [ ] See real-time progress for tool executions
- [ ] Enable/disable web search
- [ ] Upload documents to knowledge base
- [ ] Query documents using RAG
- [ ] Use text-to-speech for responses
- [ ] Switch between different models
- [ ] Access application settings

If all boxes are checked - you're good to go! 🎉

# Quick Start Guide

Get up and running with LLM UI in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.10 or higher installed
- [ ] llama.cpp compiled and ready
- [ ] A GGUF model file downloaded
- [ ] Node.js installed (for MCP servers)

## Step-by-Step Setup

### 1. Start llama.cpp Server (Terminal 1)

```bash
cd /path/to/llama.cpp
./llama-server -m /path/to/your-model.gguf --port 8080 --host 0.0.0.0 -ngl 99
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

### 3. Run the Application

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

### 4. Open in Browser

Navigate to: **http://localhost:8000**

You should see the chat interface with:
- Left sidebar: Conversations
- Center: Chat area
- Input box at bottom

### 5. Test Basic Chat

Type a message and hit Enter (or click Send):

```
Hello! Can you help me with Python?
```

The response should stream in real-time!

### 6. Add an MCP Server (Optional)

Click "MCP Servers" button in sidebar:

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

### 7. Test Tool Usage

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

## Next Steps

### Customize Settings

Edit `backend/config.py` or create `.env` file:

```bash
cp .env.example .env
nano .env
```

### Add Your Search Tool

Follow the guide in `SEARXNG_INTEGRATION.md` to integrate your SearXNG search pipeline.

### Explore MCP Servers

Check out available MCP servers:
- https://github.com/modelcontextprotocol/servers
- Filesystem, GitHub, Google Drive, Slack, and more!

### Enable Document Upload

Uncomment the document upload section in `backend/app/main.py` and implement your document processing logic.

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
- **Max Tokens:** 2048

## Get Help

- Check `README.md` for detailed documentation
- Review `SEARXNG_INTEGRATION.md` for search tool integration
- Look at example code in `backend/tools/tool_executor.py`

## Success Checklist

After setup, you should be able to:

- [ ] Send messages and get streaming responses
- [ ] Create new conversations
- [ ] Switch between conversations
- [ ] See conversation history persist after refresh
- [ ] Add/remove MCP servers
- [ ] Use MCP tools in conversations
- [ ] See real-time progress for tool executions

If all boxes are checked - you're good to go! 🎉

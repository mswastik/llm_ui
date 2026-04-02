# SearXNG MCP Server

A standalone MCP (Model Context Protocol) server that provides web search capabilities using [SearXNG](https://github.com/searxng/searxng), a privacy-respecting metasearch engine.

## Features

- **Web Search**: Search the web for current information using SearXNG
- **Multi-query Generation**: Automatically generates alternative search queries for better coverage
- **Content Extraction**: Extracts and chunks content from web pages
- **Simple Search**: Quick search mode returning just titles, URLs, and snippets
- **Privacy-focused**: Uses SearXNG which doesn't track your searches

## Tools

This MCP server provides the following tools:

### `search_web`

Full-featured web search with content extraction.

**Parameters:**
- `query` (string, required): The search query
- `max_results` (integer, optional, default: 30): Maximum results to fetch per query
- `top_k` (integer, optional, default: 22): Final number of results to return
- `enable_multi_query` (boolean, optional, default: true): Generate multiple search queries

**Returns:**
```json
{
  "success": true,
  "sources": [
    {
      "id": 1,
      "title": "Page Title",
      "url": "https://example.com",
      "snippet": "Brief description...",
      "chunk_content": "Extracted content..."
    }
  ],
  "content": "# 🔍 Search Results\n\n...",
  "chunks": ["content chunk 1", "content chunk 2"]
}
```

### `search_web_simple`

Quick search returning basic results without content extraction.

**Parameters:**
- `query` (string, required): The search query
- `num_results` (integer, optional, default: 10): Number of results to return

**Returns:**
```json
{
  "success": true,
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com",
      "snippet": "Brief description...",
      "engine": "google"
    }
  ],
  "query": "your search query"
}
```

### `get_search_configuration`

Get the current server configuration.

**Returns:**
```json
{
  "searxng_url": "http://localhost:8888/search",
  "num_search_results": 25,
  "chunk_size": 1200,
  "similarity_threshold": 0.4,
  "enable_multi_query": true,
  "llm_configured": false
}
```

## Installation

### Option 1: Using uvx (Recommended)

```bash
# Run directly without installation
uvx --from git+https://github.com/your-org/llm_ui#subdirectory=mcp_servers/searxng_search mcp-server-searxng
```

### Option 2: Install locally

```bash
cd mcp_servers/searxng_search

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Option 3: Install as Python package

```bash
cd mcp_servers/searxng_search
pip install -e .
```

Then run with:
```bash
mcp-server-searxng
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEARXNG_URL` | Your SearXNG instance URL | `http://localhost:8888/search` |

### Setting up SearXNG

You need a running SearXNG instance. Here are the options:

#### Using Docker (Recommended)

```bash
docker run -d --name searxng -p 8888:8080 \
  -e BASE_URL="http://localhost:8888" \
  searxng/searxng:latest
```

#### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3'
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8888:8080"
    environment:
      - BASE_URL=http://localhost:8888
    volumes:
      - ./searxng:/etc/searxng:rw
```

Then run:
```bash
docker-compose up -d
```

#### Self-hosted Installation

```bash
git clone https://github.com/searxng/searxng.git
cd searxng
pip install -e .
searxng run
```

## Usage

### With Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "searxng": {
      "command": "python",
      "args": ["/path/to/mcp_servers/searxng_search/server.py"],
      "env": {
        "SEARXNG_URL": "http://localhost:8888/search"
      }
    }
  }
}
```

Or if installed as package:
```json
{
  "mcpServers": {
    "searxng": {
      "command": "mcp-server-searxng",
      "env": {
        "SEARXNG_URL": "http://localhost:8888/search"
      }
    }
  }
}
```

### With uvx

```json
{
  "mcpServers": {
    "searxng": {
      "command": "uvx",
      "args": ["mcp-server-searxng"],
      "env": {
        "SEARXNG_URL": "http://localhost:8888/search"
      }
    }
  }
}
```

### With Cursor

Add to your Cursor MCP settings:
```json
{
  "mcpServers": [
    {
      "name": "searxng",
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/mcp_servers/searxng_search/server.py"],
      "env": {
        "SEARXNG_URL": "http://localhost:8888/search"
      }
    }
  ]
}
```

### With Other MCP Clients

Any MCP client that supports stdio transport can use this server:

```bash
python /path/to/mcp_servers/searxng_search/server.py
```

The server will communicate via stdin/stdout using JSON-RPC.

## Examples

### Example 1: Simple Search

**User:** "Search for recent news about quantum computing breakthroughs"

**Tool Call:**
```json
{
  "name": "search_web_simple",
  "arguments": {
    "query": "quantum computing breakthroughs 2025",
    "num_results": 10
  }
}
```

### Example 2: Full Search with Content

**User:** "Find detailed information about how transformers work in machine learning"

**Tool Call:**
```json
{
  "name": "search_web",
  "arguments": {
    "query": "how transformers work machine learning explained",
    "max_results": 30,
    "top_k": 20
  }
}
```

### Example 3: Multi-query Search

When `enable_multi_query` is true, the server automatically generates alternative queries:

Original query: "best practices for API design"
Generated queries:
- "API design best practices 2025"
- "RESTful API design guidelines"

## Troubleshooting

### Connection Errors

**Problem:** Cannot connect to SearXNG

**Solutions:**
1. Verify SearXNG is running: `curl http://localhost:8888`
2. Check the `SEARXNG_URL` environment variable
3. Ensure network connectivity between the server and SearXNG

### No Results Found

**Problem:** Search returns no results

**Solutions:**
1. Try a simpler or different query
2. Check if SearXNG engines are configured properly
3. Increase `max_results` parameter
4. Verify your SearXNG instance has working search engines

### Slow Performance

**Problem:** Search takes too long

**Solutions:**
1. Reduce `max_results` and `top_k` parameters
2. Use `search_web_simple` for quicker results
3. Disable multi-query: `enable_multi_query: false`
4. Check network latency to SearXNG instance

### Content Extraction Issues

**Problem:** Cannot extract content from web pages

**Solutions:**
1. Some websites block automated access
2. The page may require JavaScript rendering
3. Check if the URL is accessible manually
4. The snippet from SearXNG will be used as fallback

## Advanced Configuration

### LLM Integration (Optional)

For better query generation, you can configure an LLM:

```python
# Set these environment variables or modify server.py
LLM_BASE_URL="http://localhost:8080/v1"
LLM_MODEL="qwen3.5-4b"
QUERY_MODEL="qwen3.5-4b"
LLM_API_KEY="sk-12"
```

### Custom Search Categories

Modify the `_fetch_searxng_results` method to use specific categories:

```python
params = {
    "q": query,
    "format": "json",
    "categories": "general,science,technology"
}
```

## Development

### Running Tests

```bash
cd mcp_servers/searxng_search
pip install -e ".[dev]"
pytest tests/
```

### Building the Package

```bash
pip install build
python -m build
```

## License

MIT License - See the main project LICENSE for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [SearXNG](https://github.com/searxng/searxng) - The privacy-respecting metasearch engine
- [MCP SDK](https://github.com/modelcontextprotocol) - Model Context Protocol
- This implementation is based on the SearXNG integration from the [LLM UI project](https://github.com/your-org/llm_ui)

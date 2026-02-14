# Integrating Your SearXNG Search Tool

This guide shows how to integrate your existing SearXNG-based search tool (from open-webui) into this application with full real-time progress updates.

## Your Current Tool Architecture

From your description, your tool does:
1. ✅ Generate search query
2. ✅ Execute SearXNG search
3. ✅ Parse web pages
4. ✅ Generate embeddings
5. ✅ Rerank results
6. ✅ Answer user query with citations

## Integration Steps

### 1. Copy Your Tool Code

Create a new file: `backend/tools/searxng_search.py`

```python
import asyncio
from typing import AsyncGenerator, Dict, Any, List
import aiohttp

# Import your existing modules
# from your_embedding_module import embed_text
# from your_reranker_module import rerank_results
# from your_parser_module import parse_webpage


async def search_web_with_progress(
    arguments: Dict[str, Any],
    request_id: str
) -> AsyncGenerator[Dict, None]:
    """
    SearXNG-based web search with real-time progress updates.
    
    This wraps your existing search pipeline and emits progress events
    that are streamed to the UI via Server-Sent Events.
    """
    
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 10)
    
    # Step 1: Generate optimized search query
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": "Generating search query...",
        "progress": 5
    }
    
    # Your LLM-based query optimization logic here
    optimized_query = await optimize_search_query(query)
    
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": f"Searching: {optimized_query}",
        "progress": 10,
        "data": {"query": optimized_query}
    }
    
    # Step 2: Execute SearXNG search
    search_results = await execute_searxng(optimized_query, num_results)
    
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": f"Found {len(search_results)} results",
        "progress": 25,
        "data": {"result_count": len(search_results)}
    }
    
    # Step 3: Parse web pages
    parsed_contents = []
    for i, result in enumerate(search_results):
        yield {
            "type": "tool_progress",
            "tool": "search_web",
            "status": f"Parsing website {i+1}/{len(search_results)}: {result['title'][:50]}...",
            "progress": 25 + (i / len(search_results) * 30),  # 25-55%
            "data": {
                "parsing": i+1,
                "total": len(search_results),
                "current_url": result["url"]
            }
        }
        
        # Your webpage parsing logic
        content = await parse_webpage(result["url"])
        parsed_contents.append({
            "title": result["title"],
            "url": result["url"],
            "content": content,
            "snippet": result.get("snippet", "")
        })
        
        await asyncio.sleep(0.1)  # Small delay between requests
    
    # Step 4: Generate embeddings
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": "Generating embeddings for retrieved content...",
        "progress": 60
    }
    
    # Your embedding logic
    embedded_contents = await embed_all_contents(parsed_contents, query)
    
    # Step 5: Rerank results
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": "Reranking results by relevance...",
        "progress": 80
    }
    
    # Your reranking logic
    reranked_results = await rerank_by_relevance(embedded_contents, query)
    
    # Step 6: Generate answer
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": "Generating answer from sources...",
        "progress": 90
    }
    
    # Format sources for citation
    sources = [
        {
            "title": r["title"],
            "url": r["url"],
            "snippet": r["content"][:200] + "...",
            "relevance_score": r.get("score", 0.0)
        }
        for r in reranked_results[:5]  # Top 5 sources
    ]
    
    # Final result
    yield {
        "type": "tool_progress",
        "tool": "search_web",
        "status": "Complete",
        "progress": 100,
        "result": {
            "query": query,
            "optimized_query": optimized_query,
            "sources": sources,
            "total_results": len(search_results),
            "processing_time": "X.XX seconds"  # Add actual timing
        }
    }


# Helper functions - implement these based on your existing code

async def optimize_search_query(query: str) -> str:
    """Generate an optimized search query using LLM"""
    # Your implementation
    return query


async def execute_searxng(query: str, num_results: int) -> List[Dict]:
    """Execute search via SearXNG"""
    # Your SearXNG integration
    # Example:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:8888/search",  # Your SearXNG instance
            params={
                "q": query,
                "format": "json",
                "language": "en"
            }
        ) as response:
            data = await response.json()
            return data.get("results", [])[:num_results]


async def parse_webpage(url: str) -> str:
    """Parse webpage content"""
    # Your webpage parsing implementation
    # Could use beautifulsoup, trafilatura, etc.
    pass


async def embed_all_contents(contents: List[Dict], query: str) -> List[Dict]:
    """Generate embeddings for all content"""
    # Your embedding implementation
    # Use your llama.cpp embedding model
    pass


async def rerank_by_relevance(contents: List[Dict], query: str) -> List[Dict]:
    """Rerank results by relevance"""
    # Your reranking implementation
    # Use your llama.cpp reranking model
    pass
```

### 2. Register the Tool

Edit `backend/tools/tool_executor.py`:

```python
from tools.searxng_search import search_web_with_progress

class ToolExecutor:
    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager
        self.custom_tools = {
            "search_web": search_web_with_progress,  # Use your implementation
            "analyze_document": self._analyze_document_with_progress,
        }
```

### 3. Update LLM Client to Include Search Tool

Edit `backend/llm_client/client.py` in the `_get_available_tools` method:

```python
async def _get_available_tools(self) -> List[Dict]:
    """Get tool definitions to send to the LLM"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information using SearXNG. Returns relevant sources with citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to retrieve (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Add MCP tools as well
    mcp_tools = await self.mcp_manager.list_all_tools()
    for tool in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": f"{tool['server']}:{tool['name']}",
                "description": tool['description'],
                "parameters": tool['input_schema']
            }
        })
    
    return tools
```

### 4. Configure Your Services

Make sure these are running:

**SearXNG Instance:**
```bash
# Your SearXNG should be running on localhost:8888
# Or update the URL in execute_searxng()
```

**llama.cpp Embedding Server:**
```bash
# Start embedding server for your embedding model
./llama-server -m /path/to/embedding-model.gguf --port 8081 --embedding
```

**llama.cpp Reranking Server (optional):**
```bash
# If using separate reranking model
./llama-server -m /path/to/reranking-model.gguf --port 8082
```

### 5. Environment Variables

Add to your `.env` file:

```bash
# SearXNG Configuration
SEARXNG_URL=http://localhost:8888
SEARXNG_LANGUAGE=en

# Embedding Service
EMBEDDING_URL=http://localhost:8081

# Reranking Service (optional)
RERANKING_URL=http://localhost:8082

# Search Settings
DEFAULT_NUM_RESULTS=10
MAX_PARSE_WORKERS=5
```

### 6. Test the Integration

Start the application and try a search query:

```
User: "What are the latest developments in quantum computing?"
```

You should see real-time progress updates in the UI:
- ✅ Generating search query...
- ✅ Searching: quantum computing breakthroughs 2024
- ✅ Found 10 results
- ✅ Parsing website 1/10: MIT News - Quantum...
- ✅ Parsing website 2/10: Nature - Recent advances...
- ✅ Generating embeddings...
- ✅ Reranking results...
- ✅ Complete

## Displaying Citations in the UI

The frontend automatically formats sources. The tool result should include:

```python
"result": {
    "sources": [
        {
            "title": "Article Title",
            "url": "https://example.com",
            "snippet": "Relevant excerpt...",
            "relevance_score": 0.95
        }
    ]
}
```

The UI will render these as clickable citations with snippets.

## Advanced: Making it an MCP Server

Alternatively, you could package your entire search pipeline as a standalone MCP server:

```bash
# Create a new MCP server package
mkdir my-search-mcp-server
cd my-search-mcp-server
npm init -y
```

Then add it to the UI via the MCP Servers settings.

This approach gives you:
- ✅ Portability across different LLM UIs
- ✅ Standard MCP protocol compliance
- ✅ Easy sharing with others

However, you lose real-time progress updates (MCP limitation).

## Recommendation

**Use the custom tool approach** for your use case since:
1. You need granular progress updates
2. You're building a dedicated UI for your workflow
3. You have complex multi-step processing

The MCP approach is better for:
- Simple, atomic tools
- Cross-application compatibility
- Tools that don't need progress tracking

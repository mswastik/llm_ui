# SearXNG Web Search Integration

This guide explains how the SearXNG web search feature is integrated into the LLM UI application with real-time progress updates.

## Current Implementation Architecture

The application includes a sophisticated SearXNG integration that performs:
1. ✅ Multi-query generation for better coverage
2. ✅ Adaptive query generation based on initial results
3. ✅ Execute SearXNG search with privacy-focused results
4. ✅ Parse web pages and extract content
5. ✅ Generate embeddings for semantic search
6. ✅ Rerank results by relevance
7. ✅ Format citations with source tracking
8. ✅ Real-time progress updates to the UI

## How It Works

### 1. Search Flow

When a user enables web search:

1. **Query Enhancement**: The system generates multiple search queries to improve coverage
2. **Adaptive Queries**: Additional queries are generated based on initial results
3. **SearXNG Search**: Queries are sent to your configured SearXNG instance
4. **Content Extraction**: Web pages are parsed and content is extracted
5. **Semantic Processing**: Content is chunked and embedded for relevance ranking
6. **Result Reranking**: Results are reordered by semantic relevance
7. **Context Injection**: Relevant results are injected into the LLM context
8. **Citation Formatting**: Sources are formatted with citations for the response

### 2. Real-time Progress Updates

The search process provides granular progress updates:
- Generating optimized search queries...
- Searching with X queries...
- Found X unique results. Extracting content...
- Created X chunks from X sources
- Computing embeddings...
- Calculating semantic similarities...
- Filtering by similarity threshold...
- Reranking results...
- Formatting results...

### 3. Configuration

The SearXNG integration is configured in `backend/config.py`:

```python
class SearchConfig:
    searxng_url: str = "http://localhost:8888/search"  # Your SearXNG instance
    embeddings_api: str = f"{LLAMA_CPP_BASE_URL}/v1/embeddings"  # llama.cpp embeddings endpoint
    rerank_api: str = f"{LLAMA_CPP_BASE_URL}/v1/rerank"  # llama.cpp rerank endpoint
    llm_base_url: str = LLAMA_CPP_BASE_URL
    llm_model: str = LLAMA_CPP_MODEL
    query_model: str = QUERY_MODEL  # Model for query generation
    num_search_results: int = 25  # Number of results per query
    chunk_size: int = 1200  # Words per content chunk
    similarity_threshold: float = 0.4  # Minimum similarity for inclusion
    enable_multi_query: bool = True  # Enable multi-query generation
```

## Setting Up SearXNG

### 1. Install SearXNG

Follow the official SearXNG installation guide:
```bash
# Using Docker (recommended)
docker run -d --name searxng -p 8888:8080 \
  -e BASE_URL="http://localhost:8888" \
  searxng/searxng:latest

# Or install locally
git clone https://github.com/searxng/searxng.git
cd searxng
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
python searxng_extra/settings_v1.yml install
```

### 2. Configure SearXNG

Edit your SearXNG configuration to enable the JSON output format:
- Ensure the `json` output format is enabled in your SearXNG settings
- Configure search engines as needed for your use case

### 3. Update Application Configuration

Update the SearXNG URL in `backend/config.py`:

```python
SEARXNG_URL = "http://localhost:8888/search"  # Point to your SearXNG instance
```

Or use environment variables:
```bash
SEARXNG_URL="http://your-searxng-instance.com/search"
```

## Using Web Search in the UI

### 1. Enable Web Search
- Type your message in the chat input
- Toggle the "Web Search" button (magnifying glass icon)
- Send your message
- The application will perform a web search before generating the response

### 2. View Search Results
- Search progress appears in real-time
- Sources are displayed with citations ([1], [2], etc.)
- Click on citations to view source URLs
- Results are integrated into the LLM's response

## Advanced Configuration

### 1. Embedding and Reranking Models

For optimal search results, use dedicated models:

**Embedding Model**: For semantic similarity calculations
```bash
# Dedicated embedding server
./llama-server -m /path/to/embedding-model.gguf --port 8081 --embedding
```

**Reranking Model**: For result reordering (if different from embedding model)
```bash
# Optional reranking server
./llama-server -m /path/to/reranking-model.gguf --port 8082 --rerank
```

### 2. Performance Tuning

Adjust search parameters in `backend/tools/searxng_tool.py`:

```python
# Chunk size (larger = fewer chunks, smaller = more precision)
chunk_size: int = 1200

# Similarity threshold (higher = more relevant, lower = more results)
similarity_threshold: float = 0.4

# Number of search results to fetch per query
num_search_results: int = 25

# Enable/disable multi-query generation
enable_multi_query: bool = True
```

### 3. Custom Search Categories

Modify the SearXNG request parameters to focus on specific categories:

```python
async def _fetch_searxng_results(self, query: str) -> List[Tuple[str, str, str]]:
    """Fetch search results from SearXNG"""
    params = {
        "q": query, 
        "format": "json", 
        "categories": "general"  # Change to "science", "news", etc.
    }
    # ... rest of implementation
```

## Troubleshooting

### Common Issues

#### SearXNG Connection Issues
**Problem**: Cannot connect to SearXNG instance
**Solutions**:
1. Verify SearXNG is running: `curl http://localhost:8888`
2. Check URL in configuration matches your instance
3. Ensure network connectivity between application and SearXNG

#### Slow Search Performance
**Problem**: Search takes too long
**Solutions**:
1. Reduce `num_search_results` in configuration
2. Lower `chunk_size` for faster processing
3. Increase similarity threshold to filter early
4. Check network latency to SearXNG instance

#### Poor Search Results
**Problem**: Irrelevant results or low-quality content
**Solutions**:
1. Adjust `similarity_threshold` (try 0.3-0.6 range)
2. Verify embedding model quality
3. Check SearXNG search engine configuration
4. Enable/disable multi-query generation based on needs

#### Embedding/Reranking Failures
**Problem**: Semantic processing fails
**Solutions**:
1. Verify llama.cpp server supports embeddings endpoint
2. Check model compatibility with embedding/reranking
3. Ensure sufficient VRAM for embedding model
4. Verify API endpoint configuration

## Customization

### 1. Modify Search Behavior

Edit `backend/tools/searxng_tool.py` to customize:
- Query generation logic
- Content extraction methods
- Chunking strategies
- Similarity calculations
- Result formatting

### 2. Add Search Categories

Extend the search to include specific categories by modifying the SearXNG API call:

```python
params = {
    "q": query, 
    "format": "json", 
    "categories": "general,science,technology"  # Multiple categories
}
```

### 3. Adjust Progress Reporting

Modify the progress reporting granularity in the search methods to provide more or less detailed updates to the UI.

## Integration with Other Features

The SearXNG search seamlessly integrates with:
- **RAG (Document Search)**: Can be used alongside document search
- **MCP Tools**: Works in conjunction with other tools
- **Model Selection**: Respects selected model for responses
- **Citation System**: Provides proper source attribution
- **Real-time UI**: Shows progress updates as they happen

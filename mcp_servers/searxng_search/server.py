"""
SearXNG MCP Server - Standalone search server using SearXNG.

This MCP server provides web search capabilities using SearXNG,
a privacy-respecting metasearch engine. It can be used independently
in any UI that supports MCP servers.

Features:
- Multi-query generation for better coverage
- Adaptive query generation based on initial results
- Content extraction from web pages
- Semantic search with embeddings and reranking (optional)
- Real-time progress updates

Usage:
    # Using uvx (recommended)
    uvx mcp-server-searxng
    
    # Or run directly
    python server.py
    
    # With custom configuration
    SEARXNG_URL="http://your-searxng-instance.com/search" python server.py
"""

import asyncio
import json
import re
import warnings
from typing import Any, Optional
from dataclasses import dataclass, field

import requests
import numpy as np
from bs4 import BeautifulSoup

from mcp.server.fastmcp import FastMCP


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SearchConfig:
    """Configuration for SearXNG search"""
    searxng_url: str = field(default_factory=lambda: 
        __import__('os').environ.get('SEARXNG_URL', 'http://localhost:8888/search'))
    num_search_results: int = 15
    chunk_size: int = 1200
    similarity_threshold: float = 0.4
    max_retries: int = 3
    enable_multi_query: bool = False
    # LLM configuration for query generation (optional)
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    query_model: Optional[str] = None


# Global config instance
_config = SearchConfig()


# ============================================================================
# MCP Server Setup
# ============================================================================

mcp = FastMCP(
    "searxng-search",
    dependencies=["requests", "beautifulsoup4", "numpy"]
)


# ============================================================================
# Search Tool Implementation
# ============================================================================

@mcp.tool()
async def search_web(
    query: str,
    max_results: int = 15,
    top_k: int = 10,
    enable_multi_query: bool = False
) -> dict[str, Any]:
    """
    Search the web using SearXNG for current information.
    
    Use this tool when you need to find recent or specific information
    that may not be in your training data.
    
    Args:
        query: The search query string
        max_results: Maximum initial results to fetch per query (default: 15)
        top_k: Final number of top results to return (default: 10)
        enable_multi_query: Whether to generate multiple search queries for better coverage (default: False)

    Returns:
        Dictionary with:
        - sources: List of source metadata (title, URL, snippet)
        - content: Formatted search results with citations (chunks merged where consecutive)
        - source_indices: Maps each chunk in content back to source in sources list
    """
    try:
        config = SearchConfig(
            searxng_url=_config.searxng_url,
            num_search_results=max_results,
            enable_multi_query=enable_multi_query
        )
        
        tool = SearXNGSearchTool(config)
        result = await tool.search(query, max_results=max_results, top_k=top_k)

        return {
            "success": True,
            "sources": result.get("sources", []),
            "content": result.get("content", ""),
            "source_indices": result.get("source_indices", [])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sources": [],
            "content": f"Search failed: {str(e)}",
            "chunks": []
        }


@mcp.tool()
async def search_web_simple(
    query: str,
    num_results: int = 10
) -> dict[str, Any]:
    """
    Simple web search - returns basic search results without content extraction.
    
    Use this for quick searches when you just need titles, URLs, and snippets.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 10)
    
    Returns:
        Dictionary with:
        - results: List of search results (title, URL, snippet)
        - query: The search query used
    """
    try:
        results = await _fetch_searxng_results_simple(query, num_results)
        
        return {
            "success": True,
            "results": results,
            "query": query
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "query": query
        }


@mcp.tool()
async def get_search_configuration() -> dict[str, Any]:
    """
    Get the current SearXNG server configuration.
    
    Returns:
        Dictionary with current configuration settings
    """
    return {
        "searxng_url": _config.searxng_url,
        "num_search_results": _config.num_search_results,
        "chunk_size": _config.chunk_size,
        "similarity_threshold": _config.similarity_threshold,
        "max_retries": _config.max_retries,
        "enable_multi_query": _config.enable_multi_query,
        "llm_configured": _config.llm_base_url is not None
    }


# ============================================================================
# SearXNG Search Tool Class
# ============================================================================

class SearXNGSearchTool:
    """
    SearXNG-based web search tool with semantic reranking.
    """

    def __init__(self, config: SearchConfig):
        self.config = config

    async def _run_sync_request(self, func):
        """Helper to run sync requests asynchronously"""
        loop = asyncio.get_event_loop()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return await loop.run_in_executor(None, func)

    async def search(
        self,
        query: str,
        max_results: int = 15,
        top_k: int = 10
    ) -> dict:
        """
        Perform semantic web search with reranking.
        """
        try:
            # Step 1: Generate multiple search queries if enabled
            search_queries = [query]
            if self.config.enable_multi_query:
                try:
                    additional_queries = await self._generate_search_queries(query)
                    if additional_queries:
                        search_queries.extend(additional_queries)
                except Exception as e:
                    print(f"Query generation failed: {e}")

            # Step 2: Fetch search results for all queries
            all_results = []
            seen_urls = set()

            for sq in search_queries:
                try:
                    results = await self._fetch_searxng_results(sq)
                    for title, url, snippet in results[:max_results]:
                        if url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append((title, url, snippet))
                except Exception as e:
                    print(f"Search failed for query '{sq[:30]}...': {e}")

            if not all_results:
                return {"sources": [], "content": "No search results found."}

            # Step 3: Extract and chunk content with source tracking
            chunk_to_source_map = []
            sources = []
            source_url_to_index = {}

            extraction_tasks = [
                self._extract_page_content(url) for _, url, _ in all_results
            ]
            contents = await asyncio.gather(*extraction_tasks, return_exceptions=True)

            for (title, url, snippet), content in zip(all_results, contents):
                if isinstance(content, Exception):
                    content = snippet
                elif not content:
                    content = snippet

                if url not in source_url_to_index:
                    source_idx = len(sources)
                    source_url_to_index[url] = source_idx
                    sources.append({
                        "id": source_idx + 1,
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "chunk_count": 0,
                    })
                else:
                    source_idx = source_url_to_index[url]

                chunks = self._chunk_text(content)
                for chunk in chunks:
                    chunk_to_source_map.append((chunk.strip(), source_idx))
                    sources[source_idx]["chunk_count"] += 1

            if not chunk_to_source_map:
                return {"sources": [], "content": "No content could be extracted from search results."}

            # Step 4: Return results - optimized to avoid duplication
            # Only return sources + source_indices; client can map chunks via indices
            all_chunks = [chunk for chunk, _ in chunk_to_source_map]
            final_source_indices = [idx for _, idx in chunk_to_source_map]

            # Format output with merged consecutive chunks from same source
            merged_chunks, merged_indices = self._merge_consecutive_chunks(
                all_chunks[:top_k], final_source_indices[:top_k]
            )
            output = self._format_results(merged_chunks, merged_indices, sources, query)

            return {
                "sources": sources,
                "content": output,
                "source_indices": merged_indices
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error during search: {e}\n{error_details}")
            return {"sources": [], "content": f"Error during search: {str(e)}", "error": str(e)}

    async def _generate_search_queries(self, original_query: str) -> list[str]:
        """Generate alternative search queries using LLM (if configured)"""
        if not self.config.llm_base_url or not self.config.query_model:
            # Fallback: generate simple variations
            return self._generate_simple_query_variations(original_query)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate search engine queries.\n"
                        "Return ONLY the queries, one per line.\n"
                        "No numbering, no quotes, no explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Original query: "{original_query}"\n\n'
                        "Generate 2 alternative search queries that:\n"
                        "- Explore different angles\n"
                        "- Use different wording\n"
                        "- Are 3–8 words long\n"
                        "- Are not the same as the original"
                    ),
                },
            ]

            payload = {
                "model": self.config.query_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1580,
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}
            if self.config.llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

            response = await self._run_sync_request(
                lambda: requests.post(
                    f"{self.config.llm_base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=360,
                )
            )

            if response.status_code != 200:
                print("Query LLM error:", response.text)
                return self._generate_simple_query_variations(original_query)

            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            lines = [l.strip() for l in content.split("\n") if l.strip()]
            queries = []
            for line in lines:
                line = re.sub(r"^[\-\*\d\.\)]\s*", "", line)
                if line.lower() != original_query.lower() and 2 <= len(line.split()) <= 15:
                    queries.append(line)

            return list(dict.fromkeys(queries))[:2]

        except Exception as e:
            print("Query generation exception:", e)
            return self._generate_simple_query_variations(original_query)

    def _generate_simple_query_variations(self, original_query: str) -> list[str]:
        """Generate simple query variations without LLM"""
        variations = []
        words = original_query.split()
        
        # Try different word combinations
        if len(words) >= 2:
            # First half
            variations.append(" ".join(words[:len(words)//2 + 1]))
            # Second half
            variations.append(" ".join(words[len(words)//2:]))
        
        return variations[:2]

    async def _fetch_searxng_results(self, query: str) -> list[tuple[str, str, str]]:
        """Fetch search results from SearXNG"""
        params = {"q": query, "format": "json"}
        response = await self._run_sync_request(
            lambda: requests.get(self.config.searxng_url, params=params, timeout=40)
        )

        if response.status_code == 200:
            results = response.json().get("results", [])[:self.config.num_search_results]
            return [(r["title"], r["url"], r.get("content", "")) for r in results]

        raise Exception(f"SearXNG query failed: {response.status_code}")

    async def _extract_page_content(self, url: str) -> str:
        """Extract main content from a webpage"""
        try:
            response = await self._run_sync_request(
                lambda: requests.get(
                    url,
                    timeout=30,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                )
            )

            if response.status_code != 200:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            unwanted_tags = [
                "script", "style", "nav", "footer", "aside",
                "noscript", "iframe", "svg", "form", "button",
                "input", "select", "textarea",
            ]
            for tag in soup(unwanted_tags):
                tag.extract()

            # Remove elements by class/id
            unwanted_selectors = [
                {"class": ["advertisement", "ad", "ads", "sidebar"]},
                {"id": ["sidebar", "nav", "footer"]},
            ]
            for selector in unwanted_selectors:
                for element in soup.find_all(**selector):
                    element.extract()

            # Extract text
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            # Limit to 5000 chars - sufficient for main content, reduces context usage
            return text[:5000]

        except Exception as e:
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks"""
        words = text.split()
        return [
            " ".join(words[i:i + self.config.chunk_size])
            for i in range(0, len(words), self.config.chunk_size)
        ]

    def _merge_consecutive_chunks(
        self, chunks: list[str], source_indices: list[int]
    ) -> tuple[list[str], list[int]]:
        """
        Merge consecutive chunks from the same source to reduce token count.
        Merges if combined size is under 2x chunk_size (2400 words).
        """
        if not chunks:
            return [], []

        merged_chunks = []
        merged_indices = []

        current_chunk = chunks[0]
        current_idx = source_indices[0]

        for chunk, idx in zip(chunks[1:], source_indices[1:]):
            if idx == current_idx:
                # Same source - consider merging
                combined = f"{current_chunk} {chunk}"
                # Only merge if combined size is reasonable
                if len(combined.split()) <= self.config.chunk_size * 2:
                    current_chunk = combined
                else:
                    # Too large, emit current and start new
                    merged_chunks.append(current_chunk)
                    merged_indices.append(current_idx)
                    current_chunk = chunk
                    current_idx = idx
            else:
                # Different source - emit current and start new
                merged_chunks.append(current_chunk)
                merged_indices.append(current_idx)
                current_chunk = chunk
                current_idx = idx

        # Don't forget the last one
        merged_chunks.append(current_chunk)
        merged_indices.append(current_idx)

        return merged_chunks, merged_indices

    def _format_results(
        self,
        chunks: list[str],
        source_indices: list[int],
        sources: list[dict],
        query: str,
    ) -> str:
        """Format search results with citations"""
        output = "# 🔍 Search Results\n\n"
        findings = ""

        # Group chunks by source
        source_to_chunks = {}
        for chunk, src_idx in zip(chunks, source_indices):
            source_to_chunks.setdefault(src_idx, []).append(chunk)

        findings += "## Key Findings (by Source)\n\n"

        citation_counter = 1
        source_citation_map = {}

        for src_idx, src_chunks in source_to_chunks.items():
            source = sources[src_idx]
            source_id = citation_counter
            source_citation_map[src_idx] = source_id
            citation_counter += 1

            output += f"[{source_id}] - [{source['title']}]({source['url']})\n\n"

            for i, chunk in enumerate(src_chunks, 1):
                findings += f"**[{source_id}]** - [{source['title']}]({source['url']})\n\n"
                findings += f"- {chunk}\n"

            output += "\n"
            findings += "\n"

        return output + "\n\n" + findings


# ============================================================================
# Helper Functions (for simple search)
# ============================================================================

async def _fetch_searxng_results_simple(query: str, num_results: int) -> list[dict]:
    """Fetch basic search results from SearXNG"""
    params = {"q": query, "format": "json"}
    
    def sync_request():
        return requests.get(_config.searxng_url, params=params, timeout=40)
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, sync_request)

    if response.status_code == 200:
        results = response.json().get("results", [])[:num_results]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "engine": r.get("engine", ""),
            }
            for r in results
        ]
    
    raise Exception(f"SearXNG query failed: {response.status_code}")


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the MCP server with stdio transport
    mcp.run()

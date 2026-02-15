"""
SearXNG Web Search Tool - Adapter for the open-webui tool.

This module adapts the open-webui-tool.py SearXNG search functionality
for use with the LLM UI application.
"""

import requests
import numpy as np
from bs4 import BeautifulSoup
import re
import asyncio
import json
from typing import List, Dict, Tuple, Callable, Any
from dataclasses import dataclass, field
from config import LLAMA_CPP_BASE_URL, LLAMA_CPP_MODEL, QUERY_MODEL
from tools.base import SharedLLMUtils


@dataclass
class SearchConfig:
    """Configuration for SearXNG search"""
    searxng_url: str = "http://localhost:8888/search"
    embeddings_api: str = f"{LLAMA_CPP_BASE_URL}/v1/embeddings"
    rerank_api: str = f"{LLAMA_CPP_BASE_URL}/v1/rerank"
    llm_base_url: str = LLAMA_CPP_BASE_URL
    llm_model: str = LLAMA_CPP_MODEL
    query_model: str = QUERY_MODEL
    llm_api_key: str = "sk-12"
    num_search_results: int = 25
    chunk_size: int = 1200
    similarity_threshold: float = 0.4
    max_retries: int = 3
    enable_multi_query: bool = True


class SearXNGSearchTool:
    """
    SearXNG-based web search tool with semantic reranking.
    
    This tool performs multi-query web searches, extracts content from
    results, and uses embeddings + reranking to return the most relevant
    information.
    """
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.model_lock = asyncio.Lock()
    
    async def search(
        self,
        query: str,
        max_results: int = 30,
        top_k: int = 22,
        progress_callback = None
    ) -> Dict:
        """
        Perform semantic web search with reranking.
        
        Args:
            query: The search query string
            max_results: Maximum initial results to fetch per query
            top_k: Final number of top chunks to return
            progress_callback: Optional async callback for progress updates (status, progress)
            
        Returns:
            Dict with 'sources' and 'content' keys
        """
        try:
            # Helper to call progress callback (supports both sync and async)
            async def report_progress(status: str, progress: int):
                if progress_callback:
                    try:
                        result = progress_callback(status, progress)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        print(f"Progress callback error: {e}")
            
            # Step 1: Generate multiple search queries if enabled
            search_queries = [query]
            if self.config.enable_multi_query:
                await report_progress("Generating optimized search queries...", 5)
                
                try:
                    additional_queries = await self._generate_search_queries(query)
                    if additional_queries:
                        search_queries.extend(additional_queries)
                        await report_progress(f"Generated {len(additional_queries)} additional queries: {additional_queries}", 10)
                except Exception as e:
                    print(f"Query generation failed: {e}")

            # Step 2: Fetch search results for all queries with adaptive query generation
            await report_progress(f"Searching with {len(search_queries)} queries...", 15)
            
            all_results = []
            seen_urls = set()
            
            for sq in search_queries:
                try:
                    results = await self._fetch_searxng_results(sq)
                    
                    # Generate adaptive follow-up queries based on initial results
                    adaptive_queries = []
                    if self.config.enable_multi_query and results:
                        await report_progress("Generating adaptive follow-up queries...", 18)
                        
                        try:
                            adaptive_queries = await self._generate_adaptive_queries_from_results(
                                query, results
                            )
                            if adaptive_queries:
                                await report_progress(f"Generated {len(adaptive_queries)} adaptive queries: {adaptive_queries}", 20)
                        except Exception as e:
                            print(f"Adaptive query generation failed: {e}")
                    
                    # Search with adaptive queries
                    for aq in adaptive_queries:
                        try:
                            adaptive_results = await self._fetch_searxng_results(aq)
                            for title, url, snippet in adaptive_results[:max_results]:
                                if url not in seen_urls:
                                    seen_urls.add(url)
                                    all_results.append((title, url, snippet))
                        except Exception as e:
                            print(f"Adaptive search failed for '{aq[:30]}...': {e}")
                    
                    # Add results from original query
                    for title, url, snippet in results[:max_results]:
                        if url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append((title, url, snippet))
                            
                except Exception as e:
                    print(f"Search failed for query '{sq[:30]}...': {e}")
            
            if not all_results:
                return {"sources": [], "content": "No search results found."}
            
            await report_progress(f"Found {len(all_results)} unique results. Extracting content...", 25)
            
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
                        "snippet": "",  # Will be updated with first chunk
                        "chunk_count": 0,
                    })
                else:
                    source_idx = source_url_to_index[url]
                
                chunks = self._chunk_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_to_source_map.append((chunk.strip(), source_idx))
                    sources[source_idx]["chunk_count"] += 1
                    # Store first chunk as snippet
                    if i == 0 and not sources[source_idx]["snippet"]:
                        sources[source_idx]["snippet"] = chunk[:300] + "..." if len(chunk) > 300 else chunk
            
            if not chunk_to_source_map:
                return {"sources": [], "content": "No content could be extracted from search results."}
            
            await report_progress(f"Created {len(chunk_to_source_map)} chunks from {len(sources)} sources", 40)
            
            # Step 4: Embed query and chunks
            await report_progress("Computing embeddings...", 50)
            
            query_emb = await self._get_embedding_async(query)
            
            all_chunks = [chunk for chunk, _ in chunk_to_source_map]
            
            # Process embeddings in batches to avoid overwhelming the API
            batch_size = 10
            chunk_embs = []
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embs = await asyncio.gather(
                    *[self._get_embedding_async(c) for c in batch]
                )
                chunk_embs.extend(batch_embs)
                # Report progress during embedding
                progress = 50 + int(15 * (i + len(batch)) / len(all_chunks))
                await report_progress(f"Computing embeddings... {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}", progress)
            
            await report_progress("Calculating semantic similarities...", 65)
            
            # Step 5: Filter by similarity
            similarities = [
                self._cosine_similarity(query_emb, emb) for emb in chunk_embs
            ]
            
            indexed_sims = sorted(
                enumerate(similarities), key=lambda x: x[1], reverse=True
            )
            candidates_idx = [
                i for i, sim in indexed_sims if sim >= self.config.similarity_threshold
            ][:top_k * 3]
            
            candidates = [all_chunks[i] for i in candidates_idx]
            candidates_source_idx = [chunk_to_source_map[i][1] for i in candidates_idx]
            
            if not candidates:
                return {"sources": [], "content": "No relevant content found matching your query."}
            
            await report_progress(f"Filtered to {len(candidates)} candidates. Reranking...", 75)
            
            # Step 6: Rerank
            reranked_indices = await self._rerank_async_with_indices(query, candidates)
            
            final_chunks = []
            final_source_indices = []
            for idx in reranked_indices[:top_k]:
                final_chunks.append(candidates[idx])
                final_source_indices.append(candidates_source_idx[idx])
            
            await report_progress("Formatting results...", 90)
            
            # Step 7: Format output
            output = self._format_results(final_chunks, final_source_indices, sources, query)
            
            if progress_callback:
                progress_callback("Search complete!", 100)
            
            return {
                "sources": sources,
                "content": output,
                "chunks": final_chunks,
                "source_indices": final_source_indices
            }
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error during search: {e}\n{error_details}")
            return {"sources": [], "content": f"Error during search: {str(e)}", "error": str(e)}
    
    async def _generate_search_queries(self, original_query: str) -> List[str]:
        """Generate alternative search queries using LLM"""
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
                "max_tokens": 580,
                "stream": False,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.config.llm_base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                ),
            )
            
            if response.status_code != 200:
                print("Query LLM error:", response.text)
                return []
            
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
                if (
                    line.lower() != original_query.lower()
                    and 2 <= len(line.split()) <= 15
                ):
                    queries.append(line)
            
            return list(dict.fromkeys(queries))[:2]
        
        except Exception as e:
            print("Query generation exception:", e)
            return []
    
    async def _generate_adaptive_queries_from_results(
        self,
        original_query: str,
        search_results: List[Tuple[str, str, str]],
    ) -> List[str]:
        """
        Generate follow-up search queries based on initial search results.
        
        This analyzes the initial search results and generates additional
        queries to explore missing angles or deeper aspects.
        """
        # Build compact context (titles + snippets only)
        context_lines = []
        for title, _, snippet in search_results[:8]:
            line = f"- {title}: {snippet[:180]}"
            context_lines.append(line)
        
        context = "\n".join(context_lines)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You improve web search coverage.\n"
                    "Generate follow-up search queries ONLY.\n"
                    "One query per line. No numbering. No explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Original query: "{original_query}"\n\n'
                    "Here are summaries of the initial search results:\n"
                    f"{context}\n\n"
                    "Generate 2 NEW search queries that:\n"
                    "- Cover missing angles or deeper aspects\n"
                    "- Use different terminology\n"
                    "- Are not already answered by the above results\n"
                    "- Are 3–8 words long"
                ),
            },
        ]
        
        payload = {
            "model": self.config.query_model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 1080,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.config.llm_base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                ),
            )
            
            if response.status_code != 200:
                print("Adaptive query LLM error:", response.text)
                return []
            
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            
            queries = []
            for line in content.split("\n"):
                line = re.sub(r"^[\-\*\d\.\)]\s*", "", line.strip())
                if (
                    line
                    and line.lower() != original_query.lower()
                    and 2 <= len(line.split()) <= 15
                ):
                    queries.append(line)
            return list(dict.fromkeys(queries))[:2]
        
        except Exception as e:
            print("Adaptive query generation exception:", e)
            return []
    
    async def _fetch_searxng_results(self, query: str) -> List[Tuple[str, str, str]]:
        """Fetch search results from SearXNG"""
        params = {"q": query, "format": "json", "categories": "general"}
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(self.config.searxng_url, params=params, timeout=40),
        )
        
        if response.status_code == 200:
            results = response.json().get("results", [])[:self.config.num_search_results]
            return [(r["title"], r["url"], r.get("content", "")) for r in results]
        
        raise Exception(f"SearXNG query failed: {response.status_code}")
    
    async def _extract_page_content(self, url: str) -> str:
        """Extract main content from a webpage"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    url,
                    timeout=20,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                ),
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
            
            return text[:10000]
        
        except Exception as e:
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        return [
            " ".join(words[i:i + self.config.chunk_size])
            for i in range(0, len(words), self.config.chunk_size)
        ]
    
    async def _get_embedding_async(self, text: str) -> np.ndarray:
        """Get embedding vector for text with retry logic"""
        return await SharedLLMUtils.get_embedding(text, max_retries=self.config.max_retries)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return SharedLLMUtils.cosine_similarity(a, b)
    
    async def _rerank_async_with_indices(
        self, query: str, chunks: List[str]
    ) -> List[int]:
        """Rerank chunks and return indices in new order"""
        async with self.model_lock:
            return await SharedLLMUtils.rerank(query, chunks, max_retries=self.config.max_retries)
    
    def _format_results(
        self,
        chunks: List[str],
        source_indices: List[int],
        sources: List[Dict],
        query: str,
    ) -> str:
        """Format search results with citations"""
        output = "# 🔍 Search Results\n\n"
        findings = ""
        
        # Group chunks by source
        source_to_chunks = {}
        for chunk, src_idx in zip(chunks, source_indices):
            source_to_chunks.setdefault(src_idx, []).append(chunk)
        
        findings += "## 🧠 Key Findings (by Source)\n\n"
        
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


# Tool definition for LLM function calling
SEARXNG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information. Use this when you need to find recent or specific information that may not be in your training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 15)",
                    "default": 15
                }
            },
            "required": ["query"]
        }
    }
}

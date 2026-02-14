"""
title: Smart SearXNG Search (Simple Citations)
author: converted from MCP server
version: 2.1.0
description: Multi-query semantic web search with simple citation format (no event_call dependency)
required_open_webui_version: 0.3.9
"""

import requests
import numpy as np
from bs4 import BeautifulSoup
import re
import asyncio
import json
from typing import Callable, Any, List, Dict, Tuple
from pydantic import BaseModel, Field, ConfigDict

# Configuration
SEARXNG_URL = "http://localhost:8888/search"
EMBEDDINGS_API = "http://localhost:8080/v1/embeddings"
RERANK_API = "http://localhost:8080/v1/rerank"
CHUNK_SIZE = 1000


class Tools:
    class Valves(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        SEARXNG_URL: str = Field(
            default="http://localhost:8888/search", description="SearXNG instance URL"
        )
        EMBEDDINGS_API: str = Field(
            default="http://localhost:8080/v1/embeddings",
            description="Embeddings API endpoint",
        )
        RERANK_API: str = Field(
            default="http://localhost:8080/v1/rerank",
            description="Reranking API endpoint",
        )
        LLM_BASE_URL: str = Field(
            default="http://localhost:8080/v1", description="OpenAI-compatible base URL"
        )
        LLM_MODEL: str = Field(
            default="glm4.7-30ba3b", description="Model for query generation"
        )
        LLM_API_KEY: str = Field(default="sk-12", description="Optional API key")
        NUM_SEARCH_RESULTS: int = Field(
            default=25, description="Maximum initial search results to fetch"
        )
        CHUNK_SIZE: int = Field(
            default=1000, description="Size of text chunks in words"
        )
        SIMILARITY_THRESHOLD: float = Field(
            default=0.4, description="Minimum cosine similarity threshold"
        )
        MAX_RETRIES: int = Field(default=3, description="Maximum retries for API calls")
        ENABLE_MULTI_QUERY: bool = Field(
            default=True,
            description="Generate multiple search queries for better coverage",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.model_lock = asyncio.Lock()

    async def smart_web_search(
        self,
        query: str,
        max_results: int = 30,
        top_k: int = 22,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Perform semantic web search with reranking via SearXNG and local embeddings.

        :param query: The search query string
        :param max_results: Maximum initial results to fetch per query (default: 20)
        :param top_k: Final number of top chunks to return (default: 15)
        :return: Formatted search results with citations
        """

        try:
            # Emit initial status
            await self._emit_status(
                __event_emitter__, "Starting smart web search...", False
            )

            # Step 1: Generate multiple search queries if enabled
            search_queries = [query]
            if self.valves.ENABLE_MULTI_QUERY:
                await self._emit_status(
                    __event_emitter__, "Generating optimized search queries...", False
                )

                try:
                    additional_queries = await self._generate_search_queries(query)
                    if additional_queries:
                        search_queries.extend(additional_queries)
                        await self._emit_status(
                            __event_emitter__,
                            f"✓ Searched for additional queries {additional_queries}",
                            False,
                        )
                    else:
                        await self._emit_status(
                            __event_emitter__,
                            "⚠ Query generation returned empty, using original query only",
                            False,
                        )
                except Exception as e:
                    await self._emit_status(
                        __event_emitter__,
                        f"⚠ Query generation failed, using original query: {str(e)[:50]}",
                        False,
                    )

            # Step 2: Fetch search results for all queries
            await self._emit_status(
                __event_emitter__,
                f"Searching with {len(search_queries)} queries...",
                False,
            )

            all_results = []
            seen_urls = set()

            for sq in search_queries:
                try:
                    results = await self._fetch_searxng_results(sq)
                    adaptive_queries = []
                    if self.valves.ENABLE_MULTI_QUERY and results:
                        await self._emit_status(
                            __event_emitter__,
                            "Generating adaptive follow-up queries...",
                            False,
                        )

                        adaptive_queries = (
                            await self._generate_adaptive_queries_from_results(
                                query, results
                            )
                        )
                        if adaptive_queries:
                            await self._emit_status(
                                __event_emitter__,
                                f"Searched for {adaptive_queries}",
                                False,
                            )
                    search_queries = [query] + adaptive_queries
                    for title, url, snippet in results[:max_results]:
                        if url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append((title, url, snippet))

                except Exception as e:
                    await self._emit_status(
                        __event_emitter__,
                        f"⚠ Search failed for query '{sq[:30]}...': {str(e)[:50]}",
                        False,
                    )

            results = all_results

            if not results:
                return "No search results found."

            await self._emit_status(
                __event_emitter__,
                f"✓ Found {len(results)} unique results. Extracting content...",
                False,
            )

            # Step 3: Extract and chunk content with source tracking
            chunk_to_source_map = []  # List of (chunk_text, source_index)
            sources = []  # List of source dicts
            source_url_to_index = {}  # Map URL to source index

            extraction_tasks = [
                self._extract_page_content(url) for title, url, snippet in results
            ]
            contents = await asyncio.gather(*extraction_tasks, return_exceptions=True)

            for (title, url, snippet), content in zip(results, contents):
                if isinstance(content, Exception):
                    content = snippet
                elif not content:
                    content = snippet

                # Get or create source index
                if url not in source_url_to_index:
                    source_idx = len(sources)
                    source_url_to_index[url] = source_idx
                    sources.append(
                        {
                            "id": source_idx + 1,
                            "title": title,
                            "url": url,
                            "chunk_count": 0,
                        }
                    )
                else:
                    source_idx = source_url_to_index[url]

                # Create chunks and track their source
                chunks = self._chunk_text(content)
                for chunk in chunks:
                    chunk_to_source_map.append((chunk.strip(), source_idx))
                    sources[source_idx]["chunk_count"] += 1

            if not chunk_to_source_map:
                return "No content could be extracted from search results."

            await self._emit_status(
                __event_emitter__,
                f"✓ Created {len(chunk_to_source_map)} chunks from {len(sources)} sources",
                False,
            )

            await self._emit_status(__event_emitter__, "Computing embeddings...", False)

            # Step 4: Embed query and chunks
            query_emb = await self._get_embedding_async(query)

            all_chunks = [chunk for chunk, _ in chunk_to_source_map]
            chunk_embs = await asyncio.gather(
                *[self._get_embedding_async(c) for c in all_chunks]
            )

            await self._emit_status(
                __event_emitter__, "Calculating semantic similarities...", False
            )

            # Step 5: Filter by similarity
            similarities = [
                self._cosine_similarity(query_emb, emb) for emb in chunk_embs
            ]

            indexed_sims = sorted(
                enumerate(similarities), key=lambda x: x[1], reverse=True
            )
            candidates_idx = [
                i for i, sim in indexed_sims if sim >= self.valves.SIMILARITY_THRESHOLD
            ][
                : top_k * 3
            ]  # Get more candidates for reranking

            candidates = [all_chunks[i] for i in candidates_idx]
            candidates_source_idx = [chunk_to_source_map[i][1] for i in candidates_idx]

            if not candidates:
                return "No relevant content found matching your query."

            await self._emit_status(
                __event_emitter__,
                f"✓ Filtered to {len(candidates)} candidates. Reranking...",
                False,
            )

            # Step 6: Rerank
            reranked_indices = await self._rerank_async_with_indices(query, candidates)

            # Apply reranking and limit to top_k
            final_chunks = []
            final_source_indices = []
            for idx in reranked_indices[:top_k]:
                final_chunks.append(candidates[idx])
                final_source_indices.append(candidates_source_idx[idx])

            source_to_chunks = {}

            for chunk, src_idx in zip(final_chunks, final_source_indices):
                source_to_chunks.setdefault(src_idx, []).append(chunk)

            await self._emit_status(__event_emitter__, "Formatting results...", False)

            # Step 7: Format output with citations
            output = self._format_results_with_citations(
                final_chunks, final_source_indices, sources, query
            )

            await self._emit_status(__event_emitter__, "✓ Search complete!", True)

            return output

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            error_msg = f"Error during smart web search: {str(e)}\n\nDetails:\n{error_details[:500]}"
            await self._emit_status(__event_emitter__, f"❌ {str(e)}", True)
            return error_msg

    async def _emit_status(self, emitter, description: str, done: bool):
        """Helper to emit status updates"""
        if emitter:
            try:
                await emitter(
                    {
                        "type": "status",
                        "data": {"description": description, "done": done},
                    }
                )
            except Exception as e:
                print(f"Failed to emit status: {e}")

    async def _generate_search_queries(self, original_query: str) -> List[str]:
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
                "model": self.valves.LLM_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 580,
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}
            if self.valves.LLM_API_KEY:
                headers["Authorization"] = f"Bearer {self.valves.LLM_API_KEY}"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.valves.LLM_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                ),
            )

            if response.status_code != 200:
                print("Query LLM error:", response.text)
                return []
            print(response.json())
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
        Generate follow-up search queries based on initial search results
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
            "model": self.valves.LLM_MODEL,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 980,
        }

        headers = {"Content-Type": "application/json"}
        if self.valves.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LLM_API_KEY}"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{self.valves.LLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            ),
        )

        if response.status_code != 200:
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

    async def _fetch_searxng_results(self, query: str) -> List[tuple]:
        """Fetch search results from SearXNG"""
        params = {"q": query, "format": "json", "categories": "general"}
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(self.valves.SEARXNG_URL, params=params, timeout=30),
        )

        if response.status_code == 200:
            results = response.json().get("results", [])[
                : self.valves.NUM_SEARCH_RESULTS
            ]
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
                    timeout=10,
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
                "script",
                "style",
                "nav",
                "footer",
                "aside",
                "noscript",
                "iframe",
                "svg",
                "form",
                "button",
                "input",
                "select",
                "textarea",
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
            " ".join(words[i : i + self.valves.CHUNK_SIZE])
            for i in range(0, len(words), self.valves.CHUNK_SIZE)
        ]

    async def _get_embedding_async(self, text: str):
        """Get embedding vector for text with retry logic"""
        for attempt in range(self.valves.MAX_RETRIES):
            try:
                payload = {"input": text, "model": "qwen3-embedding"}
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        self.valves.EMBEDDINGS_API, json=payload, timeout=60
                    ),
                )

                if response.status_code == 200:
                    return np.array(response.json()["data"][0]["embedding"])

                if attempt < self.valves.MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)

            except Exception as e:
                if attempt < self.valves.MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        raise Exception(f"Embedding failed after {self.valves.MAX_RETRIES} attempts")

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def _rerank_async_with_indices(
        self, query: str, chunks: List[str]
    ) -> List[int]:
        """Rerank chunks and return indices in new order"""
        async with self.model_lock:
            for attempt in range(self.valves.MAX_RETRIES):
                try:
                    payload = {
                        "model": "qwen3-reranker",
                        "query": query,
                        "documents": chunks,
                    }

                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.post(
                            self.valves.RERANK_API, json=payload, timeout=130
                        ),
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # Handle different response formats
                        if "results" in result:
                            results = result["results"]
                            sorted_results = sorted(
                                results,
                                key=lambda x: x.get("relevance_score", 0),
                                reverse=True,
                            )
                            return [r["index"] for r in sorted_results]
                        elif "scores" in result:
                            scores = result["scores"]
                            return list(np.argsort(scores)[::-1])
                        elif "data" in result:
                            data = result["data"]
                            sorted_data = sorted(
                                data, key=lambda x: x.get("score", 0), reverse=True
                            )
                            return [d["index"] for d in sorted_data]
                        else:
                            return list(range(len(chunks)))

                    if attempt < self.valves.MAX_RETRIES - 1:
                        await asyncio.sleep(2**attempt)

                except Exception as e:
                    if attempt < self.valves.MAX_RETRIES - 1:
                        await asyncio.sleep(2)
                    else:
                        return list(range(len(chunks)))

    def _format_results_with_citations(
        self,
        chunks: List[str],
        source_indices: List[int],
        sources: List[Dict],
        query: str,
    ) -> str:
        output = "# 🔍 Search Results\n\n"
        # output += f"**Query:** {query}\n"
        o2 = ""

        # Group chunks by source
        source_to_chunks = {}
        for chunk, src_idx in zip(chunks, source_indices):
            source_to_chunks.setdefault(src_idx, []).append(chunk)

        o2 += "## 🧠 Key Findings (by Source)\n\n"

        citation_counter = 1
        source_citation_map = {}

        for src_idx, src_chunks in source_to_chunks.items():
            source = sources[src_idx]
            source_id = citation_counter
            source_citation_map[src_idx] = source_id
            citation_counter += 1

            # output += f"### [{source_id}] {source['title']}\n"
            # output += f"{source['url']}\n\n"
            output += f"[{source_id}] - " f"[{source['title']}]({source['url']})\n\n"

            for i, chunk in enumerate(src_chunks, 1):
                o2 += (
                    f"**[{source_id}]** - " f"[{source['title']}]({source['url']})\n\n"
                )
                o2 += f"- {chunk}\n"

            output += "\n"
            o2 += "\n"

        return output + "\n\n" + o2

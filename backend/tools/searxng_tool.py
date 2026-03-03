"""
SearXNG Web Search Tool - Adapter for the open-webui tool.

This module adapts the open-webui-tool.py SearXNG search functionality
for use with the LLM UI application.

Features:
1. LLM-based search term extraction from user queries (handles paragraphs)
2. Thinking model support - waits for thinking to complete before extracting search terms
3. Reasoning over search results to determine if additional searches are needed
"""

import requests
import numpy as np
from bs4 import BeautifulSoup
import re
import asyncio
import json
import inspect
import warnings
from typing import List, Dict, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
from tools.base import SharedLLMUtils
from backend.settings import settings_manager


@dataclass
class SearchConfig:
    """Configuration for SearXNG search"""
    searxng_url: str = "http://localhost:8888/search"
    embeddings_api: str = "http://localhost:8080/v1/embeddings"
    rerank_api: str = "http://localhost:8080/v1/rerank"
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen3-4b"
    query_model: str = "qwen3-4b"
    llm_api_key: str = "sk-12"
    num_search_results: int = 20
    chunk_size: int = 1200
    similarity_threshold: float = 0.4
    max_retries: int = 3
    enable_multi_query: bool = True
    enable_search_term_extraction: bool = True  # Extract search terms from paragraphs
    enable_result_reasoning: bool = True  # Reason over results to determine if more searches needed
    max_search_iterations: int = 3  # Maximum iterations of search + reasoning
    thinking_timeout: int = 120  # Timeout for thinking models in seconds
    embedding_timeout: int = 120  # Timeout for embedding requests
    #embedding_batch_size: int = 5  # Max concurrent embedding requests
    #embedding_max_retries: int = 5  # Max retries for embedding requests

    @classmethod
    def from_settings(cls, settings: dict = None):
        """Create SearchConfig from settings"""
        if settings is None:
            settings = settings_manager.get_settings()
        
        # Get LLM settings from settings manager
        llm_base_url = settings.get('llama_cpp_base_url', cls.llm_base_url)
        llm_model = settings.get('llama_cpp_model', cls.llm_model)
        query_model = settings.get('query_model', cls.query_model)
        
        return cls(
            searxng_url=settings.get('searxng_url', cls.searxng_url),
            embeddings_api=f"{llm_base_url}/embeddings",
            rerank_api=f"{llm_base_url}/rerank",
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            query_model=query_model,
            num_search_results=settings.get('searxng_num_results', cls.num_search_results),
            chunk_size=settings.get('searxng_chunk_size', cls.chunk_size),
            similarity_threshold=settings.get('searxng_similarity_threshold', cls.similarity_threshold),
            max_retries=settings.get('searxng_max_retries', cls.max_retries),
            enable_multi_query=settings.get('searxng_enable_multi_query', cls.enable_multi_query),
        )


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

    async def _run_sync_request(self, func):
        """Helper to run sync requests with suppressed deprecation warnings"""
        loop = asyncio.get_event_loop()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return await loop.run_in_executor(None, func)
    
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
                        if inspect.iscoroutinefunction(progress_callback):
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
                        "snippet": snippet,  # Store the original snippet
                        "chunk_content": "",  # Will be updated with first chunk
                        "chunk_count": 0,
                    })
                else:
                    source_idx = source_url_to_index[url]

                chunks = self._chunk_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_to_source_map.append((chunk.strip(), source_idx))
                    sources[source_idx]["chunk_count"] += 1
                    # Store first chunk as chunk_content
                    if i == 0 and not sources[source_idx]["chunk_content"]:
                        sources[source_idx]["chunk_content"] = chunk[:500] + "..." if len(chunk) > 500 else chunk
            
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
                await progress_callback("Search complete!", 100)
            
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
            # Suppress deprecation warnings from run_in_executor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{self.config.llm_base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=160,
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

    async def _extract_search_terms_from_query(self, user_query: str) -> List[str]:
        """
        Extract focused search terms from a user query (especially paragraphs).
        
        This method uses an LLM to analyze the user's query and extract concise
        search terms that can be used for web search. It handles:
        - Long paragraphs by extracting key concepts
        - Multiple questions by identifying the core topics
        - Complex queries by breaking them into searchable terms
        
        Args:
            user_query: The original user query (can be a paragraph)
            
        Returns:
            List of search terms (1-3 terms)
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You extract search terms from user queries.\n"
                        "Your task is to identify the KEY search terms that should be used for web search.\n\n"
                        "Guidelines:\n"
                        "- Extract 1-3 focused search terms (2-6 words each)\n"
                        "- Focus on the main topic/concept, not the question format\n"
                        "- Remove conversational elements\n"
                        "- Keep technical terms intact\n"
                        "- Return ONLY the search terms, one per line\n"
                        "- No explanations, no numbering, no quotes"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'User query: "{user_query}"\n\n'
                        "Extract the key search terms for web search:"
                    ),
                },
            ]
            
            payload = {
                "model": self.config.query_model,
                "messages": messages,
                "temperature": 0.3,  # Lower temperature for more focused extraction
                "max_tokens": 1200,
                "stream": False,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

            # Run with timeout to avoid blocking
            response = await asyncio.wait_for(
                self._run_sync_request(
                    lambda: requests.post(
                        f"{self.config.llm_base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=160,
                    )
                ),
                timeout=self.config.thinking_timeout
            )
            
            if response.status_code != 200:
                print("Search term extraction LLM error:", response.text)
                return [user_query]  # Fallback to original query
            
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            
            # Parse extracted terms
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            
            search_terms = []
            for line in lines:
                # Clean up the line
                line = re.sub(r"^[\-\*\d\.\)]\s*", "", line)
                line = line.strip('"\'')
                
                # Validate term length
                if 2 <= len(line.split()) <= 10:
                    search_terms.append(line)
            
            # Return extracted terms or fallback to original
            return search_terms[:3] if search_terms else [user_query]
            
        except asyncio.TimeoutError:
            print("Search term extraction timed out")
            return [user_query]
        except Exception as e:
            print(f"Search term extraction exception: {e}")
            return [user_query]

    async def _wait_for_thinking_completion(
        self, 
        messages: List[Dict], 
        progress_callback=None
    ) -> str:
        """
        Wait for a thinking model to complete its reasoning before proceeding.
        
        This method handles models that output thinking/reasoning content
        before the actual response. It waits for the thinking to complete
        and returns the final non-thinking content.
        
        Args:
            messages: The conversation messages
            progress_callback: Optional callback for progress updates
            
        Returns:
            The final content after thinking completes
        """
        try:
            from tools.base import SharedLLMUtils
            
            # Report progress
            if progress_callback:
                progress_callback("Model is thinking...", 5)
            
            # Make the LLM call and wait for complete response
            payload = {
                "model": self.config.query_model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1500,
                "stream": False,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
            
            loop = asyncio.get_event_loop()
            
            # Run with timeout
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{self.config.llm_base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=160,
                    ),
                ),
                timeout=self.config.thinking_timeout
            )
            
            if response.status_code != 200:
                print("Thinking completion LLM error:", response.text)
                return ""
            
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            
            # Remove any <think> tags if present in the response
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()
            
            if progress_callback:
                progress_callback("Thinking complete", 10)
            
            return content
            
        except asyncio.TimeoutError:
            print("Thinking completion timed out")
            if progress_callback:
                progress_callback("Thinking timed out, proceeding...", 10)
            return ""
        except Exception as e:
            print(f"Thinking completion exception: {e}")
            if progress_callback:
                progress_callback("Thinking error, proceeding...", 10)
            return ""

    async def _reason_over_search_results(
        self,
        original_query: str,
        search_results: Dict,
        progress_callback=None
    ) -> Dict:
        """
        Reason over search results to determine if additional searches are needed.
        
        This method analyzes the search results and determines:
        1. Whether the results adequately answer the user's query
        2. What aspects are missing or need more specific information
        3. Whether additional searches should be performed
        4. What specific follow-up searches would be most helpful
        
        Args:
            original_query: The original user query
            search_results: The search results dict with 'sources' and 'content'
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with keys:
            - 'needs_more_search': bool
            - 'reasoning': str (explanation of the decision)
            - 'follow_up_queries': List[str] (suggested queries if more search needed)
            - 'coverage_score': float (0-1, how well results cover the query)
        """
        try:
            if progress_callback:
                progress_callback("Analyzing search results coverage...", 85)
            
            # Build compact summary of results
            sources = search_results.get("sources", [])
            content = search_results.get("content", "")
            
            # Create summary of what we found
            result_summary = []
            for source in sources[:5]:  # Limit to top 5 sources
                title = source.get("title", "")
                snippet = source.get("snippet", "")[:200]
                result_summary.append(f"- {title}: {snippet}")
            
            results_text = "\n".join(result_summary)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You analyze search results to determine if they adequately answer a user's query.\n\n"
                        "Your task:\n"
                        "1. Evaluate if the search results cover the key aspects of the query\n"
                        "2. Identify what information is missing or incomplete\n"
                        "3. Decide if additional searches are needed\n"
                        "4. If yes, suggest specific follow-up search queries\n\n"
                        "Respond in this EXACT JSON format:\n"
                        '{\n'
                        '  "needs_more_search": true/false,\n'
                        '  "reasoning": "Your analysis of result coverage",\n'
                        '  "follow_up_queries": ["query1", "query2"],\n'
                        '  "coverage_score": 0.8\n'
                        '}\n\n'
                        'coverage_score: 0.0 (no coverage) to 1.0 (complete coverage)'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Original query: "{original_query}"\n\n'
                        f"Search results found ({len(sources)} sources):\n{results_text}\n\n"
                        f"Full content summary:\n{content[:2000]}\n\n"
                        "Analyze whether these results adequately answer the query."
                    ),
                },
            ]
            
            payload = {
                "model": self.config.query_model,
                "messages": messages,
                "temperature": 0.2,  # Low temperature for consistent JSON
                "max_tokens": 800,
                "stream": False,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

            response = await asyncio.wait_for(
                self._run_sync_request(
                    lambda: requests.post(
                        f"{self.config.llm_base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=190,
                    )
                ),
                timeout=self.config.thinking_timeout
            )
            
            if response.status_code != 200:
                print("Result reasoning LLM error:", response.text)
                # Return default: no more search needed
                return {
                    "needs_more_search": False,
                    "reasoning": "Could not analyze results, assuming adequate coverage",
                    "follow_up_queries": [],
                    "coverage_score": 0.5
                }
            
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            
            # Parse JSON response
            try:
                # Extract JSON from response (may have markdown formatting)
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group()
                
                reasoning_result = json.loads(content)
                
                # Validate response structure
                return {
                    "needs_more_search": reasoning_result.get("needs_more_search", False),
                    "reasoning": reasoning_result.get("reasoning", "No reasoning provided"),
                    "follow_up_queries": reasoning_result.get("follow_up_queries", [])[:2],
                    "coverage_score": reasoning_result.get("coverage_score", 0.5)
                }
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse reasoning JSON: {e}")
                print(f"Raw content: {content}")
                # Return conservative default
                return {
                    "needs_more_search": False,
                    "reasoning": "Could not parse analysis, assuming adequate coverage",
                    "follow_up_queries": [],
                    "coverage_score": 0.5
                }
            
        except asyncio.TimeoutError:
            print("Result reasoning timed out")
            if progress_callback:
                progress_callback("Analysis timed out", 85)
            return {
                "needs_more_search": False,
                "reasoning": "Analysis timed out",
                "follow_up_queries": [],
                "coverage_score": 0.5
            }
        except Exception as e:
            print(f"Result reasoning exception: {e}")
            if progress_callback:
                progress_callback("Analysis error", 85)
            return {
                "needs_more_search": False,
                "reasoning": f"Analysis error: {str(e)}",
                "follow_up_queries": [],
                "coverage_score": 0.5
            }

    async def search_with_reasoning(
        self,
        query: str,
        max_results: int = 30,
        top_k: int = 22,
        progress_callback=None,
        wait_for_thinking: bool = True
    ) -> Dict:
        """
        Enhanced search with thinking model support and result reasoning.
        
        This method:
        1. Extracts search terms from paragraph queries
        2. Waits for thinking models to complete before searching
        3. Performs iterative search with reasoning over results
        4. Determines if additional searches are needed
        
        Args:
            query: The user's query (can be a paragraph)
            max_results: Maximum results per query
            top_k: Final number of top chunks to return
            progress_callback: Callback for progress updates
            wait_for_thinking: Whether to wait for thinking models
            
        Returns:
            Enhanced search results with reasoning metadata
        """
        try:
            # Helper to report progress with detailed data
            async def report_progress(status: str, progress: int, data: Dict = None):
                if progress_callback:
                    try:
                        result = progress_callback(status, progress, data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        print(f"Progress callback error: {e}")
            
            # Step 1: Wait for thinking to complete if enabled
            if wait_for_thinking and self.config.enable_search_term_extraction:
                await report_progress("Analyzing query and extracting search terms...", 2, {
                    "step": "query_analysis",
                    "status": "in_progress"
                })
                
                # Extract search terms from the query
                search_terms = await self._extract_search_terms_from_query(query)
                
                if len(search_terms) > 1:
                    await report_progress(f"Extracted {len(search_terms)} search terms", 4, {
                        "step": "query_analysis",
                        "status": "completed",
                        "search_terms": search_terms,
                        "original_query": query
                    })
                else:
                    await report_progress(f"Using search term: {search_terms[0]}", 4, {
                        "step": "query_analysis",
                        "status": "completed",
                        "search_terms": search_terms,
                        "original_query": query
                    })
            else:
                search_terms = [query]
                await report_progress("Using original query for search", 4, {
                    "step": "query_analysis",
                    "status": "skipped",
                    "search_terms": search_terms
                })
            
            # Step 2: Perform initial search with extracted terms
            all_sources = []
            all_chunks = []
            all_source_indices = []
            iteration_results = []
            search_steps = []  # Track all search steps for display
            
            for iteration in range(self.config.max_search_iterations):
                current_query = search_terms[iteration] if iteration < len(search_terms) else search_terms[-1]
                
                await report_progress(f"Search {iteration + 1}/{self.config.max_search_iterations}: \"{current_query[:50]}...\"", 
                                     10 + (iteration * 25), {
                    "step": "search",
                    "iteration": iteration + 1,
                    "query": current_query,
                    "status": "in_progress"
                })
                
                search_result = await self.search(
                    query=current_query,
                    max_results=max_results,
                    top_k=top_k // (iteration + 1),
                    progress_callback=progress_callback
                )
                
                # Collect results
                if "error" not in search_result:
                    sources_count = len(search_result.get("sources", []))
                    chunks_count = len(search_result.get("chunks", []))
                    
                    all_sources.extend(search_result.get("sources", []))
                    all_chunks.extend(search_result.get("chunks", []))
                    all_source_indices.extend(search_result.get("source_indices", []))
                    iteration_results.append(search_result)
                    
                    # Record this search step
                    search_steps.append({
                        "iteration": iteration + 1,
                        "query": current_query,
                        "sources_found": sources_count,
                        "chunks_created": chunks_count,
                        "status": "completed"
                    })
                    
                    await report_progress(f"Found {sources_count} sources, {chunks_count} chunks", 
                                         10 + (iteration * 25) + 15, {
                        "step": "search",
                        "iteration": iteration + 1,
                        "query": current_query,
                        "status": "completed",
                        "sources_found": sources_count,
                        "chunks_created": chunks_count,
                        "search_steps": search_steps
                    })
                
                # Step 3: Reason over results (skip on last iteration)
                if iteration < self.config.max_search_iterations - 1 and self.config.enable_result_reasoning:
                    await report_progress("Analyzing search results coverage...", 85 + (iteration * 5), {
                        "step": "reasoning",
                        "iteration": iteration + 1,
                        "status": "in_progress"
                    })
                    
                    reasoning = await self._reason_over_search_results(
                        original_query=query,
                        search_results=search_result,
                        progress_callback=progress_callback
                    )
                    
                    # Store reasoning for metadata
                    search_result["reasoning"] = reasoning
                    
                    await report_progress(f"Coverage score: {reasoning['coverage_score']:.1f}", 90 + (iteration * 5), {
                        "step": "reasoning",
                        "iteration": iteration + 1,
                        "status": "completed",
                        "coverage_score": reasoning["coverage_score"],
                        "needs_more_search": reasoning["needs_more_search"],
                        "reasoning": reasoning["reasoning"]
                    })
                    
                    # Check if more search is needed
                    if not reasoning["needs_more_search"] or reasoning["coverage_score"] >= 0.8:
                        await report_progress("Results adequately cover the query - stopping search", 92, {
                            "step": "reasoning",
                            "decision": "stop",
                            "reason": "adequate_coverage",
                            "coverage_score": reasoning["coverage_score"]
                        })
                        break
                    
                    # Get follow-up queries
                    if reasoning["follow_up_queries"]:
                        search_terms = reasoning["follow_up_queries"]
                        await report_progress(f"Need more info - follow-up: {search_terms}", 95, {
                            "step": "reasoning",
                            "decision": "continue",
                            "follow_up_queries": search_terms,
                            "reason": reasoning["reasoning"]
                        })
                    else:
                        break
                else:
                    break
            
            # Combine results
            if not all_sources:
                return {
                    "sources": [],
                    "content": "No search results found.",
                    "search_iterations": len(iteration_results),
                    "reasoning": "No results from any iteration",
                    "search_steps": search_steps
                }

            # Deduplicate sources by URL and remap indices
            seen_urls = {}
            unique_sources = []
            unique_chunks = []
            unique_source_indices = []  # New indices (0, 1, 2, ...)
            
            for source, chunk in zip(all_sources, all_chunks):
                url = source.get("url", "")
                if url not in seen_urls:
                    # New unique source
                    new_idx = len(unique_sources)
                    seen_urls[url] = new_idx
                    unique_sources.append(source)
                    unique_chunks.append(chunk)
                    unique_source_indices.append(new_idx)
                else:
                    # Duplicate source - still add chunk but map to existing source
                    existing_idx = seen_urls[url]
                    unique_chunks.append(chunk)
                    unique_source_indices.append(existing_idx)

            # Format final results
            await report_progress("Formatting final results with citations...", 95, {
                "step": "formatting",
                "total_sources": len(unique_sources),
                "total_chunks": len(unique_chunks)
            })

            output = self._format_results(unique_chunks, unique_source_indices, unique_sources, query)

            await report_progress("Search complete!", 100, {
                "step": "complete",
                "total_sources": len(unique_sources),
                "search_iterations": len(iteration_results),
                "search_steps": search_steps
            })

            return {
                "sources": unique_sources,
                "content": output,
                "chunks": unique_chunks,
                "source_indices": unique_source_indices,
                "search_iterations": len(iteration_results),
                "original_query": query,
                "search_terms_used": search_terms,
                "search_steps": search_steps
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in search_with_reasoning: {e}\n{error_details}")
            return {
                "sources": [],
                "content": f"Error during search: {str(e)}",
                "error": str(e)
            }

    async def _fetch_searxng_results(self, query: str) -> List[Tuple[str, str, str]]:
        """Fetch search results from SearXNG"""
        params = {"q": query, "format": "json", "categories": "general"}
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

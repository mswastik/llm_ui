import asyncio
import aiohttp
import os
import requests
import numpy as np
import contextvars
from typing import List, Dict, Optional, Any

# ContextVar to pass document_ids from stream handler to tool executor
current_document_ids: contextvars.ContextVar = contextvars.ContextVar('document_ids', default=None)

class SharedLLMUtils:
    """Shared utilities for embeddings and reranking to reduce code duplication."""

    @staticmethod
    def _get_base_url() -> str:
        """Get the current llama.cpp base URL from environment or settings."""
        try:
            # Try to import and use settings manager for the configured URL
            from backend.settings import settings_manager
            settings = settings_manager.get_settings()
            return settings.get('llama_cpp_base_url', 'http://localhost:8080').rstrip('/')
        except (ImportError, Exception):
            # Fallback to environment variable if settings not available
            return os.getenv("LLAMA_CPP_URL", "http://localhost:8080").rstrip('/')

    @staticmethod
    async def get_embedding(text: str, model: str = "Qwen3-4B-Embedding", max_retries: int = 3) -> np.ndarray:
        """Get embedding vector for text with retry logic."""
        base_url = SharedLLMUtils._get_base_url()
        embeddings_api = f"{base_url}/v1/embeddings"
        last_error = None
        for attempt in range(max_retries):
            try:
                payload = {"input": text, "model": model}
                # Using synchronous requests in a thread pool for simplicity as seen in original code, 
                # but could be converted to aiohttp.
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(embeddings_api, json=payload, timeout=60)
                )
                
                if response.status_code == 200:
                    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
                else:
                    error_msg = f"Status {response.status_code}: {response.text}"
                    print(f"[Embedding] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                    last_error = error_msg
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Embedding] Attempt {attempt + 1}/{max_retries} error: {error_msg}")
                last_error = error_msg
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        raise Exception(f"Embedding failed after {max_retries} attempts. Last error: {last_error}")

    @staticmethod
    async def rerank(query: str, documents: List[str], model: str = "Qwen3-4B-Embedding", max_retries: int = 3) -> List[int]:
        """Rerank documents and return indices in new order."""
        base_url = SharedLLMUtils._get_base_url()
        rerank_api = f"{base_url}/v1/rerank"
        last_error = None
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "query": query,
                    "documents": documents,
                }
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(rerank_api, json=payload, timeout=130)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "results" in result:
                        sorted_results = sorted(result["results"], key=lambda x: x.get("relevance_score", 0), reverse=True)
                        return [r["index"] for r in sorted_results]
                    elif "scores" in result:
                        return list(np.argsort(result["scores"])[::-1])
                    elif "data" in result:
                        sorted_data = sorted(result["data"], key=lambda x: x.get("score", 0), reverse=True)
                        return [d["index"] for d in sorted_data]
                else:
                    error_msg = f"Status {response.status_code}: {response.text}"
                    print(f"[Rerank] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                    last_error = error_msg
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Rerank] Attempt {attempt + 1}/{max_retries} error: {error_msg}")
                last_error = error_msg
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        print(f"[Rerank] Failed after {max_retries} attempts, falling back to original order")
        return list(range(len(documents)))



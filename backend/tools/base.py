"""
Shared utilities for embeddings and reranking.
"""
import asyncio
import os
import numpy as np
import contextvars
from typing import List

# ContextVar to pass document_ids from stream handler to tool executor
current_document_ids: contextvars.ContextVar = contextvars.ContextVar('document_ids', default=None)


def _get_base_url() -> str:
    """Get the current llama.cpp base URL."""
    try:
        from backend.settings import settings_manager
        settings = settings_manager.get_settings()
        return settings.get('llama_cpp_base_url', 'http://localhost:8080').rstrip('/')
    except (ImportError, Exception):
        return os.getenv("LLAMA_CPP_URL", "http://localhost:8080").rstrip('/')


async def get_embedding(text: str, model: str = "Qwen3-4B-Embedding", max_retries: int = 3) -> np.ndarray:
    """Get embedding vector for text with retry logic."""
    import aiohttp
    base_url = _get_base_url()
    url = f"{base_url}/v1/embeddings"
    last_error = None
    for attempt in range(max_retries):
        try:
            payload = {"input": text, "model": model}
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return np.array(data["data"][0]["embedding"], dtype=np.float32)
                    else:
                        error_msg = f"Status {response.status}"
                        print(f"[Embedding] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                        last_error = error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[Embedding] Attempt {attempt + 1}/{max_retries} error: {error_msg}")
            last_error = error_msg
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    raise Exception(f"Embedding failed after {max_retries} attempts. Last error: {last_error}")


async def rerank(query: str, documents: List[str], model: str = "Qwen3-4B-Embedding", max_retries: int = 3) -> List[int]:
    """Rerank documents and return indices in new order."""
    import aiohttp
    import json
    base_url = _get_base_url()
    url = f"{base_url}/v1/rerank"
    last_error = None
    for attempt in range(max_retries):
        try:
            payload = {"model": model, "query": query, "documents": documents}
            timeout = aiohttp.ClientTimeout(total=130)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "results" in result:
                            sorted_results = sorted(result["results"], key=lambda x: x.get("relevance_score", 0), reverse=True)
                            return [r["index"] for r in sorted_results]
                        elif "scores" in result:
                            return list(np.argsort(result["scores"])[::-1])
                        elif "data" in result:
                            sorted_data = sorted(result["data"], key=lambda x: x.get("score", 0), reverse=True)
                            return [d["index"] for d in sorted_data]
                    else:
                        error_msg = f"Status {response.status}"
                        print(f"[Rerank] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                        last_error = error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[Rerank] Attempt {attempt + 1}/{max_retries} error: {error_msg}")
            last_error = error_msg
        if attempt < max_retries - 1:
            await asyncio.sleep(2)
    print(f"[Rerank] Failed after {max_retries} attempts, falling back to original order")
    return list(range(len(documents)))

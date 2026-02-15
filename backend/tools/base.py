import asyncio
import aiohttp
import requests
import numpy as np
from typing import List, Dict, Optional, Any
from config import LLAMA_CPP_BASE_URL

class SharedLLMUtils:
    """Shared utilities for embeddings and reranking to reduce code duplication."""
    
    @staticmethod
    async def get_embedding(text: str, model: str = "qwen3-embedding", max_retries: int = 3) -> np.ndarray:
        """Get embedding vector for text with retry logic."""
        embeddings_api = f"{LLAMA_CPP_BASE_URL}/v1/embeddings"
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
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        raise Exception(f"Embedding failed after {max_retries} attempts")

    @staticmethod
    async def rerank(query: str, documents: List[str], model: str = "qwen3-reranker", max_retries: int = 3) -> List[int]:
        """Rerank documents and return indices in new order."""
        rerank_api = f"{LLAMA_CPP_BASE_URL}/v1/rerank"
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
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    return list(range(len(documents)))
        return list(range(len(documents)))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

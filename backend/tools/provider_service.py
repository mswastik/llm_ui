"""
LLM provider helpers: auto-fetch models from an OpenAI-compatible endpoint.

Used when a provider is added or refreshed so the model selector always
reflects what the provider actually serves.
"""
from typing import Dict, List, Optional


def normalize_base_url(base_url: str) -> str:
    """Strip a trailing /v1 or /v3 so callers can append /v1/models safely."""
    return (base_url or "").rstrip("/").replace("/v1", "").replace("/v3", "")


async def fetch_models(base_url: str, api_key: Optional[str] = None,
                       timeout: float = 20) -> List[Dict]:
    """GET {base}/v1/models and return [{id, name, owned_by}]."""
    import aiohttp
    base = normalize_base_url(base_url)
    url = f"{base}/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise ValueError(f"Provider returned HTTP {response.status} for {url}")
            data = await response.json()
    models = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        if not mid:
            continue
        models.append({
            "id": mid,
            "name": m.get("name") or mid,
            "owned_by": m.get("owned_by") or "unknown",
        })
    return models


async def ping_provider(base_url: str, api_key: Optional[str] = None) -> Dict:
    """Check provider reachability; returns {ok, models?, error?}."""
    try:
        models = await fetch_models(base_url, api_key, timeout=10)
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}

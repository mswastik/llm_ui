"""
LLM provider helpers: auto-fetch models from an OpenAI-compatible endpoint.

Used when a provider is added or refreshed so the model selector always
reflects what the provider actually serves.
"""
from typing import Dict, List, Optional


def normalize_base_url(base_url: str) -> str:
    """Strip a trailing /v1 or /v3 so callers can append /v1/models safely."""
    return (base_url or "").rstrip("/").replace("/v1", "").replace("/v3", "")


def _ctx_from_args(raw: Dict) -> Optional[int]:
    """Context window a llama.cpp model is loaded with.

    The router's /v1/models entries embed the launch args; --ctx-size is the
    per-model window. Returns None for providers that don't expose it.
    """
    status = raw.get("status") or {}
    args = status.get("args")
    if not isinstance(args, list):
        return None
    for flag in ("--ctx-size", "--context-size", "-c"):
        for i, a in enumerate(args):
            if isinstance(a, str) and a == flag and i + 1 < len(args):
                try:
                    v = int(str(args[i + 1]).split(",")[0])  # -c accepts a per-slot list
                    if v > 0:
                        return v
                except ValueError:
                    pass
    return None


async def fetch_models(base_url: str, api_key: Optional[str] = None,
                       timeout: float = 20) -> List[Dict]:
    """GET {base}/v1/models and return [{id, name, owned_by, context_window?}]."""
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
        entry = {
            "id": mid,
            "name": m.get("name") or mid,
            "owned_by": m.get("owned_by") or "unknown",
        }
        # OpenRouter-style providers report the window per model; llama.cpp reports
        # it in its launch args. A miss means "unknown" — the UI then shows a bare
        # token count rather than guessing a utilisation %.
        ctx = m.get("context_length") or m.get("context_window")
        if not isinstance(ctx, (int, float)) or ctx <= 0:
            ctx = _ctx_from_args(m)
        if isinstance(ctx, (int, float)) and ctx > 0:
            entry["context_window"] = int(ctx)
        models.append(entry)
    return models



async def ping_provider(base_url: str, api_key: Optional[str] = None) -> Dict:
    """Check provider reachability; returns {ok, models?, error?}."""
    try:
        models = await fetch_models(base_url, api_key, timeout=10)
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}

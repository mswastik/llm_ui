"""
MCP server registry browser + installer.

Lists installable MCP servers from the Smithery registry
(registry.smithery.ai — public API, no auth) and installs a chosen server
into the app's MCP manager (same path as the Settings form / add_mcp_server
tool).

- Listing: GET https://registry.smithery.ai/servers?q=<query>&perPage=N
  (blank query = sorted by use count).
- Detail:  GET https://registry.smithery.ai/servers/{namespace}/{slug} →
  `connections` array with transport type, deployment URL (http) or
  command/args/env (stdio).
- Install: picks the first usable connection and adds the server:
  http → streamable-http with url; stdio → command/args/env.
- Enrichment: stars / GitHub link from the GitHub API (reuses the
  skills.sh registry cache; rate-limit safe).
"""
import asyncio
import re
from typing import Dict, List, Optional
from urllib.parse import quote

import aiohttp

from tools.skill_registry import _cached_repo_info, _store_repo_info  # shared GitHub cache

SEARCH_URL = "https://registry.smithery.ai/servers"
DETAIL_URL = "https://registry.smithery.ai/servers/{ns}/{slug}"
REQUEST_TIMEOUT = 30

_QUALIFIED_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


async def _get_json(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status} for {url}")
        return await response.json()


async def search_mcp_registry(query: str = "", limit: int = 24) -> List[Dict]:
    """Search Smithery. Blank query = most-used servers."""
    limit = max(1, min(int(limit or 24), 100))
    q = (query or "").strip()
    params = {"perPage": str(limit)}
    if len(q) >= 2:
        params["q"] = q
    url = f"{SEARCH_URL}?{'&'.join(f'{k}={quote(v)}' for k, v in params.items())}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
        data = await _get_json(session, url)
    servers = []
    for s in data.get("servers", []):
        qn = s.get("qualifiedName") or ""
        servers.append({
            "qualified_name": qn,
            "name": s.get("displayName") or qn.split("/")[-1] or qn,
            "description": s.get("description") or "",
            "use_count": s.get("useCount") or 0,
            "verified": bool(s.get("verified")),
            "homepage": s.get("homepage"),
            "icon_url": s.get("iconUrl"),
        })
    if len(q) < 2:
        servers.sort(key=lambda x: x["use_count"], reverse=True)
    return servers


async def enrich_mcp_servers(servers: List[Dict], max_fetches: int = 15) -> List[Dict]:
    """Add stars/repo_url from GitHub when namespace/slug is a GitHub repo."""
    unique = {}
    for s in servers:
        qn = s.get("qualified_name") or ""
        if _QUALIFIED_RE.match(qn) and qn not in unique:
            unique[qn] = None

    pending = [qn for qn in unique if _cached_repo_info(*qn.split("/")) is None]
    if pending:
        rate_limited = False
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
            for qn in pending[:max_fetches]:
                if rate_limited:
                    break
                owner, repo = qn.split("/", 1)
                try:
                    async with session.get(f"https://api.github.com/repos/{owner}/{repo}") as response:
                        if response.status == 200:
                            data = await response.json()
                            info = {
                                "stars": data.get("stargazers_count"),
                                "gh_description": (data.get("description") or "").strip() or None,
                            }
                        elif response.status in (403, 429):
                            rate_limited = True
                            print("[MCP REGISTRY] GitHub rate limit hit — enrichment deferred")
                            continue
                        else:
                            info = {"stars": None, "gh_description": None}
                        _store_repo_info(owner, repo, info)
                        unique[qn] = info
                except Exception:
                    _store_repo_info(owner, repo, {"stars": None, "gh_description": None})
                    unique[qn] = {"stars": None, "gh_description": None}
    else:
        for qn in unique:
            owner, repo = qn.split("/", 1)
            info = _cached_repo_info(owner, repo)
            if info is not None:
                unique[qn] = info

    for s in servers:
        qn = s.get("qualified_name") or ""
        if not _QUALIFIED_RE.match(qn):
            s["stars"] = None
            s["repo_url"] = None
            continue
        owner, repo = qn.split("/", 1)
        info = unique.get(qn) or {}
        s["stars"] = info.get("stars")
        s["repo_url"] = f"https://github.com/{owner}/{repo}"
        if not s.get("description") and info.get("gh_description"):
            s["description"] = info["gh_description"]
    return servers


async def get_mcp_detail(qualified_name: str) -> Dict:
    ns, slug = qualified_name.split("/", 1)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
        return await _get_json(session, DETAIL_URL.format(ns=ns, slug=slug))


async def install_mcp_from_registry(qualified_name: str, mcp_manager) -> Dict:
    """Add a registry MCP server to the app. Returns the added config."""
    if mcp_manager is None:
        raise ValueError("MCP manager unavailable")
    if not _QUALIFIED_RE.match(qualified_name):
        raise ValueError(f"Invalid server id: {qualified_name}")
    detail = await get_mcp_detail(qualified_name)
    if "error" in detail:
        raise ValueError(detail["error"])
    name = (detail.get("displayName") or qualified_name.split("/")[-1]).strip()
    connections = detail.get("connections") or []
    if not connections:
        raise ValueError("No connection config found for this server")

    chosen = None
    for conn in connections:
        ctype = conn.get("type")
        if ctype == "stdio" and conn.get("command"):
            chosen = conn
            break
        if ctype in ("http", "streamable-http") and (conn.get("deploymentUrl") or conn.get("url")):
            chosen = conn
            break
    if chosen is None:
        raise ValueError("No usable connection config (stdio command or http URL) found")

    ctype = chosen.get("type")
    if ctype == "stdio":
        transport_type = "stdio"
        command = chosen.get("command")
        args = chosen.get("args") or []
        env = chosen.get("env") or {}
        url = None
        headers = chosen.get("headers") or {}
    else:
        transport_type = "streamable-http"
        url = chosen.get("deploymentUrl") or chosen.get("url")
        command = None
        args = []
        env = {}
        headers = chosen.get("headers") or {}

    success, error = await mcp_manager.add_server(
        name, command, args, env, transport_type, url, timeout=60.0, headers=headers)
    return {
        "name": name,
        "qualified_name": qualified_name,
        "transport_type": transport_type,
        "url": url,
        "command": command,
        "connected": success,
        "error": error,
    }

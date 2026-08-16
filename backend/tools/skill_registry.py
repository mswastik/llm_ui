"""
Skill registry browser + installer (agent platform, Phase 4.5).

Lists installable skills from the skills.sh registry (Vercel's open agent
skills directory — the same catalog Hermes ships with) and installs a
chosen skill into the local skills/ directory.

- Listing: GET https://www.skills.sh/api/search?q=... (public, no auth).
- Install: a skills.sh entry resolves to a GitHub repo
  (owner/repo[/path]); we locate SKILL.md at the common layouts
  (skills/<name>/, <path>/, <name>/, repo root) and copy the skill's files
  into skills/<name>/ so the existing loader/index/load_skill pipeline
  picks it up unchanged.

Security: names are slugified; GitHub fetches are pinned to
raw.githubusercontent.com + api.github.com; file count and total size are
capped. Installed skill content is LLM instructions — any dangerous commands
it suggests still go through run_command's blocklist/allowlist/approval.
"""
import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp

from settings import settings_manager

SEARCH_URL = "https://www.skills.sh/api/search"
GITHUB_RAW = "https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"

MAX_FILES = 60
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_FILE_BYTES = 1_500_000
REQUEST_TIMEOUT = 30


def _skills_dir() -> str:
    return settings_manager.get_settings().get("skills_dir") or "./skills"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")
    return slug or "skill"


async def _get_json(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status} for {url}")
        return await response.json()


async def search_registry(query: str, limit: int = 25) -> List[Dict]:
    """Search the skills.sh registry (public API, no auth)."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    limit = max(1, min(int(limit or 25), 100))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
        data = await _get_json(session, f"{SEARCH_URL}?q={quote(q)}&limit={limit}")
    out = []
    for sk in data.get("skills", []):
        out.append({
            "id": sk.get("id"),
            "name": sk.get("skillId") or sk.get("name"),
            "display": sk.get("name") or sk.get("skillId"),
            "installs": sk.get("installs", 0),
            "source": sk.get("source"),
        })
    return out


# Broad queries merged to approximate a "most popular" listing. The public
# skills.sh API has no unauthenticated catalog endpoint, so we fan out a few
# generic queries and merge by id, sorted by install count.
_POPULAR_QUERIES = ["ai", "search", "data", "writing", "automation", "web"]


async def popular_registry(limit: int = 25) -> List[Dict]:
    """Return the most-installed skills across several broad queries."""
    limit = max(1, min(int(limit or 25), 100))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
        async def fetch_one(q: str) -> List[Dict]:
            try:
                data = await _get_json(session, f"{SEARCH_URL}?q={quote(q)}&limit={limit}")
                return data.get("skills", [])
            except Exception:
                return []

        results = await asyncio.gather(*[fetch_one(q) for q in _POPULAR_QUERIES])

    merged: Dict[str, Dict] = {}
    for skills in results:
        for sk in skills:
            sid = sk.get("id")
            if not sid or sid in merged:
                continue
            merged[sid] = {
                "id": sid,
                "name": sk.get("skillId") or sk.get("name"),
                "display": sk.get("name") or sk.get("skillId"),
                "installs": sk.get("installs", 0),
                "source": sk.get("source"),
            }
    ranked = sorted(merged.values(), key=lambda x: x["installs"], reverse=True)
    return ranked[:limit]


# ─── GitHub repo enrichment (stars / description / link) ─────────────────
# Unauthenticated GitHub API is rate-limited (60 req/hr/IP), so repo info is
# cached in-process for an hour and failures degrade to nulls.

_REPO_CACHE: Dict[str, Dict] = {}
_REPO_CACHE_TTL = 3600
_GITHUB_REPO = "https://api.github.com/repos/{owner}/{repo}"
_GITHUB_SOURCE_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _cached_repo_info(owner: str, repo: str) -> Optional[Dict]:
    import time
    key = f"{owner}/{repo}".lower()
    entry = _REPO_CACHE.get(key)
    if entry and time.time() - entry.get("ts", 0) < _REPO_CACHE_TTL:
        return entry.get("data")
    return None


def _store_repo_info(owner: str, repo: str, data: Optional[Dict]):
    import time
    key = f"{owner}/{repo}".lower()
    _REPO_CACHE[key] = {"ts": time.time(), "data": data}
    if len(_REPO_CACHE) > 500:
        _REPO_CACHE.clear()


async def enrich_registry(skills: List[Dict], max_fetches: int = 15) -> List[Dict]:
    """Add stars/description/repo_url per skill from GitHub (cached).

    Sources that are not GitHub owner/repo pairs are left with nulls.
    Rate-limited or failed fetches degrade gracefully (stars=None); on a
    GitHub rate limit the remaining sources are skipped (not cached) so the
    next request can retry them.
    """
    if not skills:
        return skills

    # Map each unique GitHub source to its enrichment data.
    unique_sources = {}
    for sk in skills:
        src = (sk.get("source") or "").strip()
        if _GITHUB_SOURCE_RE.match(src) and src not in unique_sources:
            unique_sources[src] = None

    pending = [src for src in unique_sources if _cached_repo_info(*src.split("/")) is None]
    if pending:
        rate_limited = False
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
            for src in pending[:max_fetches]:
                if rate_limited:
                    break
                owner, repo = src.split("/", 1)
                try:
                    async with session.get(_GITHUB_REPO.format(owner=owner, repo=repo)) as response:
                        if response.status == 200:
                            data = await response.json()
                            info = {
                                "stars": data.get("stargazers_count"),
                                "description": (data.get("description") or "").strip() or None,
                            }
                        elif response.status in (403, 429):
                            # Rate limited — skip the rest, retry on next request.
                            rate_limited = True
                            print("[REGISTRY] GitHub rate limit hit — enrichment deferred")
                            continue
                        else:
                            info = {"stars": None, "description": None}
                        _store_repo_info(owner, repo, info)
                        unique_sources[src] = info
                except Exception:
                    _store_repo_info(owner, repo, {"stars": None, "description": None})
                    unique_sources[src] = {"stars": None, "description": None}
    else:
        for src in unique_sources:
            owner, repo = src.split("/", 1)
            info = _cached_repo_info(owner, repo)
            if info is not None:
                unique_sources[src] = info

    for sk in skills:
        src = (sk.get("source") or "").strip()
        if not _GITHUB_SOURCE_RE.match(src):
            sk["stars"] = None
            sk["description"] = None
            sk["repo_url"] = None
            continue
        owner, repo = src.split("/", 1)
        info = unique_sources.get(src) or {}
        sk["stars"] = info.get("stars")
        sk["description"] = info.get("description")
        sk["repo_url"] = f"https://github.com/{owner}/{repo}"

    return skills


def _parse_id(skill_id: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """'owner/repo' or 'owner/repo/path' → (owner, repo, path)."""
    parts = (skill_id or "").strip().split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", owner) or not re.fullmatch(r"[A-Za-z0-9._-]+", repo):
        return None
    path = "/".join(parts[2:]) or None
    return owner, repo, path


MAX_TAR_BYTES = 100 * 1024 * 1024  # tarball download cap


async def _download_tarball(owner: str, repo: str) -> bytes:
    """Download the repo tarball from codeload.github.com.

    Codeload has no API rate limits and is case-insensitive, so this works
    even when the GitHub contents API is rate-limited or the registry's repo
    casing differs from the real one.
    """
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/HEAD"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Repository {owner}/{repo} is not reachable (HTTP {response.status})")
            data = await response.read()  # full body; read(n) only returns buffered bytes
    if len(data) > MAX_TAR_BYTES:
        raise ValueError(f"Repository {owner}/{repo} is too large to install from")
    return data


def _tar_entries(data: bytes):
    """Yield (repo_relative_path, member) for a GitHub tarball."""
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            rel = parts[1] if len(parts) > 1 else ""
            if not rel:
                continue
            yield rel, member


def _score_skill_dir(dir_path: str, name: str) -> int:
    """Higher = better match for the requested skill name."""
    base = os.path.basename(dir_path.rstrip("/")) if dir_path else ""
    if base == name:
        return 100
    if base and (name in base or base in name):
        return 90
    if dir_path.startswith("skills/"):
        return 60
    return 20


def _pick_skill_dir(data: bytes, name: str) -> Optional[str]:
    """Find the best directory containing a SKILL.md in the tarball."""
    best = None
    best_score = -1
    for rel, member in _tar_entries(data):
        if not member.isfile():
            continue
        if not rel.endswith("SKILL.md"):
            continue
        dir_path = os.path.dirname(rel)
        score = _score_skill_dir(dir_path, name)
        # Prefer shallower paths when scores tie.
        if score > best_score or (score == best_score and dir_path.count("/") < (best or "").count("/")):
            best = dir_path
            best_score = score
    return best


def _extract_skill_dir(data: bytes, skill_dir: str, target_dir: str) -> List[str]:
    """Extract files under skill_dir into target_dir (sanitized + capped)."""
    import io
    import tarfile
    wrote: List[str] = []
    total = 0
    os.makedirs(target_dir, exist_ok=True)
    target_abs = os.path.abspath(target_dir)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            rel = parts[1] if len(parts) > 1 else ""
            if not rel.startswith(skill_dir + "/") and rel != skill_dir:
                continue
            rel_under = rel[len(skill_dir):].lstrip("/")
            if not rel_under:
                continue
            if rel_under.startswith(".") or "/." in rel_under:
                continue
            if member.isdir():
                continue
            if not member.isfile():
                continue  # skip symlinks/devices
            if len(wrote) >= MAX_FILES or total >= MAX_TOTAL_BYTES or member.size > MAX_FILE_BYTES:
                continue
            dest = os.path.join(target_dir, rel_under)
            if not os.path.abspath(dest).startswith(target_abs + os.sep) and os.path.abspath(dest) != target_abs:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            f = tar.extractfile(member)
            if f is None:
                continue
            data_part = f.read(MAX_FILE_BYTES + 1)
            if len(data_part) > MAX_FILE_BYTES:
                continue
            with open(dest, "wb") as out:
                out.write(data_part)
            wrote.append(rel_under)
            total += len(data_part)
    return wrote


async def install_registry_skill(skill_id: str) -> Dict:
    """Install a registry skill into skills/<name>/.

    Downloads the repo tarball (codeload — no API rate limits), locates the
    SKILL.md closest to the requested skill name, and extracts that
    directory verbatim. Raises ValueError with a user-readable message.
    """
    parsed = _parse_id(skill_id)
    if not parsed:
        raise ValueError(f"Invalid skill id: {skill_id}")
    owner, repo, path = parsed
    name = _slugify(os.path.basename(path) if path else repo)

    target_dir = os.path.join(_skills_dir(), name)
    if os.path.isdir(target_dir):
        raise ValueError(f"Skill '{name}' is already installed")

    try:
        data = await _download_tarball(owner, repo)
    except ValueError as e:
        raise
    except Exception as e:
        raise ValueError(f"Could not download {owner}/{repo}: {str(e)[:200]}")

    skill_dir = _pick_skill_dir(data, name)
    if skill_dir is None:
        raise ValueError(
            f"No SKILL.md found in {owner}/{repo} matching '{name}' — "
            "the repository may not contain agent skills."
        )

    wrote = _extract_skill_dir(data, skill_dir, target_dir)
    if not wrote:
        raise ValueError(f"Skill directory '{skill_dir or '.'}' is empty or unreachable")

    # Parse description from the SKILL.md frontmatter for the index.
    description = ""
    md_path = os.path.join(target_dir, "SKILL.md")
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                head = f.read(2000)
            m = re.search(r"^description:\s*(.+)$", head, re.MULTILINE)
            if m:
                description = m.group(1).strip()
        except Exception:
            pass

    return {
        "name": name,
        "description": description,
        "source": f"{owner}/{repo}",
        "files": wrote,
    }

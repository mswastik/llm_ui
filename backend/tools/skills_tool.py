"""
Agent skills engine (Phase 3).

A skill is a directory under SKILLS_DIR containing a SKILL.md file:

    skills/<name>/SKILL.md        # frontmatter + instructions
    skills/<name>/scripts/...     # optional helper files
    skills/<name>/assets/...      # optional assets

SKILL.md format (Claude/Hermes-compatible):

    ---
    name: <slug>
    description: one-line description for the skill index
    triggers: optional comma-separated trigger words
    ---
    <full instructions body>

Discovery scans SKILLS_DIR at stream start; a compact index (name +
description) is appended to the system prompt. The LLM calls load_skill to
pull the full instructions into context only when the skill applies.

Pending self-improvement drafts live in SKILLS_DIR/_drafts/<name>/ and are
excluded from the index until accepted.
"""
import os
import re
import shutil
from typing import Dict, List, Optional

from settings import settings_manager

MAX_SKILL_CONTENT_CHARS = 60_000

LOAD_SKILL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Load a skill's full instructions from the skills directory. The "
            "system prompt lists available skills; call this when a listed "
            "skill is relevant to the current task. Returns the SKILL.md "
            "content and a manifest of any helper files in the skill."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name (as listed in the available-skills index)"}
            },
            "required": ["name"]
        }
    }
}

CREATE_SKILL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "create_skill",
        "description": (
            "Create a reusable skill: a named set of instructions the agent "
            "can load on demand later. Use when the user asks to save a "
            "procedure, workflow, or recipe as a skill ('create a skill for "
            "X', 'save this as a skill'). The skill becomes available "
            "immediately in the skills index."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short slug name, e.g. 'news-fetch'"},
                "description": {"type": "string", "description": "One-line description shown in the skills index"},
                "instructions": {"type": "string", "description": "Step-by-step instructions the agent follows when the skill is loaded"}
            },
            "required": ["name", "description", "instructions"]
        }
    }
}


UPDATE_SKILL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "update_skill",
        "description": (
            "Update an existing skill's description or instructions. "
            "Use when the user asks to modify, improve, or fix a skill that "
            "already exists ('update skill X', 'improve skill Y'). The skill "
            "is overwritten in place and becomes available immediately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name to update (as listed in the available-skills index)"},
                "description": {"type": "string", "description": "One-line description shown in the skills index"},
                "instructions": {"type": "string", "description": "Step-by-step instructions the agent follows when the skill is loaded"}
            },
            "required": ["name", "description", "instructions"]
        }
    }
}

DELETE_SKILL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "delete_skill",
        "description": (
            "Delete a skill entirely. Use when the user asks to remove, delete, "
            "or unpublish a skill ('delete skill X', 'remove skill Y'). This "
            "permanently removes the skill directory and it will no longer appear "
            "in the skills index."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name to delete (as listed in the available-skills index)"}
            },
            "required": ["name"]
        }
    }
}

SKILL_TOOL_DEFINITIONS = [LOAD_SKILL_DEFINITION, CREATE_SKILL_DEFINITION, UPDATE_SKILL_DEFINITION, DELETE_SKILL_DEFINITION]


def _skills_dir() -> str:
    return settings_manager.get_settings().get("skills_dir") or "./skills"


def _drafts_dir() -> str:
    return os.path.join(_skills_dir(), "_drafts")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")
    return slug or "skill"


def _safe_join(base: str, *parts: str) -> str:
    path = os.path.realpath(os.path.join(base, *parts))
    base_real = os.path.realpath(base)
    if not (path == base_real or path.startswith(base_real + os.sep)):
        raise ValueError("Invalid skill path")
    return path


def _parse_frontmatter(content: str) -> tuple:
    """Parse `---` frontmatter; returns (meta: dict, body: str).

    Handles YAML block scalars (`>`, `>-`, `|`, `|-` ...) for the multi-line
    `description:` style used by marketplace skills — a naive line-wise parse
    stored the literal '>' as the description and silently dropped the real
    text from the skill index.
    """
    meta: Dict[str, str] = {}
    body = content
    stripped = content.lstrip("\ufeff")
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            fm = stripped[3:end]
            body = stripped[end + 4:].lstrip("\n")
            lines = fm.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if ":" in line and not line[:1].isspace():
                    key, _, val = line.partition(":")
                    key, val = key.strip().lower(), val.strip()
                    if re.match(r"^[|>][-+]?[0-9]*$", val):
                        parts = []
                        i += 1
                        while i < len(lines) and (lines[i][:1] in (" ", "\t") or not lines[i].strip()):
                            parts.append(lines[i].strip())
                            i += 1
                        val = " ".join(p for p in parts if p).strip()
                        meta[key] = val
                        continue
                    meta[key] = val
                i += 1
    return meta, body


def _read_skill_dir(skill_path: str) -> Optional[Dict]:
    md_path = os.path.join(skill_path, "SKILL.md")
    if not os.path.isfile(md_path):
        return None
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[SKILLS] read failed {skill_path}: {e}")
        return None
    meta, body = _parse_frontmatter(content)
    manifest = []
    for root, dirs, files in os.walk(skill_path):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for fn in files:
            if fn == "SKILL.md":
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, skill_path)
            manifest.append({"path": rel, "size": os.path.getsize(full)})
    return {
        "name": meta.get("name") or os.path.basename(skill_path),
        "description": meta.get("description", ""),
        "triggers": meta.get("triggers", ""),
        "body": body,
        "manifest": manifest,
        "path": os.path.relpath(skill_path, _skills_dir()),
    }


def list_skills(include_drafts: bool = False) -> List[Dict]:
    skills = []
    root = _skills_dir()
    os.makedirs(root, exist_ok=True)
    for entry in sorted(os.listdir(root)):
        if entry.startswith(".") or entry == "_drafts":
            continue
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        skill = _read_skill_dir(path)
        if skill:
            skills.append(skill)
    if include_drafts and os.path.isdir(_drafts_dir()):
        for entry in sorted(os.listdir(_drafts_dir())):
            path = os.path.join(_drafts_dir(), entry)
            if not os.path.isdir(path):
                continue
            skill = _read_skill_dir(path)
            if skill:
                skill["draft"] = True
                skills.append(skill)
    return skills


def skill_index() -> str:
    """Compact one-line-per-skill index for the system prompt."""
    lines = []
    for s in list_skills():
        desc = s.get("description") or ""
        lines.append(f"- {s['name']}: {desc}")
    if not lines:
        return ""
    return "\n".join(lines)


def get_skill(name: str) -> Optional[Dict]:
    slug = _slugify(name)
    path = _safe_join(_skills_dir(), slug)
    if not os.path.isdir(path):
        return None
    return _read_skill_dir(path)


def get_draft(name: str) -> Optional[Dict]:
    slug = _slugify(name)
    path = _safe_join(_drafts_dir(), slug)
    if not os.path.isdir(path):
        return None
    skill = _read_skill_dir(path)
    if skill:
        skill["draft"] = True
    return skill


def write_skill(name: str, description: str, instructions: str,
                draft: bool = False) -> Dict:
    """Create or overwrite a skill (or a draft)."""
    slug = _slugify(name)
    base = _drafts_dir() if draft else _skills_dir()
    os.makedirs(base, exist_ok=True)
    path = _safe_join(base, slug)
    os.makedirs(path, exist_ok=True)
    content = (
        f"---\nname: {slug}\ndescription: {description}\n---\n\n"
        f"{instructions.strip()}\n"
    )
    with open(os.path.join(path, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(content)
    skill = _read_skill_dir(path)
    if draft:
        skill["draft"] = True
    return skill


def delete_skill(name: str, draft: bool = False) -> bool:
    slug = _slugify(name)
    base = _drafts_dir() if draft else _skills_dir()
    path = _safe_join(base, slug)
    if not os.path.isdir(path):
        return False
    shutil.rmtree(path)
    return True


def accept_draft(name: str) -> Optional[Dict]:
    """Move a draft into the live skills directory."""
    slug = _slugify(name)
    draft_path = _safe_join(_drafts_dir(), slug)
    if not os.path.isdir(draft_path):
        return None
    target = _safe_join(_skills_dir(), slug)
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(_skills_dir(), exist_ok=True)
    shutil.move(draft_path, target)
    return _read_skill_dir(target)

"""
App-administration tools: configure llm-ui through the agent itself.

The LLM can create/delete agents, add/list/remove MCP servers, add/list
LLM providers (models auto-fetched), and search/install skills from the
registry — all through chat, with the same streaming progress events as
the other tools.

Trust model: same as the rest of the app — these are configuration actions
the user explicitly asks for in the conversation. MCP server commands spawn
processes (e.g. npx); run_command's approval gates do not cover them, so
this toolset is best limited to agents you trust via enabled_tools.
"""
from typing import Any, AsyncGenerator, Dict, List, Optional

from database.models import get_db
from database.crud import create_agent, get_all_agents

# ─── Tool definitions (OpenAI function schema) ───────────────────────────

def _def(name: str, description: str, properties: dict, required: list = None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required or []},
        },
    }

ADMIN_TOOL_DEFINITIONS = [
    _def("list_agents", "List all configured AI agents (id, name, model, provider).",
         {}),
    _def("create_agent",
         "Create a new AI agent (persona + capability configuration). "
         "Use when the user asks to create/set up an agent.",
         {"name": {"type": "string", "description": "Unique agent name"},
          "description": {"type": "string", "description": "Short description"},
          "system_prompt": {"type": "string", "description": "Agent persona/instructions"},
          "model": {"type": "string", "description": "Model id from a configured provider"},
          "provider_id": {"type": "string", "description": "Provider id (optional; default provider used if omitted)"},
          "temperature": {"type": "number"},
          "top_k": {"type": "integer"},
          "max_tokens": {"type": "integer"},
          "enable_rag": {"type": "boolean", "description": "Enable knowledge base retrieval"}},
         ["name"]),
    _def("delete_agent", "Delete an AI agent by id.", {"id": {"type": "string"}}, ["id"]),
    _def("list_mcp_servers",
         "List configured MCP servers with connection status and tool counts.",
         {}),
    _def("add_mcp_server",
         "Add and connect an MCP server. stdio transport needs 'command' (e.g. "
         "'npx') + args; sse/streamable-http need a 'url'. Models/tools are "
         "auto-discovered. The server command spawns a process — only add servers "
         "the user explicitly requested.",
         {"name": {"type": "string"},
          "transport_type": {"type": "string", "enum": ["stdio", "sse", "streamable-http"]},
          "command": {"type": "string", "description": "Binary for stdio transport, e.g. 'npx'"},
          "args": {"type": "array", "items": {"type": "string"}, "description": "Args, e.g. ['-y', '@some/mcp-server']"},
          "env": {"type": "object", "description": "Environment variables (optional)"},
          "url": {"type": "string", "description": "Endpoint URL for sse/streamable-http"},
          "timeout": {"type": "number", "description": "Connection timeout seconds"}},
         ["name"]),
    _def("remove_mcp_server", "Disconnect and remove an MCP server by name.",
         {"name": {"type": "string"}}, ["name"]),
    _def("list_providers",
         "List configured LLM providers (name, base_url, model count, default flag).",
         {}),
    _def("add_provider",
         "Add an LLM provider (OpenAI-compatible endpoint). Models are fetched "
         "automatically from its /v1/models endpoint.",
         {"name": {"type": "string"},
          "base_url": {"type": "string", "description": "e.g. https://openrouter.ai/api (no /v1 suffix)"},
          "api_key": {"type": "string", "description": "Optional API key"}},
         ["name", "base_url"]),
    _def("search_skills",
         "Search the skills.sh registry for installable skills. Use before "
         "install_skill to find the skill id.",
         {"query": {"type": "string", "description": "Search term (2+ chars)"},
          "limit": {"type": "integer", "description": "Max results (default 15)"}}),
    _def("install_skill",
         "Install a skill from the registry by its id (e.g. "
         "'owner/repo/skill-name' from search_skills). Installs into skills/ "
         "and it becomes available to all agents.",
         {"id": {"type": "string", "description": "Registry skill id"}}, ["id"]),
]


class AdminTool:
    def __init__(self, mcp_manager=None):
        self.mcp_manager = mcp_manager

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncGenerator[Dict, None]:
        try:
            if tool_name == "list_agents":
                yield await self._list_agents()
            elif tool_name == "create_agent":
                yield await self._create_agent(arguments)
            elif tool_name == "delete_agent":
                yield await self._delete_agent(arguments)
            elif tool_name == "list_mcp_servers":
                yield await self._list_mcp_servers()
            elif tool_name == "add_mcp_server":
                yield await self._add_mcp_server(arguments)
            elif tool_name == "remove_mcp_server":
                yield await self._remove_mcp_server(arguments)
            elif tool_name == "list_providers":
                yield await self._list_providers()
            elif tool_name == "add_provider":
                yield await self._add_provider(arguments)
            elif tool_name == "search_skills":
                yield await self._search_skills(arguments)
            elif tool_name == "install_skill":
                yield await self._install_skill(arguments)
            else:
                yield {"type": "tool_error", "tool": tool_name, "error": f"Unknown admin tool: {tool_name}"}
        except Exception as e:
            print(f"[ADMIN] {tool_name} failed: {e}")
            yield {"type": "tool_error", "tool": tool_name, "error": str(e)[:400]}

    # ── Agents ───────────────────────────────────────────

    async def _list_agents(self) -> Dict:
        async with get_db() as db:
            agents = await get_all_agents(db)
        return {
            "type": "tool_progress", "tool": "list_agents",
            "status": f"{len(agents)} agent(s)", "progress": 100,
            "result": {
                "agents": [{
                    "id": a.id, "name": a.name,
                    "model": a.model, "provider_id": getattr(a, "provider_id", None),
                    "description": a.description,
                } for a in agents]
            },
        }

    async def _create_agent(self, arguments: Dict) -> Dict:
        name = str(arguments.get("name", "")).strip()
        if not name:
            raise ValueError("name is required")
        async with get_db() as db:
            existing = await get_all_agents(db)
            if any(a.name.lower() == name.lower() for a in existing):
                raise ValueError(f"Agent '{name}' already exists")
            agent_data = {
                "name": name,
                "description": arguments.get("description") or "",
                "system_prompt": arguments.get("system_prompt") or "",
                "model": arguments.get("model") or "",
                "provider_id": arguments.get("provider_id"),
                "temperature": float(arguments.get("temperature") or 0.7),
                "top_k": int(arguments.get("top_k") or 40),
                "max_tokens": int(arguments.get("max_tokens") or 16048),
                "enable_rag": 1 if arguments.get("enable_rag") else 0,
            }
            agent = await create_agent(db, agent_data)
        return {
            "type": "tool_progress", "tool": "create_agent",
            "status": f"Agent '{agent.name}' created", "progress": 100,
            "result": {"id": agent.id, "name": agent.name, "model": agent.model,
                       "provider_id": getattr(agent, "provider_id", None)},
        }

    async def _delete_agent(self, arguments: Dict) -> Dict:
        from database.crud import delete_agent as _delete
        agent_id = str(arguments.get("id", "")).strip()
        if not agent_id:
            raise ValueError("id is required")
        async with get_db() as db:
            deleted = await _delete(db, int(agent_id) if str(agent_id).isdigit() else agent_id)
            await db.commit()
        if not deleted:
            raise ValueError(f"Agent '{agent_id}' not found")
        return {"type": "tool_progress", "tool": "delete_agent", "status": "Agent deleted",
                "progress": 100, "result": {"deleted": True, "id": agent_id}}

    # ── MCP servers ──────────────────────────────────────

    async def _list_mcp_servers(self) -> Dict:
        if not self.mcp_manager:
            raise ValueError("MCP manager unavailable")
        servers = await self.mcp_manager.list_servers()
        return {
            "type": "tool_progress", "tool": "list_mcp_servers",
            "status": f"{len(servers)} server(s)", "progress": 100,
            "result": {"servers": [{
                "name": s.get("name"),
                "transport_type": s.get("transport_type"),
                "is_connected": s.get("is_connected"),
                "tool_count": len(s.get("tools") or []),
            } for s in servers]},
        }

    async def _add_mcp_server(self, arguments: Dict) -> Dict:
        if not self.mcp_manager:
            raise ValueError("MCP manager unavailable")
        name = str(arguments.get("name", "")).strip()
        transport_type = str(arguments.get("transport_type") or "stdio").strip()
        command = arguments.get("command")
        args = arguments.get("args") or []
        env = arguments.get("env") or {}
        url = arguments.get("url")
        timeout = float(arguments.get("timeout") or 60.0)
        if not name:
            raise ValueError("name is required")
        if transport_type in ("sse", "streamable-http") and not url:
            raise ValueError("url is required for sse/streamable-http transport")
        if transport_type == "stdio" and not command:
            raise ValueError("command is required for stdio transport")
        success, error = await self.mcp_manager.add_server(
            name, command, args, env, transport_type, url, timeout=timeout)
        return {
            "type": "tool_progress", "tool": "add_mcp_server",
            "status": f"Server '{name}' {'connected' if success else 'added but connection failed'}",
            "progress": 100,
            "result": {"name": name, "connected": success, "error": error},
        }

    async def _remove_mcp_server(self, arguments: Dict) -> Dict:
        if not self.mcp_manager:
            raise ValueError("MCP manager unavailable")
        name = str(arguments.get("name", "")).strip()
        if not name:
            raise ValueError("name is required")
        success = await self.mcp_manager.remove_server(name)
        if not success:
            raise ValueError(f"Server '{name}' not found")
        return {"type": "tool_progress", "tool": "remove_mcp_server",
                "status": f"Server '{name}' removed", "progress": 100,
                "result": {"removed": True, "name": name}}

    # ── Providers ────────────────────────────────────────

    async def _list_providers(self) -> Dict:
        from database.provider_crud import list_providers
        async with get_db() as db:
            providers = await list_providers(db)
        return {
            "type": "tool_progress", "tool": "list_providers",
            "status": f"{len(providers)} provider(s)", "progress": 100,
            "result": {"providers": [{
                "id": p["id"], "name": p["name"], "base_url": p["base_url"],
                "is_default": p["is_default"], "model_count": len(p["models"] or []),
            } for p in providers]},
        }

    async def _add_provider(self, arguments: Dict) -> Dict:
        from database.provider_crud import create_provider, list_providers
        from tools.provider_service import fetch_models
        name = str(arguments.get("name", "")).strip()
        base_url = str(arguments.get("base_url", "")).strip().rstrip("/")
        api_key = str(arguments.get("api_key") or "").strip() or None
        if not name or not base_url:
            raise ValueError("name and base_url are required")
        try:
            models = await fetch_models(base_url, api_key)
            fetch_error = None
        except Exception as e:
            models = []
            fetch_error = str(e)[:300]
        async with get_db() as db:
            providers = await list_providers(db)
            if any(p["name"].lower() == name.lower() for p in providers):
                raise ValueError(f"Provider '{name}' already exists")
            provider = await create_provider(
                db, name, base_url, api_key=api_key, models=models,
                is_default=1 if not providers else 0)
            await db.commit()
        return {
            "type": "tool_progress", "tool": "add_provider",
            "status": f"Provider '{name}' added with {len(models)} models"
                      + (f" (fetch failed: {fetch_error})" if fetch_error else ""),
            "progress": 100,
            "result": {"id": provider["id"], "name": provider["name"],
                       "models_fetched": len(models), "error": fetch_error},
        }

    # ── Skills registry ──────────────────────────────────

    async def _search_skills(self, arguments: Dict) -> Dict:
        from tools.skill_registry import search_registry
        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit") or 15)
        if len(query) < 2:
            raise ValueError("query must be at least 2 characters")
        skills = await search_registry(query, limit)
        return {
            "type": "tool_progress", "tool": "search_skills",
            "status": f"{len(skills)} result(s) for '{query}'", "progress": 100,
            "result": {"query": query, "skills": skills},
        }

    async def _install_skill(self, arguments: Dict) -> Dict:
        from tools.skill_registry import install_registry_skill
        skill_id = str(arguments.get("id", "")).strip()
        if not skill_id:
            raise ValueError("id is required")
        result = await install_registry_skill(skill_id)
        return {
            "type": "tool_progress", "tool": "install_skill",
            "status": f"Installed skill '{result['name']}' ({len(result['files'])} files)",
            "progress": 100, "result": result,
        }

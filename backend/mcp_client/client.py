"""
Robust MCP Client using FastMCP library.

This module provides a clean, high-level interface for interacting with MCP servers
using the FastMCP Client class, which handles all protocol details and connection management.
"""

import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


from fastmcp import Client as FastMCPClient
from mcp.types import CallToolResult


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    # Transport type: 'stdio', 'sse', or 'http'
    transport_type: str = "stdio"
    # For stdio transport
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    # For HTTP/SSE transport
    url: Optional[str] = None
    # Custom HTTP headers (e.g. Authorization) for HTTP/SSE transport
    headers: Dict[str, str] = field(default_factory=dict)
    # Connection timeout in seconds
    timeout: float = 60.0
    # Whether server is enabled
    enabled: bool = True
    # Tool names to exclude from LLM function calling (to reduce prompt size)
    disabled_tools: List[str] = field(default_factory=list)


@dataclass
class MCPServerInstance:
    """Runtime instance of an MCP server connection."""
    config: MCPServerConfig
    client: Optional[FastMCPClient] = None
    tools: List[Dict] = field(default_factory=list)
    is_connected: bool = False
    is_initialized: bool = False
    error: Optional[str] = None

    def get_tool_full_name(self, tool_name: str) -> str:
        """Get full tool name with server prefix."""
        return f"{self.config.name}:{tool_name}"


class MCPClientManager:
    """
    Manages multiple MCP server connections using FastMCP.
    
    This manager handles:
    - Server lifecycle (connect, disconnect, reconnect)
    - Tool discovery and caching
    - Tool execution with proper error handling
    - Multiple transport types (stdio, SSE, HTTP)
    """

    # Browser-automation tools (browser-mcp's social_* family) perform real
    # human-paced actions: 4-45s gaps, scroll theater, typing rhythm, dwell
    # pauses, and post-click verification. They legitimately take minutes, so
    # they get a generous budget instead of the per-server config timeout
    # (default 60s), which regularly kills a LinkedIn social_post mid-action.
    SLOW_TOOL_TIMEOUT: float = 300.0
    SLOW_TOOL_NAMES: frozenset = frozenset({
        "social_post", "social_comment", "social_like",
        "social_read_feed", "social_status",
    })

    def __init__(self):
        self.servers: Dict[str, MCPServerInstance] = {}
        self._initialized = False
        self._connection_locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self):
        """Initialize all MCP servers from database."""
        if self._initialized:
            return

        from database.models import get_db
        from database.crud import get_enabled_mcp_servers

        async with get_db() as db:
            server_configs = await get_enabled_mcp_servers(db)

        tasks = []
        for config in server_configs:
            server_config = MCPServerConfig(
                name=config["name"],
                transport_type=config.get("transport_type", "stdio"),
                command=config.get("command"),
                args=config.get("args", []),
                env=config.get("env", {}),
                url=config.get("url"),
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 60.0),
                enabled=config.get("enabled", True),
                disabled_tools=config.get("disabled_tools", [])
            )
            
            if server_config.enabled:
                tasks.append(self._connect_server(server_config))

        if tasks:
            await asyncio.gather(*tasks)

        self._initialized = True

    async def _connect_server(self, config: MCPServerConfig) -> tuple[bool, str | None]:
        """
        Connect to an MCP server and discover its tools.

        Args:
            config: Server configuration

        Returns:
            Tuple of (is_connected, error_message)
        """
        try:
            # Create lock for this server if not exists
            if config.name not in self._connection_locks:
                self._connection_locks[config.name] = asyncio.Lock()

            async with self._connection_locks[config.name]:
                # Create server instance first and register it in memory
                instance = MCPServerInstance(config=config)
                self.servers[config.name] = instance

                try:
                    # Build FastMCP client based on transport type
                    client = self._create_client(config)
                    instance.client = client

                    # Connect and initialize
                    try:
                        # For all transport types, use the client's context manager
                        # FastMCP handles transport-specific connection details internally
                        async def _connect_and_list():
                            async with client:
                                return await client.list_tools()

                        # Apply timeout to connection and tool list discovery
                        tools = await asyncio.wait_for(_connect_and_list(), timeout=config.timeout)
                        instance.tools = self._parse_tools(tools, config.name)
                        instance.is_connected = True
                        instance.is_initialized = True
                        instance.error = None

                        print(f"Connected to MCP server '{config.name}': {len(instance.tools)} tools available")

                    except BaseException as conn_error:
                        instance.error = str(conn_error)
                        instance.is_connected = False
                        instance.is_initialized = False
                        print(f"Failed to connect to MCP server '{config.name}': {conn_error}")
                        import traceback
                        traceback.print_exc()

                except BaseException as create_error:
                    instance.error = str(create_error)
                    instance.is_connected = False
                    instance.is_initialized = False
                    print(f"Failed to create MCP client for '{config.name}': {create_error}")
                    import traceback
                    traceback.print_exc()

                return instance.is_connected, instance.error

        except BaseException as e:
            print(f"Error in _connect_server for '{config.name}': {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _create_client(self, config: MCPServerConfig) -> FastMCPClient:
        """
        Create a FastMCP client based on transport type.

        Args:
            config: Server configuration

        Returns:
            FastMCP Client instance
        """
        from fastmcp.client.transports import StdioTransport
        from fastmcp.client.transports import SSETransport, StreamableHttpTransport
        
        if config.transport_type in ("sse", "http", "streamable-http"):
            # HTTP/SSE transport
            if not config.url:
                raise ValueError(f"URL required for {config.transport_type} transport")
            transport_cls = SSETransport if config.transport_type == "sse" else StreamableHttpTransport
            headers = config.headers or None
            if headers:
                return FastMCPClient(transport_cls(url=config.url, headers=headers), timeout=config.timeout)
            return FastMCPClient(config.url, timeout=config.timeout)

        elif config.transport_type == "stdio":
            # Stdio transport
            if not config.command:
                raise ValueError("Command required for stdio transport")

            command_path = config.command
            if not Path(command_path).is_absolute():
                resolved = shutil.which(command_path)
                if resolved:
                    command_path = resolved

            if not Path(command_path).exists():
                raise FileNotFoundError(
                    f"Command not found for stdio transport: {config.command}. "
                    f"Resolved path: {command_path}. "
                    "On Windows, shell builtins like 'echo' are not valid executables; "
                    "use a real executable or run through cmd.exe /c <command>."
                )

            # For stdio transport, we need to use StdioTransport explicitly
            # This properly handles commands like 'uvx', 'npx' with arguments
            # Pass environment variables to the transport
            transport_kwargs = {
                "command": command_path,
                "args": config.args,
            }
            # Only add env if it's not empty
            if config.env:
                transport_kwargs["env"] = config.env
            
            return FastMCPClient(
                StdioTransport(**transport_kwargs)
            )

        else:
            raise ValueError(f"Unknown transport type: {config.transport_type}")

    def _parse_tools(self, tools_result, server_name: str) -> List[Dict]:
        """
        Parse tools from FastMCP list_tools result.
        
        Args:
            tools_result: Result from client.list_tools()
            server_name: Name of the server
            
        Returns:
            List of tool dictionaries in standard format
        """
        parsed_tools = []
        
        # Debug: log the type and sample of tools_result
        print(f"[DEBUG] tools_result type: {type(tools_result)}, length: {len(tools_result) if hasattr(tools_result, '__len__') else 'N/A'}")
        if len(tools_result) > 0:
            first_tool = tools_result[0]
            print(f"[DEBUG] First tool type: {type(first_tool)}")
            print(f"[DEBUG] First tool dir: {[attr for attr in dir(first_tool) if not attr.startswith('_')][:20]}")
            if hasattr(first_tool, '__dict__'):
                print(f"[DEBUG] First tool attrs: {first_tool.__dict__}")
        
        # FastMCP returns tools as a list of Tool objects or dicts
        # Handle both formats
        for tool in tools_result:
            try:
                # Try to get tool attributes - handle both object and dict formats
                if hasattr(tool, 'name'):
                    # Tool object format
                    tool_name = tool.name
                    tool_description = getattr(tool, 'description', '') or ''
                    input_schema = getattr(tool, 'inputSchema', None) or getattr(tool, 'input_schema', None) or {"type": "object", "properties": {}}
                elif isinstance(tool, dict):
                    # Dict format
                    tool_name = tool.get('name', '')
                    tool_description = tool.get('description', '') or ''
                    input_schema = tool.get('inputSchema', None) or tool.get('input_schema', None) or {"type": "object", "properties": {}}
                else:
                    # Unknown format, skip
                    print(f"Warning: Unknown tool format: {type(tool)}")
                    continue
                
                if not tool_name:
                    continue
                    
                tool_info = {
                    "server": server_name,
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema
                }
                parsed_tools.append(tool_info)
                
            except Exception as e:
                print(f"Error parsing tool: {e}, tool: {tool}")
                import traceback
                traceback.print_exc()
                continue
        
        return parsed_tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result as dictionary
            
        Raises:
            ValueError: If server not found or not connected
            Exception: If tool call fails
        """
        if server_name not in self.servers:
            raise ValueError(f"MCP server '{server_name}' not found")

        instance = self.servers[server_name]
        
        if not instance.is_connected or not instance.client:
            # Try to reconnect
            print(f"Reconnecting to MCP server '{server_name}'...")
            success = await self._connect_server(instance.config)
            if not success:
                raise ValueError(f"MCP server '{server_name}' is not connected and reconnection failed")

        try:
            # FastMCP requires using the context manager for tool calls
            # The context manager handles session lifecycle
            async def _call():
                async with instance.client:
                    return await instance.client.call_tool(tool_name, arguments)

            # Slow browser-automation tools get a longer budget than the
            # per-server config timeout so humanized pacing isn't cut off.
            timeout = instance.config.timeout
            if tool_name in self.SLOW_TOOL_NAMES:
                timeout = max(timeout, self.SLOW_TOOL_TIMEOUT)

            result: CallToolResult = await asyncio.wait_for(_call(), timeout=timeout)
            
            # Parse result - FastMCP returns CallToolResult with content list
            return self._parse_tool_result(result)
                
        except Exception as e:
            print(f"Tool call failed for '{server_name}:{tool_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_tool_result(self, result: CallToolResult) -> Dict[str, Any]:
        """Parse FastMCP CallToolResult to dict with text content and sources."""
        parsed = {"content": [], "is_error": getattr(result, 'isError', False), "sources": []}

        if result.content:
            for item in result.content:
                if hasattr(item, 'type') and item.type == "text":
                    text = item.text if hasattr(item, 'text') else str(item)
                    parsed["content"].append({"type": "text", "text": text})
                    text_sources = self._extract_sources_from_text(text)
                    if text_sources:
                        parsed["sources"].extend(text_sources)
                    if hasattr(item, 'source') and item.source:
                        parsed["sources"].append({"title": str(item.source), "url": str(item.source)})
                else:
                    parsed["content"].append({"type": getattr(item, 'type', 'unknown'), "data": str(item)})

        # Deduplicate sources by URL
        seen = set()
        parsed["sources"] = [s for s in parsed.get("sources", []) if not (s.get("url") and s["url"] in seen or seen.add(s["url"]))]
        return parsed

    @staticmethod
    def _safe_url(url_str: str) -> str:
        """Return a safe absolute URL or extract one from text. Returns empty string if unsafe."""
        if not url_str:
            return ""
        s = str(url_str)
        if s.startswith(('http://', 'https://', '#', '/', 'data:', 'ftp://', 'file://', 'mailto:')):
            return s
        import re
        m = re.search(r'https?://\S+', s)
        return m.group(0) if m else ""

    @staticmethod
    def _extract_url_from_text(text: str) -> str:
        import re
        match = re.search(r'https?://\S+', text)
        return match.group(0) if match else ""

    def _normalize_sources(self, sources_list: list) -> list:
        """Normalize sources from various MCP formats to {title, url, snippet, document_id}."""
        normalized = []
        for source in sources_list:
            if not isinstance(source, dict):
                continue
            title = source.get("title") or source.get("name") or str(source)
            url = self._safe_url(source.get("url") or source.get("uri") or source.get("link") or source.get("href") or "")
            if not url:
                continue
            snippet = source.get("snippet") or source.get("description") or source.get("text") or ""
            doc_id = source.get("document_id") or source.get("id") or ""
            normalized.append({"title": str(title), "url": url, "snippet": str(snippet), "document_id": str(doc_id)})
        return normalized

    def _extract_sources_from_text(self, text: str) -> list:
        """Extract sources from text: SOURCES section or JSON array."""
        import json, re

        # Try plain text SOURCES format first
        text_stripped = text.strip()
        for marker in ['SOURCES:', 'SOURCES', 'REFERENCES:', 'REFERENCES', 'LINKS:', 'LINKS']:
            idx = text_stripped.find(marker)
            if idx >= 0:
                rest = text_stripped[idx + len(marker):].strip()
                sources = []
                for line in rest.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # [N] Title — URL  or  [N] URL
                    m = re.match(r'\[\s*\d+\s*\]\s*(?:([^—\[\]]+?)\s*[—\-@:]\s+)?(https?://\S+)', line)
                    if m:
                        t = (m.group(1) or m.group(2)).strip()
                        url = m.group(2)
                        sources.append({"title": t, "url": url, "snippet": ""})
                if sources:
                    return sources

        # Try JSON array
        bracket_start = text.find('[')
        if bracket_start >= 0:
            bracket_end = text.rfind(']')
            if bracket_end > bracket_start:
                try:
                    parsed = json.loads(text[bracket_start:bracket_end + 1])
                    if isinstance(parsed, list):
                        return self._normalize_sources(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
        return []



    async def list_all_tools(self, include_disabled: bool = True) -> List[Dict]:
        """
        List all tools from all connected servers.
        
        Args:
            include_disabled: If True, includes disabled tools too (used for UI display).
                If False, filters out disabled tools (used for LLM tool definitions).

        Returns:
            List of tool dictionaries with server info
        """
        all_tools = []

        for server_name, instance in self.servers.items():
            if instance.is_connected:
                disabled = set(instance.config.disabled_tools or [])
                for tool in instance.tools:
                    if not include_disabled and tool.get("name") in disabled:
                        continue
                    # Add server name to each tool
                    tool_with_server = {**tool, 'server': server_name}
                    all_tools.append(tool_with_server)

        return all_tools

    async def list_servers(self) -> List[Dict]:
        """
        List all registered MCP servers with their status.
        
        Returns:
            List of server status dictionaries
        """
        return [
            {
                "name": instance.config.name,
                "transport_type": instance.config.transport_type,
                "command": instance.config.command,
                "args": instance.config.args,
                "env": instance.config.env,
                "url": instance.config.url,
                "headers": instance.config.headers,
                "tool_count": len(instance.tools),
                "is_connected": instance.is_connected,
                "is_initialized": instance.is_initialized,
                "error": instance.error
            }
            for instance in self.servers.values()
        ]

    async def add_server(
        self,
        name: str,
        command: Optional[str] = None,
        args: List[str] = None,
        env: Dict[str, str] = None,
        transport_type: str = "stdio",
        url: Optional[str] = None,
        timeout: float = 60.0,
        headers: Dict[str, str] = None
    ) -> bool:
        """
        Add and connect to a new MCP server.
        
        Args:
            name: Server name
            command: Command for stdio transport
            args: Command arguments
            env: Environment variables
            transport_type: Transport type ('stdio', 'sse', 'http')
            url: URL for HTTP/SSE transport
            timeout: Connection timeout in seconds
            headers: Custom HTTP headers (e.g. Authorization) for HTTP/SSE transport
            
        Returns:
            True if server was added and connected successfully
        """
        from database.models import get_db
        from database.crud import add_mcp_server

        args = args or []
        env = env or {}
        headers = headers or {}
        
        # Save to database
        async with get_db() as db:
            await add_mcp_server(db, name, command, args, env, transport_type, url, timeout=timeout, headers=headers)

        # Create config and connect
        config = MCPServerConfig(
            name=name,
            transport_type=transport_type,
            command=command,
            args=args,
            env=env,
            url=url,
            headers=headers,
            timeout=timeout
        )

        success, error = await self._connect_server(config)
        return success, error

    async def remove_server(self, name: str) -> bool:
        """
        Remove an MCP server.
        
        Args:
            name: Server name to remove
            
        Returns:
            True if server was removed
        """
        # Remove from database first
        from database.crud import remove_mcp_server
        from database.models import get_db
        async with get_db() as db:
            db_removed = await remove_mcp_server(db, name)

        in_memory = name in self.servers
        if in_memory:
            instance = self.servers[name]
            
            # Disconnect client if connected
            if instance.client and instance.is_connected:
                try:
                    await instance.client.close()
                except Exception:
                    pass

            # Remove from registry
            del self.servers[name]

        return db_removed or in_memory

    async def reconnect_server(self, name: str) -> bool:
        """
        Reconnect to a server (useful if connection was lost).
        
        Args:
            name: Server name
            
        Returns:
            True if reconnection succeeded
        """
        if name not in self.servers:
            return False
            
        instance = self.servers[name]
        
        # Close existing connection
        if instance.client:
            try:
                await instance.client.close()
            except Exception:
                pass
                
        instance.is_connected = False
        instance.is_initialized = False
        
        # Reconnect
        return await self._connect_server(instance.config)

    async def refresh_tools(self, server_name: str) -> bool:
        """
        Refresh tool list for a server.
        
        Args:
            server_name: Server name
            
        Returns:
            True if refresh succeeded
        """
        if server_name not in self.servers:
            return False
            
        instance = self.servers[server_name]
        
        if not instance.is_connected or not instance.client:
            return False
            
        try:
            async def _refresh():
                async with instance.client:
                    return await instance.client.list_tools()

            tools = await asyncio.wait_for(_refresh(), timeout=instance.config.timeout)
            instance.tools = self._parse_tools(tools, server_name)
            return True
        except Exception:
            return False

    async def cleanup(self):
        """Clean up all server connections."""
        for instance in self.servers.values():
            if instance.client and instance.is_connected:
                try:
                    await instance.client.close()
                except Exception:
                    pass
                    
        self.servers.clear()
        self._connection_locks.clear()





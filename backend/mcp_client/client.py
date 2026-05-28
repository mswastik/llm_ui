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
from contextlib import asynccontextmanager

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
    # Connection timeout in seconds
    timeout: float = 30.0
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
        
        if config.transport_type in ("sse", "http", "streamable-http"):
            # HTTP/SSE transport
            if not config.url:
                raise ValueError(f"URL required for {config.transport_type} transport")
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

            result: CallToolResult = await asyncio.wait_for(_call(), timeout=instance.config.timeout)
            
            # Parse result - FastMCP returns CallToolResult with content list
            return self._parse_tool_result(result)
                
        except Exception as e:
            print(f"Tool call failed for '{server_name}:{tool_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_tool_result(self, result: CallToolResult) -> Dict[str, Any]:
        """
        Parse FastMCP CallToolResult to dictionary.
        
        Args:
            result: CallToolResult from FastMCP
            
        Returns:
            Dictionary with tool result data, including extracted sources
        """
        # CallToolResult has:
        # - content: list of content items (text, image, resource, etc.)
        # - isError: bool indicating if this is an error
        # - structuredContent: optional structured content dict
        
        parsed = {
            "content": [],
            "is_error": result.isError if hasattr(result, 'isError') else False,
            "structured_content": None,
            "sources": []
        }
        
        # Handle structured content if available
        if hasattr(result, 'structuredContent') and result.structuredContent:
            parsed["structured_content"] = result.structuredContent
            # Extract sources from structured content (common pattern for search MCP servers)
            if isinstance(result.structuredContent, dict):
                structured = result.structuredContent
                # Check for sources in various common key names
                for key in ('sources', 'documents', 'results', 'hits', 'items'):
                    if key in structured and isinstance(structured[key], list):
                        parsed["sources"].extend(self._normalize_sources(structured[key]))
                        break
                # Also check structuredContent itself is a list
                if not parsed["sources"] and isinstance(structured, list):
                    parsed["sources"].extend(self._normalize_sources(structured))
            
        # Handle content list and extract sources from text items
        if result.content:
            for item in result.content:
                if hasattr(item, 'type'):
                    if item.type == "text":
                        text = item.text if hasattr(item, 'text') else str(item)
                        parsed["content"].append({
                            "type": "text",
                            "text": text
                        })
                        # Try to extract sources from text content (JSON array)
                        text_sources = self._extract_sources_from_text(text)
                        if text_sources:
                            parsed["sources"].extend(text_sources)
                        # Check if item has a source URL (MCP spec citation)
                        if hasattr(item, 'source') and item.source:
                            parsed["sources"].append({
                                "title": str(item.source),
                                "url": str(item.source)
                            })
                    elif item.type == "image":
                        parsed["content"].append({
                            "type": "image",
                            "data": item.data if hasattr(item, 'data') else None,
                            "mime_type": item.mimeType if hasattr(item, 'mimeType') else None
                        })
                    elif item.type == "resource":
                        parsed["content"].append({
                            "type": "resource",
                            "resource": item.resource if hasattr(item, 'resource') else str(item)
                        })
                    else:
                        parsed["content"].append({
                            "type": item.type,
                            "data": str(item)
                        })
                else:
                    parsed["content"].append({"type": "unknown", "data": str(item)})
        
        # Deduplicate sources by URL
        if parsed["sources"]:
            seen_urls = set()
            unique_sources = []
            for s in parsed["sources"]:
                url = s.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(s)
                elif not url and s not in unique_sources:
                    unique_sources.append(s)
            parsed["sources"] = unique_sources
                    
        return parsed

    def _normalize_sources(self, sources_list: list) -> list:
        """
        Normalize sources from various MCP server formats to a standard format.
        
        Standard format: {title, url, snippet, document_id}
        
        Handles formats like:
        - [{title, url, snippet}]
        - [{name, uri, description}]
        - [{text, source, ...}]
        - [{url, title, ...}]
        """
        normalized = []
        for source in sources_list:
            if not isinstance(source, dict):
                continue
            
            # Extract title
            title = (
                source.get("title") or
                source.get("name") or
                source.get("label") or
                source.get("filename") or
                source.get("source") or
                str(source)
            )
            
            # Extract URL
            url = (
                source.get("url") or
                source.get("uri") or
                source.get("link") or
                source.get("href") or
                source.get("source") or
                ""
            )
            
            # Extract snippet/description
            snippet = (
                source.get("snippet") or
                source.get("description") or
                source.get("summary") or
                source.get("text") or
                ""
            )
            
            # Extract document_id
            doc_id = (
                source.get("document_id") or
                source.get("id") or
                source.get("doc_id") or
                ""
            )
            
            normalized.append({
                "title": str(title),
                "url": str(url),
                "snippet": str(snippet),
                "document_id": str(doc_id)
            })
        
        return normalized

    def _extract_sources_from_text(self, text: str) -> list:
        """
        Try to extract sources from text content.
        
        Handles multiple formats:
        1. Plain text SOURCES section: "SOURCES:\n[1] Title — URL\n[2] Title — URL"
        2. JSON array: [{title, url}, ...]
        3. JSON array embedded in text
        """
        import json
        
        # Try plain text SOURCES format first (most common for search MCP servers)
        # Format: SOURCES:\n[1] Title — URL\n[2] Title — URL
        text_sources = self._extract_sources_from_text_format(text)
        if text_sources:
            return text_sources
        
        # Try JSON array
        return self._extract_sources_from_json(text)

    def _extract_sources_from_text_format(self, text: str) -> list:
        """
        Extract sources from plain text SOURCES section.
        
        Handles formats like:
        SOURCES:
        [1] Title — URL
        [2] Title — URL
        
        Also handles variations:
        SOURCES:\n[1] Title - URL\n[2] Title @ URL
        """
        text = text.strip()
        
        # Find SOURCES section
        sources_marker = None
        for marker in ['SOURCES:', 'SOURCES', 'REFERENCES:', 'REFERENCES', 'LINKS:', 'LINKS']:
            idx = text.find(marker)
            if idx != -1:
                sources_marker = idx
                break
        
        if sources_marker is None:
            return []
        
        sources_text = text[sources_marker + len(marker):].strip()
        if not sources_text:
            return []
        
        sources = []
        # Match lines like: [1] Title — URL or [1] Title - URL or [1] Title @ URL
        # Also handle: [1] URL (just URL, no title)
        import re
        for line in sources_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Pattern: [N] Title — URL (or - or @ or :)
            match = re.match(r'\[\s*(\d+)\s*\]\s*(.+?)\s*[—\-\-@:]+\s*(.+)', line)
            if match:
                title = match.group(2).strip()
                url = match.group(3).strip()
                sources.append({
                    "title": title,
                    "url": url,
                    "snippet": ""
                })
                continue
            
            # Pattern: [N] URL (just URL, no title)
            match = re.match(r'\[\s*(\d+)\s*\]\s*(https?://.+)', line)
            if match:
                url = match.group(2).strip()
                sources.append({
                    "title": url,
                    "url": url,
                    "snippet": ""
                })
                continue
        
        return sources if sources else []

    def _extract_sources_from_json(self, text: str) -> list:
        """
        Try to extract a JSON array of sources from text content.
        
        Looks for patterns like:
        - [\n  { ... sources ... }\n]\n at the end of text
        - JSON array anywhere in text
        """
        import json
        
        # Try to find a JSON array at the end of the text
        text = text.strip()
        if not text.startswith('['):
            # Try to find JSON array embedded in text
            bracket_start = text.find('[\n')
            if bracket_start == -1:
                bracket_start = text.find('[ ')
            if bracket_start == -1:
                bracket_start = text.find('[')
            
            if bracket_start == -1:
                return []
            
            # Find the matching closing bracket
            bracket_end = text.rfind(']')
            if bracket_end == -1:
                return []
            
            json_str = text[bracket_start:bracket_end + 1]
        else:
            json_str = text
        
        try:
            sources = json.loads(json_str)
            if isinstance(sources, list):
                return self._normalize_sources(sources)
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
        timeout: float = 30.0
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
            
        Returns:
            True if server was added and connected successfully
        """
        from database.models import get_db
        from database.crud import add_mcp_server

        args = args or []
        env = env or {}
        
        # Save to database
        async with get_db() as db:
            await add_mcp_server(db, name, command, args, env, transport_type, url)

        # Create config and connect
        config = MCPServerConfig(
            name=name,
            transport_type=transport_type,
            command=command,
            args=args,
            env=env,
            url=url,
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

    @asynccontextmanager
    async def get_server_client(self, server_name: str):
        """
        Context manager to get a server's client for direct access.
        
        Usage:
            async with manager.get_server_client("my_server") as client:
                tools = await client.list_tools()
                
        Args:
            server_name: Server name
            
        Yields:
            FastMCP Client instance
        """
        if server_name not in self.servers:
            raise ValueError(f"MCP server '{server_name}' not found")
            
        instance = self.servers[server_name]
        
        if not instance.client:
            raise ValueError(f"No client available for '{server_name}'")
            
        # The client manages its own context
        async with instance.client:
            yield instance.client


# Global instance for convenience
_mcp_manager: Optional[MCPClientManager] = None


def get_mcp_manager() -> MCPClientManager:
    """Get or create the global MCP manager instance."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    return _mcp_manager

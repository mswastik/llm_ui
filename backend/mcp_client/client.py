"""
Robust MCP Client using FastMCP library.

This module provides a clean, high-level interface for interacting with MCP servers
using the FastMCP Client class, which handles all protocol details and connection management.
"""

import asyncio
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
    # env: Dict[str, str] = field(default_factory=dict)
    # For HTTP/SSE transport
    url: Optional[str] = None
    # Connection timeout in seconds
    timeout: float = 30.0
    # Whether server is enabled
    enabled: bool = True


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

        for config in server_configs:
            server_config = MCPServerConfig(
                name=config["name"],
                transport_type=config.get("transport_type", "stdio"),
                command=config.get("command"),
                args=config.get("args", []),
                #env=config.get("env", {}),
                url=config.get("url"),
                enabled=config.get("enabled", True)
            )
            
            if server_config.enabled:
                await self._connect_server(server_config)

        self._initialized = True

    async def _connect_server(self, config: MCPServerConfig) -> bool:
        """
        Connect to an MCP server and discover its tools.

        Args:
            config: Server configuration

        Returns:
            True if connection and discovery succeeded
        """
        try:
            # Create lock for this server if not exists
            if config.name not in self._connection_locks:
                self._connection_locks[config.name] = asyncio.Lock()

            async with self._connection_locks[config.name]:
                # Build FastMCP client based on transport type
                client = self._create_client(config)

                # Create server instance
                instance = MCPServerInstance(config=config, client=client)

                # Connect and initialize
                try:
                    # For all transport types, use the client's context manager
                    # FastMCP handles transport-specific connection details internally
                    async with client:
                        # Discover tools
                        tools = await client.list_tools()
                        instance.tools = self._parse_tools(tools, config.name)
                        instance.is_connected = True
                        instance.is_initialized = True
                        instance.error = None

                    print(f"Connected to MCP server '{config.name}': {len(instance.tools)} tools available")

                except Exception as conn_error:
                    instance.error = str(conn_error)
                    instance.is_connected = False
                    instance.is_initialized = False
                    print(f"Failed to connect to MCP server '{config.name}': {conn_error}")
                    import traceback
                    traceback.print_exc()

                self.servers[config.name] = instance
                return instance.is_connected

        except Exception as e:
            print(f"Error connecting to MCP server '{config.name}': {e}")
            import traceback
            traceback.print_exc()
            return False

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

            # For stdio transport, we need to use StdioTransport explicitly
            # This properly handles commands like 'uvx', 'npx' with arguments
            return FastMCPClient(
                StdioTransport(
                    command=config.command,
                    args=config.args,
                    #timeout=config.timeout
                )
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
            async with instance.client:
                result: CallToolResult = await instance.client.call_tool(tool_name, arguments)
            
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
            Dictionary with tool result data
        """
        # CallToolResult has:
        # - content: list of content items (text, image, resource, etc.)
        # - isError: bool indicating if this is an error
        # - structuredContent: optional structured content dict
        
        parsed = {
            "content": [],
            "is_error": result.isError if hasattr(result, 'isError') else False,
            "structured_content": None
        }
        
        # Handle structured content if available
        if hasattr(result, 'structuredContent') and result.structuredContent:
            parsed["structured_content"] = result.structuredContent
            
        # Handle content list
        if result.content:
            for item in result.content:
                if hasattr(item, 'type'):
                    if item.type == "text":
                        parsed["content"].append({
                            "type": "text",
                            "text": item.text if hasattr(item, 'text') else str(item)
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
                    
        return parsed

    async def list_all_tools(self) -> List[Dict]:
        """
        List all tools from all connected servers.

        Returns:
            List of tool dictionaries with server info
        """
        all_tools = []

        for server_name, instance in self.servers.items():
            if instance.is_connected:
                for tool in instance.tools:
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
            #env=env,
            url=url,
            timeout=timeout
        )

        return await self._connect_server(config)

    async def remove_server(self, name: str) -> bool:
        """
        Remove an MCP server.
        
        Args:
            name: Server name to remove
            
        Returns:
            True if server was removed
        """
        if name not in self.servers:
            return False

        instance = self.servers[name]
        
        # Disconnect client if connected
        if instance.client and instance.is_connected:
            try:
                await instance.client.close()
            except Exception:
                pass

        # Remove from registry
        del self.servers[name]

        # Remove from database
        from database.crud import remove_mcp_server
        from database.models import get_db
        async with get_db() as db:
            await remove_mcp_server(db, name)

        return True

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
            async with instance.client:
                tools = await instance.client.list_tools()
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

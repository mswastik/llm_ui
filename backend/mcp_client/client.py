import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import subprocess

from database.models import get_db
from database.crud import get_enabled_mcp_servers, add_mcp_server as db_add_mcp_server

# HTTP client for SSE and StreamableHTTP transports
import aiohttp


@dataclass
class MCPServer:
    """Represents an MCP server connection"""
    name: str
    transport_type: str  # 'stdio', 'sse', or 'streamable-http'
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    tools: List[Dict] = field(default_factory=list)
    http_session: Optional[aiohttp.ClientSession] = None
    request_id: int = 0
    
    def get_next_request_id(self) -> int:
        """Get next unique request ID for JSON-RPC"""
        self.request_id += 1
        return self.request_id


class MCPClientManager:
    """Manages multiple MCP server connections"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize all MCP servers from database"""
        if self._initialized:
            return

        async with get_db() as db:
            server_configs = await get_enabled_mcp_servers(db)

        for config in server_configs:
            await self._start_server(
                config["name"],
                config.get("command"),
                config.get("args", []),
                config.get("env", {}),
                config.get("transport_type", "stdio"),
                config.get("url")
            )

        self._initialized = True
    
    async def _start_server(
        self,
        name: str,
        command: Optional[str],
        args: List[str],
        env: Dict[str, str],
        transport_type: str = "stdio",
        url: Optional[str] = None
    ) -> bool:
        """Start an MCP server process or connect via HTTP"""
        try:
            if transport_type in ("sse", "streamable-http"):
                # HTTP-based transport
                if not url:
                    print(f"Error: URL required for {transport_type} transport")
                    return False
                
                # Create HTTP session
                http_session = aiohttp.ClientSession()
                
                server = MCPServer(
                    name=name,
                    transport_type=transport_type,
                    command=command,
                    args=args,
                    env=env,
                    url=url,
                    http_session=http_session
                )
                
                # Discover tools from the server
                await self._discover_tools(server)
                
            elif transport_type == "stdio":
                # Stdio-based transport (existing behavior)
                if not command:
                    print(f"Error: Command required for stdio transport")
                    return False
                
                # Prepare environment - inherit from current process to ensure PATH is available
                import os
                process_env = os.environ.copy()
                process_env.update(env)

                # Start the MCP server process
                process = await asyncio.create_subprocess_exec(
                    command,
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=process_env
                )

                server = MCPServer(
                    name=name,
                    transport_type=transport_type,
                    command=command,
                    args=args,
                    env=env,
                    process=process
                )

                # Discover tools from the server
                await self._discover_tools(server)
            else:
                print(f"Unknown transport type: {transport_type}")
                return False

            self.servers[name] = server
            return True

        except Exception as e:
            print(f"Failed to start MCP server '{name}': {e}")
            return False
    
    async def _discover_tools(self, server: MCPServer):
        """Discover available tools from an MCP server"""
        try:
            if server.transport_type in ("sse", "streamable-http"):
                # HTTP-based tool discovery
                await self._discover_tools_http(server)
            else:
                # Stdio-based tool discovery (existing behavior)
                await self._discover_tools_stdio(server)
        except Exception as e:
            print(f"Failed to discover tools from '{server.name}': {e}")
            server.tools = []
    
    async def _discover_tools_stdio(self, server: MCPServer):
        """Discover tools via stdio transport"""
        try:
            # Send tools/list request according to MCP protocol
            request = {
                "jsonrpc": "2.0",
                "id": server.get_next_request_id(),
                "method": "tools/list",
                "params": {}
            }

            # Write request to server's stdin
            request_json = json.dumps(request) + "\n"
            server.process.stdin.write(request_json.encode())
            await server.process.stdin.drain()

            # Read response from stdout
            response_line = await server.process.stdout.readline()
            if not response_line:
                # Check stderr for errors if stdout is empty
                error_output = await server.process.stderr.read(1024)
                print(f"MCP server '{server.name}' closed stdout unexpectedly. Stderr: {error_output.decode()}")
                server.tools = []
                return

            try:
                response = json.loads(response_line.decode())
                if "result" in response and "tools" in response["result"]:
                    server.tools = response["result"]["tools"]
            except json.JSONDecodeError as e:
                # Try to read some more from stdout/stderr to see what happened
                remaining = await server.process.stdout.read(1024)
                stderr_output = await server.process.stderr.read(1024)
                print(f"Failed to parse JSON from '{server.name}': {e}")
                print(f"Raw response line: {response_line.decode()}")
                print(f"Remaining stdout: {remaining.decode()}")
                print(f"Stderr: {stderr_output.decode()}")
                server.tools = []

        except Exception as e:
            print(f"Failed to discover tools from '{server.name}': {e}")
            server.tools = []
    
    async def _discover_tools_http(self, server: MCPServer):
        """Discover tools via SSE or StreamableHTTP transport"""
        try:
            # For SSE and StreamableHTTP, we typically POST to an endpoint
            # MCP over HTTP usually uses /mcp or /messages endpoint
            endpoint = server.url.rstrip('/')
            
            request = {
                "jsonrpc": "2.0",
                "id": server.get_next_request_id(),
                "method": "tools/list",
                "params": {}
            }
            
            async with server.http_session.post(
                endpoint,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result and "tools" in result["result"]:
                        server.tools = result["result"]["tools"]
                else:
                    print(f"HTTP tool discovery failed for '{server.name}': {response.status}")
                    server.tools = []
                    
        except Exception as e:
            print(f"Failed to discover tools via HTTP from '{server.name}': {e}")
            server.tools = []
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"MCP server '{server_name}' not found")

        server = self.servers[server_name]
        
        if server.transport_type in ("sse", "streamable-http"):
            return await self._call_tool_http(server, tool_name, arguments)
        else:
            return await self._call_tool_stdio(server, tool_name, arguments)
    
    async def _call_tool_stdio(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via stdio transport"""
        # Send tools/call request according to MCP protocol
        request = {
            "jsonrpc": "2.0",
            "id": server.get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        # Write request
        request_json = json.dumps(request) + "\n"
        server.process.stdin.write(request_json.encode())
        await server.process.stdin.drain()

        # Read response
        response_line = await server.process.stdout.readline()
        response = json.loads(response_line.decode())

        if "result" in response:
            return response["result"]
        elif "error" in response:
            raise Exception(f"MCP tool call failed: {response['error']}")
        else:
            raise Exception("Invalid MCP response")
    
    async def _call_tool_http(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via SSE or StreamableHTTP transport"""
        endpoint = server.url.rstrip('/')
        
        request = {
            "jsonrpc": "2.0",
            "id": server.get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        async with server.http_session.post(
            endpoint,
            json=request,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    raise Exception(f"MCP tool call failed: {result['error']}")
                else:
                    raise Exception("Invalid MCP response")
            else:
                raise Exception(f"HTTP tool call failed with status: {response.status}")
    
    async def list_all_tools(self) -> List[Dict]:
        """List all tools from all servers"""
        all_tools = []
        
        for server_name, server in self.servers.items():
            for tool in server.tools:
                all_tools.append({
                    "server": server_name,
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {})
                })
        
        return all_tools
    
    async def list_servers(self) -> List[Dict]:
        """List all active MCP servers"""
        return [
            {
                "name": server.name,
                "transport_type": server.transport_type,
                "command": server.command,
                "url": server.url,
                "tool_count": len(server.tools)
            }
            for server in self.servers.values()
        ]
    
    async def add_server(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Dict[str, str],
        transport_type: str = "stdio",
        url: Optional[str] = None
    ) -> bool:
        """Add a new MCP server"""
        # Save to database
        async with get_db() as db:
            await db_add_mcp_server(db, name, command, args, env, transport_type, url)

        # Start the server
        return await self._start_server(name, command, args, env, transport_type, url)
    
    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server"""
        if name not in self.servers:
            return False
        
        server = self.servers[name]
        
        # Terminate the process
        if server.process:
            try:
                # Check if process is still running
                if server.process.returncode is None:
                    server.process.terminate()
                    await server.process.wait()
            except ProcessLookupError:
                # Process already exited
                pass
            except Exception as e:
                print(f"Error terminating MCP server process: {e}")
        
        # Remove from active servers
        del self.servers[name]
        
        # Remove from database
        from database.crud import remove_mcp_server as db_remove_mcp_server
        async with get_db() as db:
            await db_remove_mcp_server(db, name)
        
        return True
    
    async def cleanup(self):
        """Clean up all server processes and HTTP sessions"""
        for server in self.servers.values():
            if server.transport_type == "stdio" and server.process:
                try:
                    # Check if process is still running
                    if server.process.returncode is None:
                        server.process.terminate()
                        await server.process.wait()
                except (ProcessLookupError, Exception):
                    pass
            elif server.transport_type in ("sse", "streamable-http") and server.http_session:
                # Close HTTP session
                try:
                    await server.http_session.close()
                except Exception:
                    pass

        self.servers.clear()

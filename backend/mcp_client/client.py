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
    session_id: Optional[str] = None  # For FastMCP SSE transport
    
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
            # Wait a bit for the server to initialize
            await asyncio.sleep(0.5)

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

            # Read response from stdout with timeout
            try:
                response_line = await asyncio.wait_for(
                    server.process.stdout.readline(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print(f"Timeout waiting for tool list from '{server.name}'")
                server.tools = []
                return

            if not response_line:
                # Check stderr for errors if stdout is empty
                try:
                    error_output = await asyncio.wait_for(
                        server.process.stderr.read(1024),
                        timeout=1.0
                    )
                    print(f"MCP server '{server.name}' closed stdout unexpectedly. Stderr: {error_output.decode()}")
                except asyncio.TimeoutError:
                    pass
                server.tools = []
                return

            try:
                response = json.loads(response_line.decode())
                if "result" in response and "tools" in response["result"]:
                    server.tools = response["result"]["tools"]
                    print(f"Discovered {len(server.tools)} tools from '{server.name}'")
                elif "error" in response:
                    print(f"MCP tool discovery error from '{server.name}': {response['error']}")
                    server.tools = []
                else:
                    print(f"Unexpected MCP response from '{server.name}': {response}")
                    server.tools = []
            except json.JSONDecodeError as e:
                # Try to read some more from stdout/stderr to see what happened
                try:
                    remaining = await asyncio.wait_for(
                        server.process.stdout.read(1024),
                        timeout=1.0
                    )
                    stderr_output = await asyncio.wait_for(
                        server.process.stderr.read(1024),
                        timeout=1.0
                    )
                    print(f"Failed to parse JSON from '{server.name}': {e}")
                    print(f"Raw response line: {response_line.decode()}")
                    print(f"Remaining stdout: {remaining.decode()}")
                    print(f"Stderr: {stderr_output.decode()}")
                except asyncio.TimeoutError:
                    print(f"Timeout reading additional output from '{server.name}'")
                server.tools = []

        except Exception as e:
            print(f"Failed to discover tools from '{server.name}': {e}")
            server.tools = []
    
    async def _discover_tools_http(self, server: MCPServer):
        """Discover tools via SSE or StreamableHTTP transport - FastMCP compatible"""
        try:
            endpoint = server.url.rstrip('/')
            
            request = {
                "jsonrpc": "2.0",
                "id": server.get_next_request_id(),
                "method": "tools/list",
                "params": {}
            }
            
            if server.transport_type == "sse":
                # FastMCP SSE transport:
                # 1. GET /sse - establishes session, returns session_id via SSE
                # 2. POST /messages/?session_id=XXX - sends request (returns 202)
                # 3. Response comes back on THE SAME SSE stream (don't close it!)
                
                sse_endpoint = f"{endpoint}"
                messages_endpoint = f"{endpoint}/messages/"
                session_id = None
                
                print(f"[FastMCP] Step 1: Connecting to SSE endpoint: {sse_endpoint}")
                
                # Create a single session for all requests
                connector = aiohttp.TCPConnector()
                session = aiohttp.ClientSession(connector=connector)
                
                # Step 1: Get session_id from SSE and KEEP THE CONNECTION OPEN
                response = await session.get(sse_endpoint, timeout=aiohttp.ClientTimeout(total=10))
                print(f"[FastMCP] SSE status: {response.status}, Content-Type: {response.headers.get('Content-Type')}")
                
                if response.status == 200:
                    # Read SSE stream until we get session_id
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        print(f"[FastMCP] SSE: {line[:150]}")
                        if line.startswith('data: ') and 'session_id=' in line:
                            data = line[6:].strip()
                            session_id = data.split('session_id=')[1].split('&')[0]
                            messages_path = data.split('?')[0]
                            if messages_path.startswith('/'):
                                messages_endpoint = f"{endpoint[:-4] if endpoint.endswith('/sse') else endpoint}{messages_path}"
                            print(f"[FastMCP] Got session_id: {session_id}")
                            print(f"[FastMCP] Messages endpoint: {messages_endpoint}")
                            break
                    
                    if not session_id:
                        print("[FastMCP] Failed to get session_id")
                        await session.close()
                        server.tools = []
                        return
                    
                    # Step 2: Send initialize request first (MCP protocol requirement)
                    print(f"[FastMCP] Step 2: Sending initialize request...")
                    init_request = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "llm-ui", "version": "1.0"}
                        }
                    }
                    async with session.post(
                        messages_endpoint,
                        params={"session_id": session_id},
                        json=init_request,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as init_response:
                        print(f"[FastMCP] INIT POST status: {init_response.status}")
                    
                    # Listen for initialize response
                    print(f"[FastMCP] Waiting for initialize response...")
                    init_done = False
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        print(f"[FastMCP] SSE: {line[:150]}")
                        
                        if not line or line.startswith(':'):
                            continue
                            
                        if line.startswith('data: '):
                            data = line[6:].strip()
                            if not data:
                                continue
                            try:
                                result = json.loads(data)
                                if "result" in result:
                                    print(f"[FastMCP] Initialize successful")
                                    init_done = True
                                    break
                                elif "error" in result:
                                    print(f"[FastMCP] Init error: {result['error']}")
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    if not init_done:
                        print("[FastMCP] Initialization failed")
                        await session.close()
                        server.tools = []
                        return
                    
                    # Step 3: Now send tools/list request
                    print(f"[FastMCP] Step 3: Sending tools/list request...")
                    async with session.post(
                        messages_endpoint,
                        params={"session_id": session_id},
                        json=request,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as post_response:
                        print(f"[FastMCP] POST status: {post_response.status}")
                    
                    # Step 4: Listen for tools/list response on SAME SSE stream
                    print(f"[FastMCP] Step 4: Listening for tools response...")
                    try:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            print(f"[FastMCP] SSE response: {line[:150]}")
                            
                            if not line or line.startswith(':'):
                                continue
                                
                            if line.startswith('data: '):
                                data = line[6:].strip()
                                if not data:
                                    continue
                                try:
                                    result = json.loads(data)
                                    print(f"[FastMCP] Parsed: {list(result.keys())}")
                                    if "result" in result and "tools" in result.get("result", {}):
                                        server.tools = result["result"]["tools"]
                                        print(f"[FastMCP] SUCCESS: Discovered {len(server.tools)} tools")
                                        server.session_id = session_id
                                        await session.close()
                                        return
                                    elif "error" in result:
                                        print(f"[FastMCP] Error: {result['error']}")
                                        break
                                except json.JSONDecodeError as e:
                                    print(f"[FastMCP] JSON error: {e}")
                                    continue
                    except Exception as e:
                        print(f"[FastMCP] SSE listening error: {e}")
                    
                    await session.close()
                    server.tools = []
                    print("[FastMCP] No tools received")
                else:
                    await session.close()
                    server.tools = []
                    print("[FastMCP] SSE connection failed")
                
            else:
                # StreamableHTTP transport
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
            import traceback
            traceback.print_exc()
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
        request = {
            "jsonrpc": "2.0",
            "id": server.get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        request_json = json.dumps(request) + "\n"
        server.process.stdin.write(request_json.encode())
        await server.process.stdin.drain()

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
        base_url = server.url.rstrip('/')
        
        request = {
            "jsonrpc": "2.0",
            "id": server.get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        if server.transport_type == "sse":
            # Use the session_id from discovery
            if not server.session_id:
                raise Exception("No session_id available for SSE transport")
            
            messages_endpoint = f"{base_url}/messages/"
            
            connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(connector=connector) as session:
                # Send request
                async with session.post(
                    messages_endpoint,
                    params={"session_id": server.session_id},
                    json=request,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 202:
                        # Listen on SSE for response
                        sse_endpoint = f"{base_url}"
                        async with session.get(sse_endpoint, timeout=aiohttp.ClientTimeout(total=30)) as sse_response:
                            async for line in sse_response.content:
                                line = line.decode('utf-8').strip()
                                if not line or line.startswith(':'):
                                    continue
                                if line.startswith('data: '):
                                    data = line[6:].strip()
                                    if not data:
                                        continue
                                    result = json.loads(data)
                                    if "result" in result:
                                        return result["result"]
                                    elif "error" in result:
                                        raise Exception(f"MCP error: {result['error']}")
                    elif response.status == 200:
                        result = await response.json()
                        if "result" in result:
                            return result["result"]
                        elif "error" in result:
                            raise Exception(f"MCP error: {result['error']}")
            
            raise Exception("No response received")
        else:
            async with server.http_session.post(
                base_url,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result:
                        return result["result"]
                    elif "error" in result:
                        raise Exception(f"MCP error: {result['error']}")
                else:
                    raise Exception(f"HTTP error: {response.status}")
    
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
        async with get_db() as db:
            await db_add_mcp_server(db, name, command, args, env, transport_type, url)

        return await self._start_server(name, command, args, env, transport_type, url)
    
    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server"""
        if name not in self.servers:
            return False
        
        server = self.servers[name]
        
        if server.process:
            try:
                if server.process.returncode is None:
                    server.process.terminate()
                    await server.process.wait()
            except (ProcessLookupError, Exception) as e:
                print(f"Error terminating MCP server process: {e}")
        
        del self.servers[name]
        
        from database.crud import remove_mcp_server as db_remove_mcp_server
        async with get_db() as db:
            await db_remove_mcp_server(db, name)
        
        return True
    
    async def cleanup(self):
        """Clean up all server processes and HTTP sessions"""
        for server in self.servers.values():
            if server.transport_type == "stdio" and server.process:
                try:
                    if server.process.returncode is None:
                        server.process.terminate()
                        await server.process.wait()
                except (ProcessLookupError, Exception):
                    pass
            elif server.transport_type in ("sse", "streamable-http") and server.http_session:
                try:
                    await server.http_session.close()
                except Exception:
                    pass

        self.servers.clear()

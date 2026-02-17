import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import subprocess

from database.models import get_db
from database.crud import get_enabled_mcp_servers, add_mcp_server as db_add_mcp_server


@dataclass
class MCPServer:
    """Represents an MCP server connection"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    process: Optional[subprocess.Popen] = None
    tools: List[Dict] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


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
                config["command"],
                config["args"],
                config["env"]
            )
        
        self._initialized = True
    
    async def _start_server(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Dict[str, str]
    ) -> bool:
        """Start an MCP server process"""
        try:
            # Prepare environment - inherit from current process to ensure PATH is available
            import os
            process_env = os.environ.copy()
            process_env.update(env)
            
            # Start the MCP server process
            # Note: This is a simplified version. Real MCP communication uses stdio
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
                command=command,
                args=args,
                env=env,
                process=process
            )
            
            # Discover tools from the server
            await self._discover_tools(server)
            
            self.servers[name] = server
            return True
            
        except Exception as e:
            print(f"Failed to start MCP server '{name}': {e}")
            return False
    
    async def _discover_tools(self, server: MCPServer):
        """Discover available tools from an MCP server"""
        try:
            # Send tools/list request according to MCP protocol
            request = {
                "jsonrpc": "2.0",
                "id": 1,
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
        
        # Send tools/call request according to MCP protocol
        request = {
            "jsonrpc": "2.0",
            "id": 2,
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
                "command": server.command,
                "tool_count": len(server.tools)
            }
            for server in self.servers.values()
        ]
    
    async def add_server(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Dict[str, str]
    ) -> bool:
        """Add a new MCP server"""
        # Save to database
        async with get_db() as db:
            await db_add_mcp_server(db, name, command, args, env)
        
        # Start the server
        return await self._start_server(name, command, args, env)
    
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
        """Clean up all server processes"""
        for server in self.servers.values():
            if server.process:
                try:
                    if server.process.returncode is None:
                        server.process.terminate()
                        await server.process.wait()
                except (ProcessLookupError, Exception):
                    pass
        
        self.servers.clear()

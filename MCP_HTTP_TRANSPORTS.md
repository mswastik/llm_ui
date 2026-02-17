# MCP HTTP Transports (SSE & Streamable HTTP)

This document explains how to use MCP servers with HTTP-based transports (SSE and Streamable HTTP) in addition to the existing stdio transport.

## Overview

The application now supports three MCP transport types:

1. **Stdio** - Traditional MCP transport using stdin/stdout (e.g., `npx`, `uvx` commands)
2. **SSE** (Server-Sent Events) - HTTP-based transport using SSE for server-to-client streaming
3. **Streamable HTTP** - HTTP-based transport with streaming capabilities

## Adding an MCP Server via UI

### For SSE or Streamable HTTP:

1. Click the **"MCP Servers"** button in the sidebar
2. In the Settings modal, go to the **MCP Servers** tab
3. Fill in the form:
   - **Server Name**: A unique name for your server (e.g., `my-remote-server`)
   - **Transport Type**: Select either "SSE (Server-Sent Events)" or "Streamable HTTP"
   - **Server URL**: The full URL of the MCP server endpoint (e.g., `http://localhost:8080/mcp`)
4. Click **"Add Server"**

### For Stdio (existing behavior):

1. Click the **"MCP Servers"** button in the sidebar
2. In the Settings modal, go to the **MCP Servers** tab
3. Fill in the form:
   - **Server Name**: A unique name for your server (e.g., `filesystem`)
   - **Transport Type**: Keep "Stdio (npx/uvx command)" selected
   - **Command**: The executable command (e.g., `npx` or `uvx`)
   - **Arguments**: JSON array of arguments (e.g., `["-y", "@modelcontextprotocol/server-filesystem"]`)
4. Click **"Add Server"**

## Example Configurations

### SSE Transport Example

```json
{
  "name": "remote-mcp-server",
  "transport_type": "sse",
  "url": "http://localhost:8080/mcp"
}
```

### Streamable HTTP Transport Example

```json
{
  "name": "streaming-mcp-server",
  "transport_type": "streamable-http",
  "url": "http://localhost:3000/mcp"
}
```

### Stdio Transport Example (existing)

```json
{
  "name": "filesystem",
  "transport_type": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
}
```

## How It Works

### HTTP-based Transports (SSE & Streamable HTTP)

For HTTP transports, the application:

1. Creates an HTTP session to the specified URL
2. Sends JSON-RPC requests via POST with `Content-Type: application/json`
3. Expects JSON-RPC 2.0 compliant responses

**Request Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Tool Call Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {"arg1": "value1"}
  }
}
```

### Stdio Transport

For stdio transport, the application:

1. Spawns a subprocess with the specified command and arguments
2. Communicates via stdin/stdout using JSON-RPC
3. Manages the process lifecycle (start/stop)

## API Endpoints

### List MCP Servers
```
GET /api/mcp/servers
```

Response:
```json
{
  "servers": [
    {
      "name": "my-server",
      "transport_type": "sse",
      "command": null,
      "url": "http://localhost:8080/mcp",
      "tool_count": 5
    }
  ]
}
```

### Add MCP Server
```
POST /api/mcp/servers
```

Request body:
```json
{
  "name": "my-server",
  "transport_type": "sse",
  "url": "http://localhost:8080/mcp",
  "command": null,
  "args": [],
  "env": {}
}
```

### Remove MCP Server
```
DELETE /api/mcp/servers/{server_name}
```

## Database Schema

The `mcp_servers` table has been updated with the following columns:

- `transport_type` (TEXT): The transport type (`stdio`, `sse`, or `streamable-http`)
- `url` (TEXT): The server URL for HTTP-based transports
- `command` (TEXT): The command for stdio transport (nullable for HTTP transports)

## Migration

If you have an existing database, run the migration script:

```bash
cd backend
python database/migrate.py
```

This will add the new columns to your existing `mcp_servers` table.

## Troubleshooting

### HTTP Transport Issues

1. **Connection refused**: Ensure the MCP server is running and accessible at the specified URL
2. **CORS errors**: The MCP server should allow CORS if hosted on a different origin
3. **404 errors**: Verify the URL endpoint is correct (e.g., `/mcp`, `/messages`, etc.)

### Stdio Transport Issues

1. **Command not found**: Ensure `npx`, `uvx`, or the specified command is installed
2. **Process exits immediately**: Check stderr output in the server logs
3. **Tool discovery fails**: Verify the MCP server implements the `tools/list` method

## Best Practices

1. **Use descriptive server names** to easily identify them in the UI
2. **Test connectivity** before relying on the server for important tasks
3. **Monitor server logs** for debugging connection or tool execution issues
4. **Use HTTPS** for remote MCP servers in production environments
5. **Validate URLs** to ensure they point to trusted MCP servers

## Security Considerations

- Only connect to MCP servers from trusted sources
- For remote servers, use HTTPS to encrypt communications
- Be cautious with MCP servers that have file system or network access
- Review tool permissions and capabilities before using them

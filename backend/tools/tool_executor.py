"""
Tool executor with real-time progress updates.

Wraps MCP tool calls and custom tools to provide streaming progress
updates to the UI via Server-Sent Events.
"""
import asyncio
import json
import logging
from typing import Dict, Any, AsyncGenerator, List

from tools.rag_service import RAGService, RAG_TOOL_DEFINITION
from tools.tts_service import TTSService, TTSConfig, TTS_TOOL_DEFINITION
from tools.base import current_document_ids
from settings import settings_manager

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools with real-time progress updates."""

    def __init__(self, mcp_manager=None):
        self.mcp_manager = mcp_manager
        self.rag_service = RAGService()
        # Create TTS service from saved settings (engine, voice, kokoro device, ...)
        self.tts_service = TTSService(TTSConfig.from_settings(settings_manager.get_settings()))

    def get_tool_definitions(
        self,
        exclude_tools: List[str] = None,
        mcp_tools: List[Dict] = None,
        enable_rag: bool = False
    ) -> List[Dict]:
        exclude_tools = exclude_tools or []
        tools = []

        if enable_rag:
            sections = self.rag_service.store.list_sections()
            rag_def = json.loads(json.dumps(RAG_TOOL_DEFINITION))
            if sections:
                listing = "\n".join(f"- {s}" for s in sections)
                rag_def["function"]["description"] += (
                    "\n\nAvailable sections in the documents (use the exact title "
                    "for the 'section' parameter):\n" + listing)
            tools.append(rag_def)
        if TTS_TOOL_DEFINITION.get("function", {}).get("name") not in exclude_tools:
            tools.append(TTS_TOOL_DEFINITION)

        if mcp_tools:
            for tool in mcp_tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("inputSchema", {"type": "object", "properties": {}}))
                    }
                }
                tools.append(openai_tool)

        return tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], request_id: str) -> AsyncGenerator[Dict, None]:
        """Execute a tool and yield progress updates."""
        try:
            if tool_name == "query_documents":
                async for p in self._query_documents_with_progress(arguments, request_id):
                    yield p
            elif tool_name == "generate_speech":
                async for p in self._generate_speech_with_progress(arguments, request_id):
                    yield p
            else:
                # MCP tool execution
                yield {"type": "tool_progress", "tool": tool_name, "status": f"Starting {tool_name}...", "progress": 0}

                if ":" in tool_name:
                    server_name, actual_tool_name = tool_name.split(":", 1)
                else:
                    server_name = await self._find_tool_server(tool_name)
                    actual_tool_name = tool_name

                yield {"type": "tool_progress", "tool": tool_name, "status": f"Calling {server_name}:{actual_tool_name}...", "progress": 25}

                if self.mcp_manager:
                    result = await self.mcp_manager.call_tool(server_name, actual_tool_name, arguments)
                else:
                    result = {"error": "MCP manager not available"}

                yield {"type": "tool_progress", "tool": tool_name, "status": "Processing result...", "progress": 75}
                yield {"type": "tool_progress", "tool": tool_name, "status": "Complete", "progress": 100, "result": result}
        except Exception as e:
            logger.error(f"{tool_name} failed: {e}")
            import traceback; traceback.print_exc()
            yield {"type": "tool_error", "tool": tool_name, "error": str(e)}

    async def _find_tool_server(self, tool_name: str) -> str:
        if self.mcp_manager:
            all_tools = await self.mcp_manager.list_all_tools()
            for tool in all_tools:
                if tool.get("name") == tool_name:
                    return tool.get("server", "unknown")
                if ":" in tool_name:
                    return tool_name.split(":")[0]
        raise ValueError(f"Tool '{tool_name}' not found in any MCP server")

    # --- Custom tool implementations ---

    async def _query_documents_with_progress(self, arguments: Dict[str, Any], request_id: str) -> AsyncGenerator[Dict, None]:
        query = arguments.get("query", "")
        document_ids = arguments.get("document_ids") or current_document_ids.get()
        section = arguments.get("section")
        top_k = arguments.get("top_k", 10)

        if not query:
            yield {"type": "tool_error", "tool": "query_documents", "error": "Query is required"}
            return

        yield {"type": "tool_progress", "tool": "query_documents", "status": "Searching documents...", "progress": 0}

        result = await self.rag_service.query(query=query, document_ids=document_ids, top_k=top_k,
                                              section=section, progress_callback=None)

        if "error" in result:
            yield {"type": "tool_error", "tool": "query_documents", "error": result["error"]}
            return

        formatted_result = {
            "query": query,
            "results": result.get("results", []),
            "context": result.get("context", "No relevant content found"),
            "result_count": len(result.get("results", [])),
            "sources": result.get("sources", [])
        }

        yield {"type": "tool_progress", "tool": "query_documents", "status": f"Found {formatted_result['result_count']} relevant passages",
               "progress": 100, "result": formatted_result}

    async def _generate_speech_with_progress(self, arguments: Dict[str, Any], request_id: str) -> AsyncGenerator[Dict, None]:
        text = arguments.get("text", "")
        voice = arguments.get("voice")

        if not text:
            yield {"type": "tool_error", "tool": "generate_speech", "error": "Text is required"}
            return

        yield {"type": "tool_progress", "tool": "generate_speech", "status": "Generating speech...", "progress": 0}

        result = await self.tts_service.generate_speech(text=text, voice=voice)

        yield {"type": "tool_progress", "tool": "generate_speech", "status": "Speech generated", "progress": 100, "result": result}

    async def process_document_for_rag(self, document_id: str, filepath: str, file_type: str, progress_callback=None) -> Dict:
        return await self.rag_service.process_document(document_id=document_id, filepath=filepath, file_type=file_type, progress_callback=progress_callback)

    async def delete_document_from_rag(self, document_id: str):
        await self.rag_service.delete_document(document_id)

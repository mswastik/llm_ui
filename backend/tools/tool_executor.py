import asyncio
import json
from typing import Dict, Any, AsyncGenerator, Callable, List
from datetime import datetime

from tools.searxng_tool import SearXNGSearchTool, SearchConfig, SEARXNG_TOOL_DEFINITION
from tools.rag_service import RAGService, RAGConfig, RAG_TOOL_DEFINITION


class ToolExecutor:
    """
    Executes tools with real-time progress updates.
    
    This class wraps MCP tool calls and custom tools to provide
    streaming progress updates to the UI via Server-Sent Events.
    """
    
    def __init__(self, mcp_manager=None):
        self.mcp_manager = mcp_manager
        
        # Initialize SearXNG search tool
        self.search_tool = SearXNGSearchTool()
        
        # Initialize RAG service
        self.rag_service = RAGService()
        
        # Register custom tools that need progress tracking
        self.custom_tools = {
            "search_web": self._search_web_with_progress,
            "query_documents": self._query_documents_with_progress,
        }
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get all tool definitions for LLM function calling.
        
        Returns a list of tool definitions in OpenAI format.
        """
        tools = [
            SEARXNG_TOOL_DEFINITION,
            RAG_TOOL_DEFINITION,
        ]
        
        # Add MCP tools if available
        if self.mcp_manager:
            # MCP tools would be added here
            pass
        
        return tools
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        request_id: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Execute a tool and yield progress updates.
        
        Yields:
            Dict with structure:
            {
                "type": "tool_progress",
                "tool": tool_name,
                "status": "status message",
                "progress": 0-100 (optional),
                "data": {} (optional intermediate data)
            }
        """
        
        # Check if this is a custom tool with progress tracking
        if tool_name in self.custom_tools:
            async for progress in self.custom_tools[tool_name](arguments, request_id):
                yield progress
        else:
            # Standard MCP tool - no intermediate progress
            yield {
                "type": "tool_progress",
                "tool": tool_name,
                "status": f"Executing {tool_name}...",
                "progress": 0
            }
            
            try:
                # Parse server name from tool name if formatted as "server:tool"
                if ":" in tool_name:
                    server_name, actual_tool_name = tool_name.split(":", 1)
                else:
                    # Try to find which server has this tool
                    server_name = await self._find_tool_server(tool_name)
                    actual_tool_name = tool_name
                
                # Call MCP tool
                if self.mcp_manager:
                    result = await self.mcp_manager.call_tool(
                        server_name,
                        actual_tool_name,
                        arguments
                    )
                else:
                    result = {"error": "MCP manager not available"}
                
                yield {
                    "type": "tool_progress",
                    "tool": tool_name,
                    "status": "Complete",
                    "progress": 100,
                    "result": result
                }
                
            except Exception as e:
                yield {
                    "type": "tool_error",
                    "tool": tool_name,
                    "error": str(e)
                }
    
    async def _find_tool_server(self, tool_name: str) -> str:
        """Find which MCP server provides a tool"""
        if self.mcp_manager:
            all_tools = await self.mcp_manager.list_all_tools()
            for tool in all_tools:
                if tool["name"] == tool_name:
                    return tool["server"]
        raise ValueError(f"Tool '{tool_name}' not found in any MCP server")
    
    # Custom tool implementations with progress tracking
    
    async def _search_web_with_progress(
        self,
        arguments: Dict[str, Any],
        request_id: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Web search tool with detailed progress updates.
        
        Uses SearXNG for search, with semantic reranking.
        """
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 15)
        
        if not query:
            yield {
                "type": "tool_error",
                "tool": "search_web",
                "error": "Query is required"
            }
            return
        
        # Progress callback to yield updates
        progress_values = [0]  # Use list to allow modification in nested function
        
        def progress_callback(status: str, progress: int):
            progress_values[0] = progress
        
        # Yield initial status
        yield {
            "type": "tool_progress",
            "tool": "search_web",
            "status": "Starting web search...",
            "progress": 5
        }
        
        try:
            # Execute search
            result = await self.search_tool.search(
                query=query,
                max_results=max_results,
                top_k=max_results,
                progress_callback=progress_callback
            )
            
            # Yield progress updates during search
            # The search_tool handles the actual search, we yield final result
            yield {
                "type": "tool_progress",
                "tool": "search_web",
                "status": "Processing search results...",
                "progress": 90
            }
            
            # Check for errors
            if "error" in result:
                yield {
                    "type": "tool_error",
                    "tool": "search_web",
                    "error": result["error"]
                }
                return
            
            # Format result for LLM consumption
            formatted_result = {
                "query": query,
                "sources": result.get("sources", []),
                "content": result.get("content", "No results found"),
                "chunk_count": len(result.get("chunks", []))
            }
            
            # Final result
            yield {
                "type": "tool_progress",
                "tool": "search_web",
                "status": "Search complete",
                "progress": 100,
                "result": formatted_result
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {
                "type": "tool_error",
                "tool": "search_web",
                "error": f"Search failed: {str(e)}"
            }
    
    async def _query_documents_with_progress(
        self,
        arguments: Dict[str, Any],
        request_id: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Document query tool with detailed progress updates.
        
        Uses RAG to search through uploaded documents.
        """
        query = arguments.get("query", "")
        document_ids = arguments.get("document_ids")
        top_k = arguments.get("top_k", 10)
        
        if not query:
            yield {
                "type": "tool_error",
                "tool": "query_documents",
                "error": "Query is required"
            }
            return
        
        yield {
            "type": "tool_progress",
            "tool": "query_documents",
            "status": "Searching documents...",
            "progress": 10
        }
        
        try:
            # Progress callback
            def progress_callback(status: str, progress: int):
                pass  # We'll yield our own progress
            
            # Execute query
            result = await self.rag_service.query(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                progress_callback=progress_callback
            )
            
            yield {
                "type": "tool_progress",
                "tool": "query_documents",
                "status": "Processing results...",
                "progress": 80
            }
            
            # Check for errors
            if "error" in result:
                yield {
                    "type": "tool_error",
                    "tool": "query_documents",
                    "error": result["error"]
                }
                return
            
            # Format result
            formatted_result = {
                "query": query,
                "results": result.get("results", []),
                "context": result.get("context", "No relevant content found"),
                "result_count": len(result.get("results", []))
            }
            
            yield {
                "type": "tool_progress",
                "tool": "query_documents",
                "status": f"Found {formatted_result['result_count']} relevant passages",
                "progress": 100,
                "result": formatted_result
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {
                "type": "tool_error",
                "tool": "query_documents",
                "error": f"Document query failed: {str(e)}"
            }
    
    async def process_document_for_rag(
        self,
        document_id: str,
        filepath: str,
        file_type: str,
        progress_callback: Callable[[str, int], None] = None
    ) -> Dict:
        """
        Process a document for RAG queries.
        
        This should be called when a document is uploaded.
        """
        return await self.rag_service.process_document(
            document_id=document_id,
            filepath=filepath,
            file_type=file_type,
            progress_callback=progress_callback
        )
    
    def delete_document_from_rag(self, document_id: str):
        """
        Delete a document from the RAG index.
        
        This should be called when a document is deleted.
        """
        self.rag_service.delete_document(document_id)


# Tool definitions for export
TOOL_DEFINITIONS = [
    SEARXNG_TOOL_DEFINITION,
    RAG_TOOL_DEFINITION,
]

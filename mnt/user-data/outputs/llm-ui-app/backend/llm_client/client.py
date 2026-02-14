import aiohttp
import json
from typing import List, Dict, AsyncGenerator, Any


class LLMClient:
    """
    Client to interact with llama.cpp server.
    
    Assumes llama.cpp is running with OpenAI-compatible API
    at http://localhost:8080 (default llama.cpp port).
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.model = "llama"  # llama.cpp uses "llama" as model name
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream chat completion from llama.cpp.
        
        Yields:
            Dict with structure:
            {
                "type": "content",  # or "tool_call"
                "content": "text chunk",
                "tool_call": {...}  # if type is "tool_call"
            }
        """
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add tool definitions if your llama.cpp supports function calling
        # This depends on your model and llama.cpp version
        tools = await self._get_available_tools()
        if tools:
            payload["tools"] = tools
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if not line or line == "data: [DONE]":
                            continue
                        
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                
                                # Handle content
                                if "content" in delta and delta["content"]:
                                    yield {
                                        "type": "content",
                                        "content": delta["content"]
                                    }
                                
                                # Handle tool calls (if model supports it)
                                if "tool_calls" in delta:
                                    for tool_call in delta["tool_calls"]:
                                        yield {
                                            "type": "tool_call",
                                            "tool_call": {
                                                "name": tool_call["function"]["name"],
                                                "arguments": json.loads(
                                                    tool_call["function"]["arguments"]
                                                )
                                            }
                                        }
                            
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            print(f"Error in LLM streaming: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def _get_available_tools(self) -> List[Dict]:
        """
        Get tool definitions to send to the LLM.
        
        This would integrate with your MCP manager to get all available tools.
        For now, returns an empty list - implement based on your needs.
        """
        # TODO: Integrate with MCPClientManager to get tool definitions
        # Convert MCP tool schemas to OpenAI function calling format
        return []
    
    async def generate_title(self, first_message: str) -> str:
        """
        Generate a conversation title from the first message.
        """
        title_prompt = f"Generate a short, 3-5 word title for a conversation that starts with: '{first_message[:100]}'. Respond with ONLY the title, nothing else."
        
        messages = [{"role": "user", "content": title_prompt}]
        
        title = ""
        async for chunk in self.stream_chat(messages, temperature=0.5, max_tokens=20):
            if chunk.get("type") == "content":
                title += chunk.get("content", "")
        
        return title.strip() or "New Chat"
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Non-streaming completion (collects full response).
        """
        response = ""
        async for chunk in self.stream_chat(messages, temperature, max_tokens):
            if chunk.get("type") == "content":
                response += chunk.get("content", "")
        
        return response

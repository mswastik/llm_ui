import aiohttp
import asyncio
import json
from typing import List, Dict, AsyncGenerator, Any, Optional

from config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class LLMClient:
    """
    Client to interact with llama.cpp server.
    
    Assumes llama.cpp is running with OpenAI-compatible API
    at http://localhost:8080 (default llama.cpp port).
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        # Import settings manager here to avoid circular imports
        from backend.settings import settings_manager
        self._settings_manager = settings_manager
        
        # Use provided values or get from settings
        settings = self._settings_manager.get_settings()
        self.base_url = base_url or settings.get('llama_cpp_base_url', 'http://localhost:8080')
        self.model = model or settings.get('llama_cpp_model', 'glm4.7-30ba3b')
        self._tools: Optional[List[Dict]] = None
    
    def set_tools(self, tools: List[Dict]):
        """Set the tools available for function calling"""
        self._tools = tools
    
    def _get_current_base_url(self) -> str:
        """Get the current base URL from settings (allows dynamic updates)"""
        settings = self._settings_manager.get_settings()
        return settings.get('llama_cpp_base_url', self.base_url)
    
    def _get_current_model(self) -> str:
        """Get the current model from settings (allows dynamic updates)"""
        settings = self._settings_manager.get_settings()
        return settings.get('llama_cpp_model', self.model)
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        tools: List[Dict] = None,
        model: str = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream chat completion from llama.cpp.
        
        Yields:
            Dict with structure:
            {
                "type": "content",  # or "tool_call" or "thinking"
                "content": "text chunk",
                "tool_call": {...}  # if type is "tool_call"
            }
        """
        
        # Use provided model or fall back to default
        active_model = model or self.model
        
        # Use provided values or fall back to defaults from config
        active_temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        active_max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": True,
            "temperature": active_temperature,
            "max_tokens": active_max_tokens,
        }
        
        # Add tool definitions if available
        active_tools = tools or self._tools
        if active_tools:
            payload["tools"] = active_tools
        
        try:
            # Increased timeout for long-running requests with web search context
            timeout = aiohttp.ClientTimeout(total=600, sock_connect=30, sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"llama.cpp returned status {response.status}: {error_text}")
                    
                    # Stream content more immediately
                    buffer = ""
                    
                    # Track streaming tool call - accumulate arguments across chunks
                    streaming_tool_call = None
                    
                    async for chunk in response.content.iter_any():
                        # Decode chunk and add to buffer
                        text = chunk.decode('utf-8')
                        buffer += text
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line or line == "data: [DONE]":
                                continue
                            
                            if line.startswith("data: "):
                                data = line[6:]  # Remove "data: " prefix
                                
                                try:
                                    chunk_data = json.loads(data)
                                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                    
                                    # Handle thinking content (for thinking models like DeepSeek)
                                    # Check multiple field names that llama.cpp might use
                                    thinking_content = delta.get("thinking") or delta.get("reasoning_content")
                                    if thinking_content:
                                        yield {
                                            "type": "thinking",
                                            "content": thinking_content
                                        }
                                        await asyncio.sleep(0)
                                    
                                    # Handle content
                                    if "content" in delta and delta["content"]:
                                        yield {
                                            "type": "content",
                                            "content": delta["content"]
                                        }
                                        # Small yield to allow event loop to process
                                        await asyncio.sleep(0)
                                    
                                    # Handle tool calls (if model supports it)
                                    # llama.cpp streams tool calls incrementally
                                    if "tool_calls" in delta:
                                        for tool_call in delta["tool_calls"]:
                                            try:
                                                # Safely extract tool call data
                                                function_data = tool_call.get("function", {})
                                                if not function_data:
                                                    print(f"[DEBUG] tool_call missing 'function' key: {tool_call}")
                                                    continue
                                                
                                                tool_name = function_data.get("name")
                                                arguments_str = function_data.get("arguments", "")
                                                
                                                # If we have a tool name, this is the start of a new tool call
                                                if tool_name:
                                                    streaming_tool_call = {
                                                        "name": tool_name,
                                                        "arguments_str": arguments_str
                                                    }
                                                    print(f"[DEBUG] Starting tool call: {tool_name}")
                                                
                                                # If we already have a streaming tool call, accumulate arguments
                                                elif streaming_tool_call and arguments_str:
                                                    streaming_tool_call["arguments_str"] += arguments_str
                                                    print(f"[DEBUG] Accumulated arguments: {streaming_tool_call['arguments_str'][:100]}...")
                                                
                                                # Try to parse accumulated arguments if we have a tool name
                                                if streaming_tool_call and streaming_tool_call.get("name"):
                                                    try:
                                                        parsed_args = json.loads(streaming_tool_call["arguments_str"])
                                                        print(f"[DEBUG] Successfully parsed tool arguments: {parsed_args}")
                                                        
                                                        # Yield complete tool call
                                                        yield {
                                                            "type": "tool_call",
                                                            "tool_call": {
                                                                "name": streaming_tool_call["name"],
                                                                "arguments": parsed_args
                                                            }
                                                        }
                                                        
                                                        # Clear streaming tool call after yielding
                                                        streaming_tool_call = None
                                                        await asyncio.sleep(0)
                                                        
                                                    except json.JSONDecodeError:
                                                        # Arguments not complete yet, wait for more chunks
                                                        print(f"[DEBUG] Arguments not complete yet, waiting...")
                                                        continue
                                                    
                                            except Exception as e:
                                                print(f"[DEBUG] Error processing tool_call: {e}, tool_call data: {tool_call}")
                                                import traceback
                                                traceback.print_exc()
                                                # Continue processing other tool calls
                                
                                except json.JSONDecodeError:
                                    continue
        
        except asyncio.CancelledError:
            # Client cancelled the request
            raise
        except aiohttp.ClientConnectorError:
            yield {
                "type": "error",
                "error": f"Cannot connect to llama.cpp at {self.base_url}. Make sure it's running."
            }
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "error": "Request to llama.cpp timed out"
            }
        except Exception as e:
            print(f"Error in LLM streaming: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def _get_available_tools(self) -> List[Dict]:
        """
        Get tool definitions to send to the LLM.
        
        Returns the tools that were set via set_tools().
        """
        return self._tools or []
    
    async def generate_title(self, first_message: str, model: str = None) -> str:
        """
        Generate a conversation title from the first message.
        """
        title_prompt = f"Generate a short, 3-5 word title for a conversation that starts with: '{first_message[:100]}'. Respond with ONLY the title, nothing else."
        
        messages = [{"role": "user", "content": title_prompt}]
        
        print(f"Generating title with model: {model or self.model}")
        title = ""
        try:
            async for chunk in self.stream_chat(messages, temperature=0.5, max_tokens=20, model=model):
                if chunk.get("type") == "content":
                    title += chunk.get("content", "")
                elif chunk.get("type") == "error":
                    print(f"Error in title generation: {chunk.get('error')}")
                    return first_message[:50].strip() or "New Chat"
        except Exception as e:
            print(f"Exception in title generation: {e}")
            return first_message[:50].strip() or "New Chat"
        
        # Clean up the title - remove quotes, newlines, and extra whitespace
        title = title.strip().strip('"\'').replace('\n', ' ').strip()
        
        print(f"Generated title: '{title}'")
        return title or first_message[:50].strip() or "New Chat"
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 10048
    ) -> str:
        """
        Non-streaming completion (collects full response).
        """
        response = ""
        async for chunk in self.stream_chat(messages, temperature, max_tokens):
            if chunk.get("type") == "content":
                response += chunk.get("content", "")
        
        return response

    async def list_models(self) -> List[Dict]:
        """List all available models from the LLM server."""
        try:
            timeout = aiohttp.ClientTimeout(total=50)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Handle different base URL formats
                # If base_url already contains /v1 or /v3, don't add /v1 again
                base = self.base_url #.rstrip('/')
                print(base)
                if '/v1' in base or '/v3' in base:
                    models_url = f"{base}/models"
                else:
                    models_url = f"{base}/v1/models"
                
                async with session.get(
                    models_url,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])
                        return [
                            {
                                "id": m.get("id", ""),
                                "name": m.get("id", m.get("id", "Unknown")),
                                "owned_by": m.get("owned_by", "unknown")
                            }
                            for m in models
                        ]
                    else:
                        print(f"Failed to fetch models: {response.status}")
                        return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

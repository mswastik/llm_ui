import aiohttp
import asyncio
import json
from typing import List, Dict, AsyncGenerator, Any, Optional

from settings import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


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
        model: str = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        chat_template_kwargs: dict = None,
        tool_choice: str = None,
        base_url: str = None,
        api_key: str = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream chat completion from llama.cpp with retry logic.

        Yields:
            Dict with structure:
            {
                "type": "content",  # or "tool_call" or "thinking"
                "content": "text chunk",
                "tool_call": {...}  # if type is "tool_call"
            }
        """

        # Use provided model or get current from settings (allows dynamic updates)
        active_model = model or self._get_current_model()

        # Use provided values or fall back to defaults from settings
        # Get current settings for temperature and max tokens (allows dynamic updates)
        settings = self._settings_manager.get_settings()
        active_temperature = temperature if temperature is not None else settings.get('default_temperature', DEFAULT_TEMPERATURE)
        active_max_tokens = max_tokens if max_tokens is not None else settings.get('default_max_tokens', DEFAULT_MAX_TOKENS)

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": True,
            "temperature": active_temperature,
            "max_tokens": active_max_tokens,
            "cache_prompt": True,  # Reuse KV cache from previous request when prompt prefix matches
        }

        # Add chat_template_kwargs if provided (for thinking suppression on models that support it)
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        # Add tool definitions if available
        active_tools = tools
        if active_tools:
            payload["tools"] = active_tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        # Retry logic for transient server errors
        last_error = None
        request_completed = False  # Track if request completed successfully

        for attempt in range(max_retries):
            if request_completed:
                break  # Don't retry if already completed successfully

            try:
                # Get current base URL from settings (allows dynamic updates)
                current_base_url = base_url or self._get_current_base_url()
                request_headers = {"Content-Type": "application/json"}
                if api_key:
                    request_headers["Authorization"] = f"Bearer {api_key}"
                
                # Increased timeout for long-running requests with web search context
                timeout = aiohttp.ClientTimeout(total=1200, sock_connect=100, sock_read=420)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    print(f"[DEBUG] Starting LLM request (attempt {attempt + 1}/{max_retries})")
                    async with session.post(
                        f"{current_base_url}/v1/chat/completions",
                        json=payload,
                        headers=request_headers,
                        timeout=360
                    ) as response:
                        if response.status == 500:
                            # Server error - likely temporary overload, retry
                            error_text = await response.text()
                            raise Exception(f"llama.cpp returned status {response.status}: {error_text}")
                        elif response.status != 200:
                            # Other errors - don't retry
                            error_text = await response.text()
                            raise Exception(f"llama.cpp returned status {response.status}: {error_text}")

                        print(f"[DEBUG] LLM request successful, streaming response")
                        
                        # Stream content more immediately
                        buffer = ""

                        # Track streaming tool call - accumulate arguments across chunks
                        streaming_tool_call = None

                        # Track thinking buffer for tags that span multiple chunks
                        thinking_buffer = None

                        # Track whether native thinking field has been seen (persistent across chunks)
                        _skip_think_yield = False

                        async for chunk in response.content.iter_any():
                            # Decode chunk and add to buffer
                            text = chunk.decode('utf-8')
                            # Only log first 100 chars to reduce log spam
                            #if len(buffer) < 100 or len(text.strip()) > 10:
                            #    print(f"[DEBUG] Raw chunk received: {repr(text[:100])}")
                            buffer += text

                            # Process complete lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()

                                if not line:
                                    continue
                                if line == "data: [DONE]":
                                    request_completed = True
                                    if streaming_tool_call and streaming_tool_call.get("name"):
                                        partial = streaming_tool_call["arguments_str"]
                                        print(f"[DEBUG] Incomplete tool call at [DONE]: {streaming_tool_call['name']}")
                                        yield {
                                            "type": "tool_error",
                                            "tool": streaming_tool_call["name"],
                                            "error": f"Tool call arguments incomplete (premature EOS). Partial: {partial[:200]}"
                                        }
                                        streaming_tool_call = None
                                    continue

                                if line.startswith("data: "):
                                    data = line[6:]  # Remove "data: " prefix

                                    try:
                                        chunk_data = json.loads(data)
                                        choices = chunk_data.get("choices") or [{}]
                                        delta = choices[0].get("delta", {})
                                        finish_reason = choices[0].get("finish_reason")

                                        # Handle thinking content (for thinking models like DeepSeek)
                                        # Check multiple field names that llama.cpp might use
                                        thinking_content = delta.get("thinking") or delta.get("reasoning_content")
                                        if thinking_content:
                                            _skip_think_yield = True

                                            # Detect tool calls mis-placed in reasoning_content.
                                            # llama.cpp can misidentify Qwen3 (especially with MTP) as a
                                            # thinking model, routing tool calls to reasoning_content
                                            # instead of delta.tool_calls. Check if the content looks
                                            # like a JSON tool call.
                                            stripped_tc = thinking_content.strip()
                                            if stripped_tc.startswith('{') and '"name"' in stripped_tc and '"arguments"' in stripped_tc:
                                                try:
                                                    tc_parsed = json.loads(stripped_tc)
                                                    if isinstance(tc_parsed, dict) and "name" in tc_parsed and "arguments" in tc_parsed:
                                                        tc_args = tc_parsed["arguments"] if isinstance(tc_parsed["arguments"], dict) else {}
                                                        yield {
                                                            "type": "tool_call",
                                                            "tool_call": {
                                                                "name": tc_parsed["name"],
                                                                "arguments": tc_args
                                                            }
                                                        }
                                                        await asyncio.sleep(0)
                                                        # Skip normal thinking yield for this chunk
                                                        # (continue to content processing below)
                                                except json.JSONDecodeError:
                                                    # Partial JSON spanning multiple chunks — treat as thinking for now
                                                    yield {
                                                        "type": "thinking",
                                                        "content": thinking_content
                                                    }
                                                    await asyncio.sleep(0)
                                            else:
                                                yield {
                                                    "type": "thinking",
                                                    "content": thinking_content
                                                }
                                                await asyncio.sleep(0)
                                        elif _skip_think_yield and thinking_buffer is None and finish_reason is None:
                                            # Native thinking field has stopped and we're not inside <think>.
                                            # Only clear flag when content has no thinking tags at all.
                                            content_check = delta.get("content", "")
                                            if content_check and '<think>' not in content_check and '</think>' not in content_check:
                                                _skip_think_yield = False
                                        elif not _skip_think_yield:
                                            _skip_think_yield = False

                                        # Handle content - parse for <think> tags with streaming
                                        if "content" in delta and delta["content"]:
                                            content = delta["content"]

                                            # Process content for <think> tags - stream thinking as it arrives
                                            while content:
                                                # Look for <think> start tag
                                                think_start = content.find('<think>')

                                                if think_start != -1:
                                                    # Yield content before thinking tag
                                                    before_think = content[:think_start]
                                                    if before_think:
                                                        yield {
                                                            "type": "content",
                                                            "content": before_think
                                                        }
                                                        await asyncio.sleep(0)

                                                    # Check if end tag is in the same chunk
                                                    after_start = content[think_start + 7:]  # Skip '<think>'
                                                    think_end = after_start.find('</think>')

                                                    if think_end != -1:
                                                        # Complete thinking block in same chunk
                                                        # Skip yield if native thinking field already provided it
                                                        if not _skip_think_yield:
                                                            thinking = after_start[:think_end]
                                                            yield {
                                                                "type": "thinking",
                                                                "content": thinking
                                                            }
                                                            await asyncio.sleep(0)
                                                        # Continue with remaining content
                                                        content = after_start[think_end + 9:]  # Skip '</think>'
                                                    else:
                                                        # Start streaming thinking - yield content as it arrives
                                                        thinking_buffer = after_start
                                                        if thinking_buffer and not _skip_think_yield:
                                                            yield {
                                                                "type": "thinking",
                                                                "content": thinking_buffer
                                                            }
                                                            await asyncio.sleep(0)
                                                        content = ''
                                                else:
                                                    # If we're in thinking mode, this is continuation of thinking
                                                    if thinking_buffer is not None:
                                                        thinking_buffer += content
                                                        if not _skip_think_yield:
                                                            yield {
                                                                "type": "thinking",
                                                                "content": content
                                                            }
                                                            await asyncio.sleep(0)
                                                        # Check if this chunk contains the end tag
                                                        end_pos = content.find('</think>')
                                                        if end_pos != -1:
                                                            thinking_buffer = None
                                                        content = ''
                                                    else:
                                                        # No thinking tag, yield as regular content
                                                        yield {
                                                            "type": "content",
                                                            "content": content
                                                        }
                                                        content = ''

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
                                                             pass

                                                except Exception as e:
                                                    print(f"[DEBUG] Error processing tool_call: {e}, tool_call data: {tool_call}")
                                                    import traceback
                                                    traceback.print_exc()
                                                    # Continue processing other tool calls

                                        # Handle finish_reason — stream end signal from the model
                                        if finish_reason:
                                            print(f"[DEBUG] Stream finish_reason: {finish_reason}")
                                            # MTP models can emit premature EOS mid-tool-call.
                                            # Instead of silently discarding, yield a tool_error
                                            # so the backend can report it and continue.
                                            if streaming_tool_call and streaming_tool_call.get("name"):
                                                partial = streaming_tool_call["arguments_str"]
                                                print(f"[DEBUG] Incomplete tool call at finish_reason: {streaming_tool_call['name']}")
                                                yield {
                                                    "type": "tool_error",
                                                    "tool": streaming_tool_call["name"],
                                                    "error": f"Tool call arguments incomplete (finish_reason: {finish_reason}). Partial: {partial[:200]}"
                                                }
                                                streaming_tool_call = None

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
                                        # If this is a 500 error and we have retries left, retry
                                        if "status 500" in str(e) and attempt < max_retries - 1:
                                            last_error = e
                                            print(f"Server error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                                            await asyncio.sleep(retry_delay)
                                            continue
                                        else:
                                            yield {
                                                "type": "error",
                                                "error": str(e)
                                            }
                                    # Successfully completed streaming - mark as completed
                                    request_completed = True
            except Exception as e:
                # Handle exceptions that occur before/during request setup
                # Only retry if request didn't complete successfully
                if not request_completed and "status 500" in str(e) and attempt < max_retries - 1:
                    last_error = e
                    print(f"Server error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    if not request_completed:
                        yield {
                            "type": "error",
                            "error": str(e)
                        }
                    break
    

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

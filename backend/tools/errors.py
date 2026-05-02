"""
Centralized error handling for tool execution.

Provides decorators to catch exceptions and return/yield standardized error responses.
"""
from functools import wraps
import traceback
import logging

logger = logging.getLogger(__name__)


def handle_tool_errors(func):
    """Decorator that catches exceptions and returns error dicts.
    
    Use for synchronous or coroutine functions that return results directly.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    return wrapper


def yield_tool_errors(func):
    """Decorator for async generators that catches exceptions and yields error events.
    
    Use for async generator functions that yield progress/result events.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            async for result in func(*args, **kwargs):
                yield result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            traceback.print_exc()
            yield {"type": "tool_error", "error": str(e)}
    return wrapper

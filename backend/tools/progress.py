"""
Standardized tool progress event generation.

Provides a helper class for yielding consistent tool_progress events
across all custom tool implementations.
"""


class ToolProgress:
    """Yields standardized tool progress events.

    Usage:
        progress = ToolProgress("my_tool")
        async for p in progress.start("Starting..."):
            yield p
        result = await do_work()
        async for p in progress.update("Complete", 100, result=result):
            yield p
    """
    
    def __init__(self, tool_name: str):
        self.tool = tool_name
    
    def start(self, status: str = "Starting..."):
        """Yield a start event."""
        yield {"type": "tool_progress", "tool": self.tool, "status": status, "progress": 0}
    
    def update(self, status: str, progress: int, result=None, data=None):
        """Yield an update event."""
        event = {"type": "tool_progress", "tool": self.tool, "status": status, "progress": progress}
        if result is not None:
            event["result"] = result
        if data is not None:
            event["data"] = data
        yield event
    
    def error(self, error: str):
        """Yield an error event."""
        yield {"type": "tool_error", "tool": self.tool, "error": error}

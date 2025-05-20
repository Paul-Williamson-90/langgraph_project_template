from langchain_core.tools import BaseTool

from src.tools.example import multiply

agent_tool_kit: list[BaseTool] = [multiply]

__all__ = [
    "agent_tool_kit",
]

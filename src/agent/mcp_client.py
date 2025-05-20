from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    MultiServerMCPClient,
)

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["src/mcp_server/tools.py"],
            "transport": "stdio",
        },
    }
)


async def get_tools():
    tools = await client.get_tools()
    return tools

import json

from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    MultiServerMCPClient,
)

servers = json.load(open("./mcp_config.json", "r"))
client = MultiServerMCPClient(servers)


async def get_tools():
    tools = await client.get_tools()
    return tools

# LangGraph Project Template Setup
A project template for starting a LangGraph Agent repo.

# Setup
1. Create a .env file in the root with the following keys:
```.env
LANGSMITH_API_KEY=...
OPENAI_API_KEY=...

POSTGRES_DB=postgres # change this to whatever
POSTGRES_USER=postgres # change this to whatever
POSTGRES_PASSWORD=postgres # change this to whatever

IMAGE_NAME=agent
```

## Deployment: Use the Makefile
1. Start the agent:
```bash
make agent_start
```

2. Stop the agent:
```bash
make agent_stop
```

3. Restart the agent:
```bash
make agent_restart
```

4. Access the LangGraph/LangSmith platform via a web browser [LangSmith](https://smith.langchain.com/)
5. Click 'Deployments' in the left hand menu
6. In the top right corner click LangGraph Studio and enter:

```
http://localhost:8123/
```

7. Now you can test your deployment and see the agent working visually.

## Endpoint
LangGraph has an SDK for interfacing with a LangGraph Agent endpoint:
- [Python SDK Reference](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)
- [Python SDK Guide](https://langchain-ai.github.io/langgraph/concepts/sdk/#installation)


For example:

```
from langgraph_sdk import get_client

client = get_client(url=..., api_key=...)
await client.assistants.search()
```

## Additional Information Regarding Deployment
- [Standalone Container Guidance](https://langchain-ai.github.io/langgraph/concepts/deployment_options/#standalone-container)
- [Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/langgraph_standalone_container/)


# Tools
The project template has been setup to allow both standard LangChain tools (BaseTool) and MCP-server enabled tools.
There are examples of both in the project template.

## Standard Tools
- See: src/tools/\_\_init\_\_.py
- Adding tools to the agent_tool_kit list will extend the tools available to the agent.

## MCP Tools
All MCP server connections are configured via the mcp_config.json in the root of the project directory.
By default there is the example local server setup that has been provided (src/mcp_servers/tools.py) using a stdio transport configuration.

You can also add additional remote servers via sse transport configurations.
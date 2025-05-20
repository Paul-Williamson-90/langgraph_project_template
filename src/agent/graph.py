import asyncio
import datetime
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel import RetryPolicy
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langgraph_sdk import get_client
from pydantic import BaseModel

from src.agent.config import Configuration
from src.agent.enums import InvocationTags
from src.agent.mcp_client import get_tools
from src.agent.state import InputState, OutputState, State
from src.agent.utils import format_memories
from src.tools import agent_tool_kit


async def init_node(
    state: State, config: RunnableConfig
) -> Command[Literal["chat_bot"]]:
    """Initial node in the graph.

    This can be used to initialize any state or configurations needed for the graph run.
    Optionally, a conditional edge logic via Command can be used in this node for forking
    the graph down different routes.
    """
    return Command(
        goto="chat_bot",
        update={"messages": state.messages[-1]},
    )


async def chat_bot(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    configuration = Configuration.from_runnable_config(config)
    namespace = (
        "memories",
        configuration.user_id,
    )

    query = "\n".join(
        str(message.content)
        for message in state.messages[-configuration.n_msgs_search :]
    )
    items = await store.asearch(
        namespace, query=query, limit=configuration.memories_limit
    )

    llm = init_chat_model(
        model=configuration.model,
        model_provider=configuration.model_provider,
        tags=[InvocationTags.MODEL_CALL.value],
    )

    mcp_tools = await get_tools()
    llm_with_tools = llm.bind_tools(agent_tool_kit + mcp_tools)

    sys = configuration.system_prompt.format(
        user_info=format_memories(items),
        time=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )
    max_tokens = configuration.max_tokens

    msgs = trim_messages(
        state.messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        include_system=True,
    )
    msg = await llm_with_tools.ainvoke(
        [{"role": "system", "content": sys}, *msgs],
    )

    if not isinstance(msg, AIMessage):
        raise ValueError("The model did not return an AIMessage.")

    return {"messages": [msg]}


def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "schedule_memories"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "schedule_memories"


async def output_node(state: State, config: RunnableConfig) -> dict:
    """Output node in the graph.

    This can be used to process the output of the graph run, such as saving the results
    or sending them to a different service.
    """
    return {"message": state.messages[-1]}


async def schedule_memories(state: State, config: RunnableConfig) -> None:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    configurable = Configuration.from_runnable_config(config)
    memory_client = get_client()
    await memory_client.runs.create(
        # We enqueue the memory formation process on the same thread.
        # This means that IF this thread doesn't receive more messages before `after_seconds`,
        # it will read from the shared state and extract memories for us.
        # If a new request comes in for this thread before the scheduled run is executed,
        # that run will be canceled, and a **new** one will be scheduled once
        # this node is executed again.
        thread_id=config["configurable"]["thread_id"],
        # This memory-formation run will be enqueued and run later
        # If a new run comes in before it is scheduled, it will be cancelled,
        # then when this node is executed again, a *new* run will be scheduled
        multitask_strategy="enqueue",
        # This lets us "debounce" repeated requests to the memory graph
        # if the user is actively engaging in a conversation. This saves us $$ and
        # can help reduce the occurrence of duplicate memories.
        after_seconds=configurable.delay_seconds,
        # Specify the graph and/or graph configuration to handle the memory processing
        assistant_id=configurable.mem_assistant_id,
        input={"messages": state.messages},
        config={
            "configurable": {
                # Ensure the memory service knows where to save the extracted memories
                "user_id": configurable.user_id,
                "memory_types": configurable.memory_types,
                "model": f"{configurable.model_provider}:{configurable.model}",
            },
        },
    )


def create_graph() -> CompiledStateGraph:
    # init graph
    graph_builder = StateGraph(
        State, input=InputState, output=OutputState, config_schema=Configuration
    )

    # get mcp tools
    mcp_tools = asyncio.run(get_tools())

    # add nodes
    graph_builder.add_node("init_node", init_node, retry=RetryPolicy())
    graph_builder.add_node("chat_bot", chat_bot, retry=RetryPolicy())
    graph_builder.add_node(
        ToolNode(
            agent_tool_kit + mcp_tools,
            tags=[InvocationTags.TOOL_CALLS.value],
            name="tools",
            messages_key="messages",
        ),
    )
    graph_builder.add_node("schedule_memories", schedule_memories, retry=RetryPolicy())
    graph_builder.add_node("output_node", output_node, retry=RetryPolicy())

    # add edges
    graph_builder.add_conditional_edges(
        "chat_bot",
        tools_condition,
        {"tools": "tools", "schedule_memories": "schedule_memories"},
    )
    graph_builder.add_edge("tools", "chat_bot")
    graph_builder.add_edge("schedule_memories", "output_node")
    graph_builder.add_edge("output_node", "__end__")

    # set entry point
    graph_builder.set_entry_point("init_node")

    # compile graph
    graph = graph_builder.compile()

    return graph

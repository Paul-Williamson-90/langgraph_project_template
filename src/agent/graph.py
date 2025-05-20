from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel import RetryPolicy
from langgraph.types import Command
from pydantic import BaseModel

from src.agent.config import Configuration
from src.agent.enums import InvocationTags
from src.agent.state import InputState, OutputState, State
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


async def chat_bot(state: State, config: RunnableConfig) -> dict:
    configuration = Configuration.from_runnable_config(config)

    llm = init_chat_model(
        model=configuration.model,
        model_provider=configuration.model_provider,
        tags=[InvocationTags.MODEL_CALL.value],
    )

    llm_with_tools = llm.bind_tools(agent_tool_kit)

    sys = configuration.system_prompt
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
) -> Literal["tools", "output_node"]:
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
    return "output_node"


async def output_node(state: State, config: RunnableConfig) -> dict:
    """Output node in the graph.

    This can be used to process the output of the graph run, such as saving the results
    or sending them to a different service.
    """
    return {"message": state.messages[-1]}


def create_graph() -> CompiledStateGraph:
    # init graph
    graph_builder = StateGraph(
        State, input=InputState, output=OutputState, config_schema=Configuration
    )

    # add nodes
    graph_builder.add_node("init_node", init_node, retry=RetryPolicy())
    graph_builder.add_node("chat_bot", chat_bot, retry=RetryPolicy())
    graph_builder.add_node(
        ToolNode(
            agent_tool_kit,
            tags=[InvocationTags.TOOL_CALLS.value],
            name="tools",
            messages_key="messages",
        ),
    )
    graph_builder.add_node("output_node", output_node, retry=RetryPolicy())

    # add edges
    graph_builder.add_conditional_edges(
        "chat_bot",
        tools_condition,
        {"tools": "tools", "output_node": "output_node"},
    )
    graph_builder.add_edge("tools", "chat_bot")
    graph_builder.add_edge("output_node", "__end__")

    # set entry point
    graph_builder.set_entry_point("init_node")

    # compile graph
    graph = graph_builder.compile()

    return graph


graph = create_graph()

__all__ = ["graph"]

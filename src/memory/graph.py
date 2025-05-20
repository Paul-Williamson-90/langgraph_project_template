import asyncio
import functools
import logging
from typing import Any

from langgraph.func import entrypoint, task
from langmem import create_memory_store_manager  # type: ignore[import-untyped]

from src.memory.config import Configuration
from src.memory.state import ProcessorState, State

logger = logging.getLogger("memory")


@functools.lru_cache(maxsize=100)
def get_store_manager(function_name: str):
    configurable = Configuration.from_context()
    memory_config = next(
        conf for conf in configurable.memory_types if conf.name == function_name
    )

    kwargs: dict[str, Any] = {
        "enable_inserts": memory_config.update_mode == "insert",
    }
    if memory_config.system_prompt:
        kwargs["instructions"] = memory_config.system_prompt

    return create_memory_store_manager(
        configurable.model,
        namespace=("memories", "{user_id}", function_name),
        **kwargs,
    )


@task()  # type: ignore[arg-type]
async def process_memory_type(state: ProcessorState) -> None:
    """Extract the user's state from the conversation and update the memory."""
    configurable = Configuration.from_context()
    store_manager = get_store_manager(state["function_name"])
    await store_manager.ainvoke(
        {"messages": state["messages"], "max_steps": configurable.max_extraction_steps},
        config={
            "configurable": {
                "model": configurable.model,
                "user_id": configurable.user_id,
            }
        },
    )


@entrypoint(config_schema=Configuration)
async def graph(state: State) -> None:
    """Iterate over all memory types in the configuration.

    It will route each memory type from configuration to the corresponding memory update node.

    The memory update nodes will be executed in parallel.
    """
    if not state["messages"]:
        raise ValueError("No messages provided")
    configurable = Configuration.from_context()
    await asyncio.gather(
        *[
            process_memory_type(
                ProcessorState(messages=state["messages"], function_name=v.name),  # type: ignore[arg-type]
            )
            for v in configurable.memory_types
        ]
    )


__all__ = ["graph"]

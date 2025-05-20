from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated

### Issues Regarding Pydantic BaseModel States ###
# https://github.com/langchain-ai/langgraph/issues/4699
# Wait for resolution before using Pydantic BaseModel


@dataclass(kw_only=True)
class State:
    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


@dataclass(kw_only=True)
class InputState:
    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


@dataclass(kw_only=True)
class OutputState:
    message: AIMessage

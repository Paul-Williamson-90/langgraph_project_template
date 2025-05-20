from enum import StrEnum


class InvocationTags(StrEnum):
    """Enum for invocation tags.
    These should be used to tag llm invocations to allow
    control over verbosity to end users.
    """

    MODEL_CALL = "model_call"
    TOOL_CALLS = "tool_calls"

from typing import Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class Configuration(BaseModel):
    # Model Configuration
    model: str = "gpt-4.1-mini"
    model_provider: str = "openai"
    max_tokens: int = 1_000_000

    # Prompts
    system_prompt: str = ""

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        return cls(**configurable)

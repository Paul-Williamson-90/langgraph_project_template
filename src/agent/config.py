import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. {user_info} The current time is {time}."
)


class Configuration(BaseModel):
    # Model Configuration
    model: str = "gpt-4.1-mini"
    model_provider: str = "openai"
    max_tokens: int = 1_000_000

    # Prompts
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # user configuration
    user_id: str = "default"

    # memory configuration
    memories_limit: int = 10
    n_msgs_search: int = 3
    delay_seconds: int = 3
    mem_assistant_id: str = "memory"
    memory_types: Optional[list[dict]] = None

    # postgres
    pg_host: str = os.environ.get("POSTGRES_HOST", "localhost")
    pg_port: str = os.environ.get("POSTGRES_PORT", "5432")
    pg_user: str = os.environ.get("POSTGRES_USER", "postgres")
    pg_password: str = os.environ.get("POSTGRES_PASSWORD", "postgres")
    pg_db: str = os.environ.get("POSTGRES_DB", "postgres")

    @property
    def pg_url(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        return cls(**configurable)

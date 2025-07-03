from enum import Enum
from typing import Any

class Provider(str, Enum):
    OPENAI: str
    ANTHROPIC: str
    GEMINI: str
    GROQ: str
    BEDROCK: str


def register_expressions() -> None: ...

# Stub out the inference helpers so static type checkers resolve names.

def inference_async(*args: Any, **kwargs: Any) -> Any: ...

def inference(*args: Any, **kwargs: Any) -> Any: ...

def inference_messages(*args: Any, **kwargs: Any) -> Any: ...

def string_to_message(*args: Any, **kwargs: Any) -> Any: ...

def combine_messages(*args: Any, **kwargs: Any) -> Any: ...
"""LLM clients. Every backend implements the same `LLMClient` protocol so
agents can be swapped between local (Ollama) and Azure ML Managed Endpoints
via config file alone.
"""

from .base import LLMClient, LLMResponse
from .ollama import OllamaClient
from .huggingface import HuggingFaceClient
from .factory import make_client

__all__ = ["LLMClient", "LLMResponse", "OllamaClient", "HuggingFaceClient", "make_client"]

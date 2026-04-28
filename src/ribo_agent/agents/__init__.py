"""Agent implementations. Each agent takes an LLMClient + a config dict
and exposes `.answer(mcq)` returning a predicted letter with metadata.
"""

from .base import Agent, Prediction
from .zeroshot import ZeroShotAgent
from .rag import RAGAgent

__all__ = ["Agent", "Prediction", "ZeroShotAgent", "RAGAgent"]

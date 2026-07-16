"""AnswerProvider interface + provider factory."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Answer, ScoredChunk


class AnswerProvider(ABC):
    name: str = "base"

    @abstractmethod
    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        """Compose a grounded answer (with citations) from retrieved chunks."""


def get_answer_provider(provider: str | None = None) -> AnswerProvider:
    """Build the answer provider selected by RAGQA_ANSWER_PROVIDER.

    - ``extractive`` (default): offline, no keys -- composes an answer from the
      top retrieved chunks with file:line citations
    - ``anthropic``: Claude via the anthropic SDK (ANTHROPIC_API_KEY)
    - ``openai``: OpenAI chat completions (OPENAI_API_KEY)
    """
    from .. import config

    provider = (provider or config.answer_provider()).lower()
    if provider == "extractive":
        from .extractive import ExtractiveProvider

        return ExtractiveProvider()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=config.anthropic_model())
    if provider == "openai":
        from .openai_provider import OpenAIAnswerProvider

        return OpenAIAnswerProvider(model=config.openai_answer_model())
    raise ValueError(
        f"Unknown answer provider: {provider!r} (expected extractive|anthropic|openai)"
    )

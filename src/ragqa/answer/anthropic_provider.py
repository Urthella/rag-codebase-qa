"""Anthropic (Claude) answer provider.

Optional: requires `pip install 'ragqa[anthropic]'` and ANTHROPIC_API_KEY.
Model defaults to ``claude-sonnet-5`` (override with RAGQA_ANTHROPIC_MODEL).
The prompt enforces strict grounding with [file:start-end] citations -- see
``prompts.SYSTEM_PROMPT``.
"""

from __future__ import annotations

from ..models import Answer, Citation, ScoredChunk
from .base import AnswerProvider
from .prompts import I_DONT_KNOW, SYSTEM_PROMPT, build_user_message, extract_citations


class AnthropicProvider(AnswerProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1500):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Anthropic provider requires the anthropic SDK: "
                "pip install 'ragqa[anthropic]' and set ANTHROPIC_API_KEY"
            ) from exc
        # Resolves credentials from the environment (ANTHROPIC_API_KEY etc.)
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:  # pragma: no cover - needs API key
        if not chunks:
            return Answer(
                text=I_DONT_KNOW + " Nothing was retrieved for this question.",
                citations=[],
                provider=self.name,
                grounded=False,
            )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, chunks)}],
        )
        if response.stop_reason == "refusal":
            return Answer(
                text=I_DONT_KNOW + " (The model declined to answer.)",
                citations=[],
                provider=self.name,
                grounded=False,
            )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        citations = extract_citations(text)
        grounded = not text.startswith(I_DONT_KNOW)
        if grounded and not citations:
            # Model answered without citations -> attach retrieval citations
            # so the UI can still show provenance.
            citations = [
                Citation(
                    path=sc.chunk.path,
                    start_line=sc.chunk.start_line,
                    end_line=sc.chunk.end_line,
                )
                for sc in chunks[:3]
            ]
        return Answer(text=text, citations=citations, provider=self.name, grounded=grounded)

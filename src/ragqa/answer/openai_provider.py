"""OpenAI answer provider (optional; requires `pip install 'ragqa[openai]'`
and OPENAI_API_KEY). Model defaults to gpt-4o-mini (RAGQA_OPENAI_MODEL)."""

from __future__ import annotations

from ..models import Answer, Citation, ScoredChunk
from .base import AnswerProvider
from .prompts import I_DONT_KNOW, SYSTEM_PROMPT, build_user_message, extract_citations


class OpenAIAnswerProvider(AnswerProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1500):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAI provider requires the openai SDK: "
                "pip install 'ragqa[openai]' and set OPENAI_API_KEY"
            ) from exc
        self._client = OpenAI()
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
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, chunks)},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        citations = extract_citations(text)
        grounded = not text.startswith(I_DONT_KNOW)
        if grounded and not citations:
            citations = [
                Citation(
                    path=sc.chunk.path,
                    start_line=sc.chunk.start_line,
                    end_line=sc.chunk.end_line,
                )
                for sc in chunks[:3]
            ]
        return Answer(text=text, citations=citations, provider=self.name, grounded=grounded)

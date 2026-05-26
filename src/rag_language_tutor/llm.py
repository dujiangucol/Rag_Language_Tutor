from __future__ import annotations

import os
from dataclasses import dataclass

from rag_language_tutor.index import SearchResult


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str


class TutorLLM:
    def __init__(self, model: str, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, results: list[SearchResult], learner_level: str) -> LLMResponse:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            response = self._openai_generate(question, results, learner_level, api_key)
            if response:
                return response
        return self._offline_generate(question, results, learner_level)

    def _openai_generate(
        self,
        question: str,
        results: list[SearchResult],
        learner_level: str,
        api_key: str,
    ) -> LLMResponse | None:
        try:
            from openai import OpenAI
        except ImportError:
            return None

        context = "\n\n".join(
            f"[{result.rank}] {result.chunk.source} / {result.chunk.section}: {result.chunk.text}"
            for result in results
        )
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous but friendly RAG language tutor. "
                        "Ground answers in the provided context, cite sources by bracket number, "
                        "include a correction pattern, examples, and a tiny practice task. "
                        "If context is weak, say what is missing."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Learner level: {learner_level}\n"
                        f"Question: {question}\n\n"
                        f"Retrieved context:\n{context}"
                    ),
                },
            ],
        )
        return LLMResponse(text=completion.choices[0].message.content or "", provider="openai")

    def _offline_generate(
        self,
        question: str,
        results: list[SearchResult],
        learner_level: str,
    ) -> LLMResponse:
        if not results:
            return LLMResponse(
                text=(
                    "I do not have enough retrieved context to answer confidently. "
                    "Try ingesting more lesson material or asking a more specific question."
                ),
                provider="offline-template",
            )

        best = results[0]
        supporting = results[1:3]
        examples = extract_examples([result.chunk.text for result in results])
        citations = ", ".join(f"[{result.rank}]" for result in results[:3])

        text = "\n".join(
            [
                f"Short answer: based on the retrieved lesson material, the key idea is in {best.chunk.section}.",
                "",
                f"Explanation for a {learner_level} learner:",
                best.chunk.text[:650],
                "",
                "Useful examples:",
                *[f"- {example}" for example in examples[:3]],
                "",
                "Common mistake to watch for:",
                "Do not memorize only the rule name. Check the time meaning, the verb type, and whether the situation is temporary or general.",
                "",
                "Mini practice:",
                f"Write two original sentences that answer this question: {question}",
                "",
                f"Sources: {citations}",
            ]
        )
        if supporting:
            text += "\n\nRelated retrieved sections: " + "; ".join(
                f"{result.chunk.source} / {result.chunk.section}" for result in supporting
            )
        return LLMResponse(text=text, provider="offline-template")


def extract_examples(texts: list[str]) -> list[str]:
    examples: list[str] = []
    for text in texts:
        for line in text.split(". "):
            clean = line.strip(" -\n")
            if 15 <= len(clean) <= 140 and any(marker in clean.lower() for marker in ["example", "i ", "she ", "you ", "wǒ", "nǐ"]):
                examples.append(clean if clean.endswith(".") else f"{clean}.")
    return examples or ["Create one correct sentence, one incorrect sentence, and explain the difference."]

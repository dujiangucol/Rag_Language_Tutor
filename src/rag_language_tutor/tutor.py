from __future__ import annotations

from dataclasses import dataclass

from rag_language_tutor.config import TutorConfig
from rag_language_tutor.index import SearchResult, TutorIndex
from rag_language_tutor.llm import TutorLLM


@dataclass(frozen=True)
class TutorResponse:
    answer: str
    confidence: float
    citations: list[str]
    provider: str
    retrieved: list[SearchResult]


class RAGLanguageTutor:
    def __init__(self, index: TutorIndex, config: TutorConfig | None = None) -> None:
        self.config = config or TutorConfig.from_env()
        self.index = index
        self.llm = TutorLLM(model=self.config.model, temperature=self.config.temperature)

    @classmethod
    def from_saved_index(cls, config: TutorConfig | None = None) -> "RAGLanguageTutor":
        config = config or TutorConfig.from_env()
        return cls(TutorIndex.load(config.index_path), config)

    def answer(self, question: str, learner_level: str = "intermediate") -> TutorResponse:
        results = self.index.search(question, top_k=self.config.top_k)
        confidence = sum(result.score for result in results[:3]) / max(min(3, len(results)), 1)
        filtered = [result for result in results if result.score >= self.config.min_confidence]
        context = filtered or results[:2]
        llm_response = self.llm.generate(question, context, learner_level)
        citations = [
            f"[{result.rank}] {result.chunk.source} / {result.chunk.section}"
            for result in context
        ]
        return TutorResponse(
            answer=llm_response.text,
            confidence=float(confidence),
            citations=citations,
            provider=llm_response.provider,
            retrieved=context,
        )

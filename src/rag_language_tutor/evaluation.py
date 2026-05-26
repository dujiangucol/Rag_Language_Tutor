from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rag_language_tutor.index import TutorIndex


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_terms: tuple[str, ...]


EVAL_SET = [
    EvalCase(
        question="What is the difference between present simple and present continuous?",
        expected_terms=("present simple", "present continuous", "temporary"),
    ),
    EvalCase(
        question="When should I use a or the?",
        expected_terms=("articles", "specific", "non-specific"),
    ),
    EvalCase(
        question="How do Mandarin tones work?",
        expected_terms=("tones", "rising", "falling"),
    ),
    EvalCase(
        question="How should a tutor correct grammar mistakes?",
        expected_terms=("corrective feedback", "better", "why"),
    ),
]


def evaluate_retrieval(index: TutorIndex, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for case in EVAL_SET:
        results = index.search(case.question, top_k=top_k)
        combined = " ".join(
            f"{result.chunk.source} {result.chunk.section} {result.chunk.text}".lower()
            for result in results
        )
        hits = [term for term in case.expected_terms if term.lower() in combined]
        rows.append(
            {
                "question": case.question,
                "expected_terms": "|".join(case.expected_terms),
                "hits": "|".join(hits),
                "recall": len(hits) / len(case.expected_terms),
                "top_source": results[0].chunk.source if results else "",
                "top_section": results[0].chunk.section if results else "",
                "top_score": results[0].score if results else 0.0,
            }
        )
    return pd.DataFrame(rows)

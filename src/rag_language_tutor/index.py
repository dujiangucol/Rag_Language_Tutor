from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_language_tutor.documents import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int


class TutorIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunks: list[DocumentChunk]) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[DocumentChunk]) -> "TutorIndex":
        if not chunks:
            raise ValueError("Cannot build an index with no document chunks.")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([chunk_to_index_text(chunk) for chunk in chunks])
        return cls(vectorizer, matrix, chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.vectorizer.transform([query])
        dense_scores = cosine_similarity(query_vector, self.matrix).ravel()
        keyword_scores = np.array([keyword_overlap(query, chunk_to_index_text(chunk)) for chunk in self.chunks])
        phrase_scores = np.array([phrase_overlap(query, chunk_to_index_text(chunk)) for chunk in self.chunks])
        scores = 0.72 * dense_scores + 0.18 * keyword_scores + 0.10 * phrase_scores
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]), rank=rank + 1)
            for rank, index in enumerate(order)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: Path) -> "TutorIndex":
        with path.open("rb") as file:
            index = pickle.load(file)
        if not isinstance(index, cls):
            raise TypeError("Loaded object is not a TutorIndex.")
        return index


def keyword_overlap(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def phrase_overlap(query: str, text: str) -> float:
    query_lower = query.lower()
    text_lower = text.lower()
    phrases = [
        "present simple",
        "present continuous",
        "corrective feedback",
        "grammar mistakes",
        "mandarin tones",
        "measure words",
    ]
    matched = [phrase for phrase in phrases if phrase in query_lower and phrase in text_lower]
    relevant = [phrase for phrase in phrases if phrase in query_lower]
    if not relevant:
        return 0.0
    return len(matched) / len(relevant)


def tokenize(value: str) -> list[str]:
    return [term for term in "".join(char.lower() if char.isalnum() else " " for char in value).split() if len(term) > 2]


def chunk_to_index_text(chunk: DocumentChunk) -> str:
    return f"{chunk.source} {chunk.section} {chunk.section} {chunk.text}"

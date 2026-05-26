from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TutorConfig:
    corpus_dir: Path = ROOT / "data" / "corpus"
    raw_dir: Path = ROOT / "data" / "raw"
    index_path: Path = ROOT / "data" / "processed" / "tutor_index.pkl"
    evaluation_path: Path = ROOT / "outputs" / "evaluation" / "retrieval_eval.csv"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_k: int = 5
    min_confidence: float = 0.12

    @classmethod
    def from_env(cls) -> "TutorConfig":
        if load_dotenv:
            load_dotenv(ROOT / ".env")

        return cls(
            model=os.getenv("RAG_TUTOR_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("RAG_TUTOR_TEMPERATURE", "0.2")),
            top_k=int(os.getenv("RAG_TUTOR_TOP_K", "5")),
            min_confidence=float(os.getenv("RAG_TUTOR_MIN_CONFIDENCE", "0.12")),
        )

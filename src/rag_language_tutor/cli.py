from __future__ import annotations

import argparse
from pathlib import Path

from rag_language_tutor.config import TutorConfig
from rag_language_tutor.documents import load_documents
from rag_language_tutor.evaluation import evaluate_retrieval
from rag_language_tutor.index import TutorIndex
from rag_language_tutor.tutor import RAGLanguageTutor


def build_index(config: TutorConfig, include_raw_pdfs: bool = False) -> TutorIndex:
    paths = sorted(config.corpus_dir.glob("*.md")) + sorted(config.corpus_dir.glob("*.txt"))
    if include_raw_pdfs:
        paths += sorted(config.raw_dir.glob("*.pdf"))

    chunks = load_documents(paths)
    index = TutorIndex.build(chunks)
    index.save(config.index_path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG language tutor command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the local retrieval index.")
    ingest_parser.add_argument("--include-raw-pdfs", action="store_true", help="Also parse PDFs in data/raw.")

    ask_parser = subparsers.add_parser("ask", help="Ask the tutor a language question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--level", default="intermediate")

    subparsers.add_parser("evaluate", help="Run retrieval evaluation.")

    args = parser.parse_args()
    config = TutorConfig.from_env()

    if args.command == "ingest":
        index = build_index(config, include_raw_pdfs=args.include_raw_pdfs)
        print(f"Built index with {len(index.chunks)} chunks at {config.index_path}")
    elif args.command == "ask":
        if not Path(config.index_path).exists():
            build_index(config)
        tutor = RAGLanguageTutor.from_saved_index(config)
        response = tutor.answer(args.question, learner_level=args.level)
        print(response.answer)
        print("\nCitations:")
        for citation in response.citations:
            print(f"- {citation}")
        print(f"\nConfidence: {response.confidence:.3f}")
        print(f"Provider: {response.provider}")
    elif args.command == "evaluate":
        if not Path(config.index_path).exists():
            build_index(config)
        index = TutorIndex.load(config.index_path)
        report = evaluate_retrieval(index)
        config.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(config.evaluation_path, index=False)
        print(report.to_string(index=False))
        print(f"\nSaved evaluation to {config.evaluation_path}")

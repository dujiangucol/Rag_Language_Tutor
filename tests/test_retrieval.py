from pathlib import Path

from rag_language_tutor.documents import load_documents
from rag_language_tutor.evaluation import evaluate_retrieval
from rag_language_tutor.index import TutorIndex


def test_index_retrieves_relevant_grammar_chunk():
    chunks = load_documents([Path("data/corpus/english_grammar_guide.md")])
    index = TutorIndex.build(chunks)

    results = index.search("present simple versus present continuous", top_k=2)

    assert results
    assert "Present" in results[0].chunk.section


def test_evaluation_has_recall_values():
    chunks = load_documents(sorted(Path("data/corpus").glob("*.md")))
    index = TutorIndex.build(chunks)
    report = evaluate_retrieval(index)

    assert report["recall"].between(0, 1).all()
    assert report["recall"].mean() > 0.5

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_language_tutor.cli import build_index
from rag_language_tutor.config import TutorConfig
from rag_language_tutor.tutor import RAGLanguageTutor


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the difference between present simple and present continuous?"
    config = TutorConfig.from_env()
    if not config.index_path.exists():
        build_index(config)
    response = RAGLanguageTutor.from_saved_index(config).answer(question)
    print(response.answer)
    print("\nCitations:")
    for citation in response.citations:
        print(f"- {citation}")

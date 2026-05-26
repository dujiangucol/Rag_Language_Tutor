import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_language_tutor.cli import build_index
from rag_language_tutor.config import TutorConfig


if __name__ == "__main__":
    index = build_index(TutorConfig.from_env())
    print(f"Built index with {len(index.chunks)} chunks.")

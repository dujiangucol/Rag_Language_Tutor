import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_language_tutor.cli import build_index
from rag_language_tutor.config import TutorConfig
from rag_language_tutor.evaluation import evaluate_retrieval
from rag_language_tutor.index import TutorIndex


if __name__ == "__main__":
    config = TutorConfig.from_env()
    if not config.index_path.exists():
        build_index(config)
    report = evaluate_retrieval(TutorIndex.load(config.index_path))
    config.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.evaluation_path, index=False)
    print(report.to_string(index=False))

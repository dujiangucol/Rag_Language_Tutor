from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install streamlit to run the app: pip install streamlit") from exc

from rag_language_tutor.cli import build_index
from rag_language_tutor.config import TutorConfig
from rag_language_tutor.tutor import RAGLanguageTutor


st.set_page_config(page_title="RAG Language Tutor", page_icon="RAG", layout="wide")
st.title("RAG Language Tutor")

config = TutorConfig.from_env()
if st.sidebar.button("Rebuild index"):
    build_index(config, include_raw_pdfs=st.sidebar.checkbox("Include raw PDFs", value=False))
    st.sidebar.success("Index rebuilt")

if not config.index_path.exists():
    build_index(config)

tutor = RAGLanguageTutor.from_saved_index(config)
question = st.text_input("Ask a grammar, Mandarin, or tutoring question", "What is the difference between present simple and present continuous?")
level = st.selectbox("Learner level", ["beginner", "intermediate", "advanced"], index=1)

if st.button("Ask"):
    response = tutor.answer(question, learner_level=level)
    st.markdown(response.answer)
    st.caption(f"Provider: {response.provider} | Confidence: {response.confidence:.3f}")
    with st.expander("Retrieved evidence"):
        for result in response.retrieved:
            st.markdown(f"**[{result.rank}] {result.chunk.source} / {result.chunk.section}** — score `{result.score:.3f}`")
            st.write(result.chunk.text)

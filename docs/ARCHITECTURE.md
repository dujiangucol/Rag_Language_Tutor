# Architecture

The tutor is built around a local-first RAG pipeline.

## Pipeline

1. `documents.py` loads Markdown, text, and PDF files.
2. Documents are split into sentence-aware chunks with source metadata.
3. `index.py` builds a hybrid retrieval index with:
   - TF-IDF cosine similarity
   - keyword overlap
   - phrase-aware boosts for language-learning concepts
4. `tutor.py` retrieves evidence and computes a confidence score.
5. `llm.py` generates an answer through:
   - OpenAI, when `OPENAI_API_KEY` and the package are available
   - an offline template fallback, when no API is configured
6. `evaluation.py` measures retrieval recall against expected concepts.

## Why This Is More Than A Basic Demo

The system separates ingestion, retrieval, generation, evaluation, and presentation. That makes it easier to test each stage and reason about failure modes:

- bad chunking
- weak retrieval
- low confidence context
- unsupported answers
- missing source citations
- API unavailability

The offline fallback is deliberate. It lets reviewers run the project without secrets while still seeing the full RAG control flow.

## Secret Handling

The code only reads `OPENAI_API_KEY` from environment variables or a local `.env` file. The repository includes `.env.example` but ignores real `.env` files.

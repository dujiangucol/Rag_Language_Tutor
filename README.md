# RAG Language Tutor

A retrieval-augmented language tutor for English grammar, Mandarin learning, and tutoring pedagogy. The project is designed as a portfolio-ready GenAI/RAG system: it can run fully offline for demos, and it can optionally use OpenAI when `OPENAI_API_KEY` is available locally.

No API keys are stored in the repository. Use `.env.example` as the template for local secrets.

## What It Shows

- Document ingestion for Markdown, text, and PDF sources
- Sentence-aware chunking with source, section, and page metadata
- Hybrid retrieval using TF-IDF similarity, keyword overlap, and phrase-aware boosts
- RAG answer generation with citations and confidence scores
- Offline fallback tutor so the app works without paid APIs
- Optional OpenAI answer generation when `openai` and `OPENAI_API_KEY` are available
- Retrieval evaluation with a small gold set
- CLI entry point, wrapper scripts, and optional Streamlit app
- Tests covering chunking, indexing, retrieval, and evaluation

## Process, Techniques, And Pipeline

This project is built to demonstrate the full RAG lifecycle, not just a chatbot wrapper.

### 1. Knowledge Ingestion

The system loads curated Markdown lessons from `data/corpus/` and can also parse source PDFs from `data/raw/`. Each document keeps metadata such as source file, section title, and page number when available. This makes every answer traceable back to retrieved evidence.

### 2. Chunking Strategy

Documents are split with sentence-aware chunking instead of arbitrary character cuts. The chunker tries to preserve complete explanations and examples, then stores each chunk as a `DocumentChunk` with:

- `chunk_id`
- `text`
- `source`
- `section`
- optional `page`

This matters because language-learning content often depends on examples staying attached to the rule they explain.

### 3. Hybrid Retrieval

The retriever combines three signals:

- TF-IDF cosine similarity for semantic-ish lexical relevance
- keyword overlap for direct concept matching
- phrase-aware boosts for high-value language-learning terms such as `present simple`, `present continuous`, `corrective feedback`, and `Mandarin tones`

The goal is to make retrieval explainable and robust without requiring paid embedding APIs. The same design can later be upgraded to dense embeddings or rerankers.

### 4. Grounded Tutor Generation

For each learner question, the tutor retrieves top evidence chunks, computes a confidence score, and generates a structured tutoring answer with:

- short answer
- learner-level explanation
- examples
- common mistake
- mini practice task
- source citations

If `OPENAI_API_KEY` is available locally, the system can call OpenAI for richer generation. If no key is available, it uses an offline template so the RAG pipeline remains demoable without secrets.

### 5. Evaluation Loop

The project includes a small retrieval evaluation set in code. `python main.py evaluate` checks whether expected concepts appear in the retrieved evidence and writes a report. Current offline evaluation reaches full expected-term recall on the curated test questions.

### 6. Safety And Secret Handling

The project never hard-codes API keys. Local secrets live in `.env`, while `.env.example` documents the expected variables. Generated indexes and evaluation files are also ignored so GitHub stays clean.

## Layout

- `src/rag_language_tutor/`: core package
- `data/corpus/`: curated demo corpus
- `data/raw/`: original PDFs pulled from the phase-1 project
- `data/processed/`: generated local indexes, ignored by git
- `outputs/evaluation/`: generated evaluation outputs, ignored by git
- `app/streamlit_app.py`: optional UI
- `tests/`: regression tests

## Quick Start

```bash
python main.py ingest
python main.py ask "What is the difference between present simple and present continuous?"
python main.py evaluate
python -m pytest tests
```

Optional wrappers:

```bash
python scripts/ingest.py
python scripts/ask.py "How do Mandarin tones work?"
python scripts/evaluate.py
```

## Optional OpenAI Mode

Create a local `.env` file:

```bash
cp .env.example .env
```

Then fill the empty `OPENAI_API_KEY` value in your local `.env` file.

`.env` and `.env.*` are ignored by git. Do not commit them.

## Run The App

```bash
streamlit run app/streamlit_app.py
```

The app will build the local index if needed and display retrieved evidence for every answer.

## Current Offline Evaluation

The built-in evaluation set currently reaches full expected-term recall on the curated corpus:

- present simple vs present continuous
- articles
- Mandarin tones
- corrective feedback for grammar mistakes

This is intentionally small but explicit: the goal is to show how RAG quality is measured, not just that a chatbot can answer.

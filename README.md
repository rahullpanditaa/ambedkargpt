# AmbedkarGPT – Semantic Graph RAG (SemRAG)

AmbedkarGPT is an end-to-end implementation of the **Semantic Graph Retrieval-Augmented Generation (SemRAG)** framework, applied to *Dr. B. R. Ambedkar: Writings and Speeches*.
The system combines **semantic chunking**, **knowledge graphs**, **community detection**, and **local + global retrieval** to produce grounded, interpretable answers using a **local LLM**.

This project follows the SemRAG paper closely, excluding only the evaluation section.

---

## High-Level Architecture

```
PDF
 └── Sentence Extraction
      └── Buffer Merge
           └── Semantic Chunking (Algorithm 1)
                └── Entity & Relation Extraction
                     └── Knowledge Graph
                          └── Community Detection (Leiden)
                               └── Community Summaries (LLM)
                                    ├── Local Graph RAG (Eq. 4)
                                    ├── Global Graph RAG (Eq. 5)
                                    └── Final Answer Generation (LLM)
```

---

## Core Features

* **Semantic Chunking (Algorithm 1)**
  Sentence embeddings + cosine similarity with buffer merging and token-aware splitting.

* **Knowledge Graph Construction**
  Entities extracted from chunks using spaCy NER, relations formed via co-occurrence, and aggregated into a weighted graph.

* **Community Detection**
  Leiden algorithm groups entities into semantic communities.

* **Local Graph RAG (Equation 4)**
  Query → entities → graph expansion → chunk scoring using entity relevance × frequency.

* **Global Graph RAG (Equation 5)**
  Query → community summaries → high-level thematic retrieval.

* **Grounded Answer Generation**
  Local LLM (via Ollama) produces answers strictly based on retrieved communities and chunks.

---

## Tech Stack

* **Python** 3.9+
* **Sentence Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
* **NER**: spaCy (`en_core_web_sm`)
* **Graphs**: networkx, igraph
* **Community Detection**: Leiden algorithm
* **LLM Runtime**: Ollama (Mistral / Llama3)
* **PDF Processing**: pypdf
* **CLI / Packaging**: uv

---

## Project Structure

```
src/
├── ingest/                # PDF ingestion & sentence extraction
├── chunking/              # Buffer merge + semantic chunking
├── graph/                 # Entity extraction, graph, communities, summaries
├── retrieval/             # Local & Global Graph RAG
├── llm/                   # Final answer generation
├── utils/                 # Constants & shared helpers
├── cli_commands.py        # CLI command implementations
└── ambedkargpt.py         # CLI entrypoint

data/
├── Ambedkar_book.pdf
└── processed/             # Generated artifacts (NOT committed)
```

---

## CLI Usage

### 1. Build Index (Offline)

Runs the full preprocessing and indexing pipeline.

```bash
uv run python -m src.ambedkargpt build-index
```

This performs:

* PDF ingestion
* Semantic chunking
* Knowledge graph construction
* Community detection
* Community summarization
* Community embedding generation

> ⚠️ This step is compute-intensive and may take several minutes due to LLM summarization.

---

### 2. Local Search (Equation 4)

Retrieves the most relevant chunks using entity-aware graph traversal.

```bash
uv run python -m src.ambedkargpt local-search "caste system"
```

Output:

* Ranked chunks
* Relevance scores
* Direct textual evidence

---

### 3. Global Search (Equation 5)

Retrieves high-level thematic communities relevant to the query.

```bash
uv run python -m src.ambedkargpt global-search "religion and social reform"
```

Output:

* Top-K communities
* Similarity scores
* LLM-generated community summaries

---

### 4. Answer Generation (End-to-End)

Combines global + local retrieval and generates a grounded answer.

```bash
uv run python -m src.ambedkargpt answer "How did Ambedkar critique Hindu social structures?"
```

Output:

* Final natural-language answer
* Explicit grounding in retrieved communities and chunks

---

## Design Principles

* **Separation of Concerns**
  Retrieval is orchestrated by the CLI, not the LLM.

* **Hierarchical Retrieval**
  Communities provide global context; chunks provide grounding.

* **Interpretability**
  Answers reference the semantic communities they draw from.

* **Local-First**
  No external APIs required; everything runs locally.

---

## Notes on Generated Data

Files under `data/processed/` are **generated artifacts** and should **not** be committed to git.

Add to `.gitignore`:

```
data/processed/
```

---

## Limitations & Future Work

* Relation extraction uses co-occurrence (can be replaced with dependency-based RE)
* Community summaries are slow due to LLM inference
* Evaluation metrics from the SemRAG paper are not implemented

---

## References

* SemRAG: Semantic Graph Retrieval-Augmented Generation (Research Paper)
* Dr. B. R. Ambedkar: *Writings and Speeches*

---

## Live Demo (CLI-Based)

This project is designed to be demonstrated live using CLI commands.
No separate demo script is required — the CLI itself *is* the demo.

Below are example commands that can be run during the interview to showcase the full SemRAG pipeline.

---

### Local Graph RAG (Equation 4)

Demonstrates entity-aware, graph-based retrieval of relevant chunks.

```bash
uv run python -m src.ambedkargpt local-search "caste system"
uv run python -m src.ambedkargpt local-search "untouchability"
uv run python -m src.ambedkargpt local-search "varna hierarchy"
uv run python -m src.ambedkargpt local-search "Hindu social order"
```

---

### Global Graph RAG (Equation 5)

Demonstrates high-level thematic retrieval using community summaries.

```bash
uv run python -m src.ambedkargpt global-search "religion and social reform"
uv run python -m src.ambedkargpt global-search "social justice and equality"
uv run python -m src.ambedkargpt global-search "critique of Hinduism"
uv run python -m src.ambedkargpt global-search "democracy and constitutional morality"
```

---

### End-to-End Question Answering

Demonstrates the full SemRAG pipeline: global retrieval → local retrieval → grounded LLM answer.

```bash
uv run python -m src.ambedkargpt answer "Why did Ambedkar oppose the caste system?"
uv run python -m src.ambedkargpt answer "How did Ambedkar view religion in relation to social reform?"
uv run python -m src.ambedkargpt answer "What role did education play in Ambedkar’s vision of social equality?"
uv run python -m src.ambedkargpt answer "How did Ambedkar critique Hindu social structures?"
```

---

These commands together demonstrate:

* Semantic chunking and graph construction (offline)
* Local Graph RAG (Equation 4)
* Global Graph RAG (Equation 5)
* Grounded LLM-based answer generation

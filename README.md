# Vidhya-Setu

**A Localized Multi-Agent Framework Using GraphRAG for Adaptive STEM Learning in Low-Resource Environments**

Vidhya-Setu is a fully offline intelligent tutoring system designed for rural and under-resourced Indian schools. It runs entirely on local hardware with no internet connection, no cloud APIs, and no subscription cost. The system reads NCERT Class 9 Science textbooks, automatically builds a concept-prerequisite knowledge graph, and uses a two-agent architecture to deliver adaptive, step-by-step explanations that adjust in real time based on whether a student understands or is confused.

This project was built as part of an IEEE TechForGood internship initiative, targeting deployment on old, low-spec school hardware with no GPU. See [Hardware Requirements](#hardware-requirements) for what it actually costs to run today.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Current Project Status](#current-project-status)
- [Tech Stack](#tech-stack)
- [Hardware Requirements](#hardware-requirements)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the System](#running-the-system)
- [Knowledge Graph Stats](#knowledge-graph-stats)
- [Known Limitations](#known-limitations)
- [Team](#team)
- [License](#license)

---

## Why This Project Exists

Students in rural India studying NCERT Science have no access to personalized academic support outside the classroom. Existing AI tutoring tools require internet connectivity, cloud subscriptions, or modern hardware that rural schools simply do not have. Vidhya-Setu is built around a different assumption: the student-facing machine is an old donated desktop with no internet and no GPU, and the system still has to work.

## How It Works

When a student asks a question, the system does not just search for matching text. It identifies where that question fits within a structured map of prerequisite concepts built from the textbook itself, then walks the student through that map step by step. If the student shows signs of confusion, the system automatically steps back to a simpler, foundational concept before trying again, rather than repeating the same explanation or pushing forward regardless.

A student asking about acceleration is not handed a definition of acceleration. They are handed this path, and taken along it one concept at a time:

```
mass -> motion -> speed -> velocity -> momentum -> force -> net force -> acceleration
```

## System Architecture

The system is split into two environments by design.

**Build environment** (developer machine, run once): reads NCERT PDFs, extracts concept-prerequisite pairs using a local language model, builds a directed knowledge graph, and creates a search index of the textbook content.

**Deployment environment** (school hardware, runs every session): loads the pre-built knowledge graph and index, and uses a quantized language model to retrieve the right concept and generate explanations, entirely offline.

```
BUILD (developer machine, once)

  NCERT PDFs
      |
      v
  PDF ingestion (PyMuPDF) --> cleaned text chunks
      |
      v
  Triple extraction (Phi-3 Mini + GBNF grammar) --> concept/prerequisite pairs
      |
      v
  Graph construction + cycle removal (NetworkX) --> kg.pkl (a DAG)
      |
      v
  Embedding (BGE-small) --> embeddings.npy + chunks.json


DEPLOY (school machine, every session)

  Student question
      |
      v
  Hybrid retrieval: dense vector search + BM25 keyword search
      |
      v
  Reciprocal rank fusion --> best chunks --> matching graph concept
      |
      v
  Path tracker --> prerequisite chain, capped at 8 steps
      |
      v
  Instructor agent (Phi-3 Mini) --> explanation for the current concept
      |
      v
  Student reply --> Diagnostic agent (rule-based)
      |
      +-- understood --> advance to the next concept
      +-- confused   --> step back to an easier one, explain with an analogy
      +-- unclear    --> re-explain the same concept
      |
      v
  Session checkpoint written to disk after every exchange
```

---

## Current Project Status

The table below reflects what is actually working today, not the full target scope.

| Component | Status |
|---|---|
| PDF ingestion pipeline | Working |
| Knowledge graph construction | Working — valid DAG, cycle-free |
| Hybrid retrieval (vector + keyword fusion) | Working |
| Path tracker with backtracking | Working |
| Instructor agent | Working |
| Diagnostic agent | Working |
| Gradio web UI, one session per browser tab | Working |
| Regression checks (`scripts/run_checks.py`) | Working — 46 checks |
| Quantitative evaluation (BLEU / ROUGE-L) | Not started |
| Teacher dashboard | Not started |
| Multilingual support | Not started |
| Mobile deployment | Out of scope for this version |

Currently validated on two NCERT Class 9 Science chapters: **Chapter 4 — Describing Motion Around Us** and **Chapter 6 — How Forces Affect Motion**. Expansion to the full syllabus is planned.

---

## Tech Stack

All components are open-source, run entirely offline, and cost nothing to license.

| Layer | Technology | Why |
|---|---|---|
| Language model | Phi-3 Mini 3.8B, Q4_K_M GGUF (llama-cpp-python) | MIT licensed, so it can be resold; runs CPU-only |
| Constrained decoding | GBNF grammar (llama.cpp) | Makes malformed JSON structurally impossible during extraction |
| Knowledge graph | NetworkX directed graph | Pure Python, pickles to 24 KB |
| Search index | numpy flat index + rank-bm25 | 301 chunks is small enough for exact search; two plain files, no database to migrate |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) | Small, strong on short retrieval queries |
| Ranking | Reciprocal rank fusion | Merges vector and keyword rankings by position, not by incomparable scores |
| PDF processing | PyMuPDF / pymupdf4llm | Keeps section headings, which the extractor uses as hints |
| Web interface | Gradio | Local browser UI with no frontend build step |
| Language | Python 3.11 | |

Nothing here calls out to a network at runtime. The full check suite passes with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set.

---

## Hardware Requirements

Measured on the development machine (Windows 11, CPU-only, 4 threads), loading exactly what
the tutor loads:

| | RAM |
|---|---|
| Python and imports | 436 MB |
| + Phi-3 Mini Q4_K_M (`N_CTX=2048`) | 3,887 MB |
| Peak during one explanation | 5,072 MB |

**Practical minimum: 8 GB RAM.** The original 4 GB target is not met — the language model
accounts for essentially all of it, while the knowledge graph, search index and BM25 index
together come to about 20 MB. Reaching 4 GB needs a smaller model, which is tracked as
future work rather than assumed.

Disk: 2.4 GB, almost entirely the model file.

Timings on the same machine: retrieval ~18 ms, explanation ~15 s for 120 words, graph
rebuild ~17 minutes for 301 chunks.

---

## Project Structure

```
IEEE Project/
├── data/
│   ├── raw_pdfs/          NCERT source PDFs (not tracked in git)
│   ├── chunks/            Extracted and chunked text (JSON)
│   ├── graph/             Knowledge graph (kg.pkl) and audit files
│   ├── vectorstore/       Search index (not tracked in git, rebuild locally)
│   └── sessions/          Student session checkpoints (not tracked in git)
├── models/                GGUF model file (not tracked in git, download separately)
├── src/
│   ├── config.py          All paths and model settings, read from .env
│   ├── ingestion/         PDF extraction and text chunking
│   ├── graph/             Triple extraction, graph construction, cycle removal
│   ├── retrieval/         Hybrid retriever, search index, path tracker
│   ├── agents/            Instructor agent and diagnostic agent
│   └── ui/                Gradio web interface
├── scripts/               Pipeline runners and checks
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.11
- 8 GB RAM (see [Hardware Requirements](#hardware-requirements))
- About 8 GB free disk space (model, dependencies, and data)
- Windows, Linux, or macOS

### 1. Clone the repository

```bash
git clone https://github.com/Swaraj-Mandre/IEEE.git
cd IEEE
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate.bat      # Windows
source .venv/bin/activate       # Linux / macOS
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download the language model

The model is not included in this repository due to size.

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/Phi-3-mini-4k-instruct-gguf', filename='Phi-3-mini-4k-instruct-q4.gguf', local_dir='models')"
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and update `MODEL_PATH` to match your downloaded model location.

### 6. Build the search index

The knowledge graph (`data/graph/kg.pkl`) is committed, so it does not need rebuilding. The index is not, so build it once:

```bash
python scripts/build_vectorstore.py
```

---

## Running the System

### Start the tutor

```bash
python src/ui/app.py
```

Open the printed local URL (usually `http://127.0.0.1:7860`) in any browser. Each browser tab is an independent student session.

### Verify nothing is broken

Run this after any change. It takes under a minute and loads no language model.

```bash
python scripts/run_checks.py
```

### Rebuilding the pipeline from scratch

Only needed when adding chapters or changing source data. Put the new PDFs in `data/raw_pdfs/`, add the chunk file to `CHUNK_FILES` in `src/config.py`, then:

```bash
python scripts/run_ingestion.py        # PDFs to text chunks
python scripts/run_kg.py               # Build the knowledge graph (~17 min on CPU)
python scripts/build_vectorstore.py    # Rebuild the search index
python scripts/run_checks.py           # Confirm nothing regressed
python scripts/run_agent_test.py       # End-to-end smoke test, loads the model
```

Official PDF source: [ncert.nic.in/textbook.php](https://ncert.nic.in/textbook.php)

---

## Knowledge Graph Stats

Built from Chapters 4 and 6, using the current pipeline:

| Metric | Value |
|---|---|
| Source chunks | 301 (147 from Ch. 4, 154 from Ch. 6) |
| Concepts (nodes) | 175 |
| Prerequisite relationships (edges) | 197 |
| High-confidence edges (2+ supporting chunks) | 43 |
| Valid DAG (cycle-free) | Yes |
| Most connected concept | force (degree 24) |
| Average degree | 2.25 |
| Root concepts (no prerequisites) | 69 |
| Graph file size | 24 KB |
| Index size on disk | 0.63 MB |
| Median retrieval time | ~18 ms |

Strongest extracted relationships, by number of supporting chunks:

| Prerequisite | Concept | Chunks |
|---|---|---|
| velocity | acceleration | 28 |
| force | acceleration | 16 |
| mass | force | 14 |
| motion | force | 9 |
| velocity | velocity-time graph | 7 |

The full edge list is in `data/graph/kg_audit.json`; a 20-edge sample for manual review is in `data/graph/kg_sample_review.json`.

---

## Known Limitations

- **Two chapters only.** Full syllabus coverage is planned but not done.
- **Coverage gaps follow the textbook.** `inertia` appears in only one chunk across both chapters, so it never became a graph node, and a question about it lands on a neighbouring concept. This is a source-coverage limit, not a pipeline bug; adding Chapter 5 would fix it.
- **Singular and plural are separate concepts.** `surface` and `surfaces` are distinct nodes. Merging them needs word stemming, which is not in yet.
- **The diagnostic agent is rule-based.** It matches phrases like "I don't understand" and "makes sense" rather than judging comprehension. It is fast, predictable and free, but it can be fooled by an unusual reply, which it labels `unclear` and handles by re-explaining.
- **English only.** Multilingual support is future work.
- **No quantitative evaluation yet.** Validation to date is functional: 46 automated checks plus manual review of the graph.
- **CPU-only by default.** GPU acceleration needs a CUDA-enabled `llama-cpp-python` build. CPU works everywhere but the graph build takes about 17 minutes instead of a few.
- **Needs 8 GB RAM, not the 4 GB originally targeted.** Phi-3 Mini accounts for effectively all of it. Getting under 4 GB means a smaller model, and that trade-off has not been evaluated yet.

---

## Team

| Member | Role |
|---|---|
| Swaraj Mandre | AI Systems Developer and SLM Engineer |
| Siddhant Pawar | Agent Systems and Evaluation Developer |
| Yash Patil | Data Pipeline Engineer and Technical Writer |

Project mentor: Prof. Bhagyashri Thorat

Built as part of an IEEE TechForGood internship project, MIT School of Computing, MIT-ADT University.

---

## License

This project uses NCERT textbook content, which is publicly available educational material from the National Council of Educational Research and Training, Government of India. The codebase is intended for academic and research purposes as part of an IEEE-track internship submission.

# Vidhya-Setu

**A Localized Multi-Agent Framework Using GraphRAG for Adaptive STEM Learning in Low-Resource Environments**

Vidhya-Setu is a fully offline intelligent tutoring system designed for rural and under-resourced Indian schools. It runs entirely on local hardware with no internet connection, no cloud APIs, and no subscription cost. The system reads NCERT Class 9 Science textbooks, automatically builds a concept-prerequisite knowledge graph, and uses a two-agent AI architecture to deliver adaptive, step-by-step explanations that adjust in real time based on whether a student understands or is confused.

This project was built as part of an IEEE TechForGood internship initiative, targeting deployment on machines with as little as 4 GB RAM and no GPU.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Current Project Status](#current-project-status)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the System](#running-the-system)
- [Knowledge Graph Stats](#knowledge-graph-stats)
- [Known Limitations](#known-limitations)
- [Team](#team)
- [License](#license)

---

## Why This Project Exists

Students in rural India studying NCERT Science have no access to personalized academic support outside the classroom. Existing AI tutoring tools require internet connectivity, cloud subscriptions, or modern hardware that rural schools simply do not have. Vidhya-Setu is built around a different assumption: the student-facing machine might be an old donated desktop with 4 GB of RAM and no internet, and the system still has to work.

## How It Works

When a student asks a question, the system does not just search for matching text. It identifies where that question fits within a structured map of prerequisite concepts built from the textbook itself, then walks the student through that map step by step. If the student shows signs of confusion, the system automatically steps back to a simpler, foundational concept before trying again, rather than repeating the same explanation or pushing forward regardless.

## System Architecture

The system is split into two environments by design.

**Build environment** (developer machine, run once): reads NCERT PDFs, extracts concept-prerequisite pairs using a local language model, builds a directed knowledge graph, and creates a vector index of the textbook content.

**Deployment environment** (school hardware, runs every session): loads the pre-built knowledge graph and vector index, and uses a quantized language model to retrieve the right concept and generate explanations, entirely offline.

```
NCERT PDFs
    |
    v
PDF Ingestion (PyMuPDF) --> Text Chunks
    |
    v
Knowledge Graph Construction (Phi-3 Mini SLM) --> Concept-Prerequisite Graph (NetworkX)
    |
    v
Vector Embedding (sentence-transformers) --> ChromaDB Vector Store
    |
    v
GraphRAG Retrieval (graph traversal + vector search fusion)
    |
    v
Path Tracker (prerequisite chain + backtracking logic)
    |
    v
LangGraph Multi-Agent Orchestrator
    |-- Instructor Agent (generates step-by-step explanations)
    |-- Diagnostic Agent (scores understanding, triggers backtrack)
    |
    v
Gradio Web Interface (student-facing, runs in any local browser)
```

A full diagram is available in `paper/figures/Figure_3_1_System_Architecture.png`.

---

## Current Project Status

This project is under active development. The table below reflects what is actually working today, not the full target scope.

| Component | Status |
|---|---|
| PDF ingestion pipeline | Working |
| Knowledge graph construction | Working — valid DAG, cycle-free |
| GraphRAG retrieval (graph + vector fusion) | Working |
| Path tracker with backtracking | Working |
| Instructor Agent | Working |
| Diagnostic Agent | Working |
| LangGraph orchestrator | Working |
| Gradio web UI | Working |
| Full automated evaluation (BLEU/ROUGE across full benchmark) | In progress |
| Multilingual support | Not started — planned future work |
| Mobile deployment | Out of scope for this version |

Currently validated on two NCERT Class 9 Science chapters: **Chapter 4 — Describing Motion Around Us** and **Chapter 6 — How Forces Affect Motion**. Expansion to the full 13-chapter syllabus is planned.

---

## Tech Stack

All components are open-source and run entirely offline.

| Layer | Technology |
|---|---|
| Language model | Phi-3 Mini 3.8B, Q4_K_M quantization (GGUF, via llama-cpp-python) |
| Knowledge graph | NetworkX (directed graph) |
| Vector store | ChromaDB (embedded mode) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Agent orchestration | LangGraph |
| PDF processing | PyMuPDF |
| Web interface | Gradio |
| Language | Python 3.11 |

---

## Project Structure

```
vidhya-setu/
├── data/
│   ├── raw_pdfs/          NCERT source PDFs (not tracked in git)
│   ├── chunks/            Extracted and chunked text (JSON)
│   ├── graph/             Knowledge graph (kg.pkl) and audit files
│   ├── vectorstore/       ChromaDB index (not tracked in git, rebuild locally)
│   └── sessions/          Saved student session checkpoints
├── models/                 GGUF model files (not tracked in git, download separately)
├── src/
│   ├── ingestion/          PDF extraction and text chunking
│   ├── graph/              Knowledge graph construction from SLM extraction
│   ├── retrieval/           GraphRAG retriever and path tracker
│   ├── agents/              Instructor Agent, Diagnostic Agent, LangGraph orchestrator
│   └── ui/                  Gradio web interface
├── scripts/                 Pipeline runner scripts (run_ingestion.py, run_kg.py, etc.)
├── paper/                   IEEE paper drafts and figures
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.11
- Approximately 8 GB free disk space (model, dependencies, and data)
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

The model is not included in this repository due to size. Download it directly:

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='microsoft/Phi-3-mini-4k-instruct-gguf',
    filename='Phi-3-mini-4k-instruct-q4.gguf',
    local_dir='models',
)
"
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and update `MODEL_PATH` to match your downloaded model location.

### 6. Add source PDFs

Place NCERT Class 9 Science chapter PDFs into `data/raw_pdfs/`. Official source: [ncert.nic.in/textbook.php](https://ncert.nic.in/textbook.php)

---

## Running the System

The pre-built knowledge graph (`data/graph/kg.pkl`) is included in this repository, so you do not need to rebuild it to run the demo. The vector store must be rebuilt locally on first run, since it is excluded from version control.

### Quick start — run the tutor

```bash
python src/ui/app.py
```

Open the printed local URL (typically `http://127.0.0.1:7860`) in any browser.

### Rebuilding the pipeline from scratch

Only needed if you are adding new chapters or changing source data.

```bash
python scripts/run_ingestion.py        # Step 1: PDF to text chunks
python scripts/run_kg.py               # Step 2: Build knowledge graph
python scripts/run_retreival_test.py   # Step 3: Validate retrieval
python scripts/run_agent_test.py       # Step 4: Validate agent pipeline
```

---

## Knowledge Graph Stats

Current validated knowledge graph, built from Chapters 4 and 6:

| Metric | Value |
|---|---|
| Total concepts (nodes) | 797 |
| Total relationships (edges) | 754 |
| Valid DAG (cycle-free) | Yes |
| Most connected concept | velocity |
| Source chapters | Ch. 4 — Describing Motion, Ch. 6 — How Forces Affect Motion |

---

## Known Limitations

- Currently validated on two chapters only; full syllabus coverage is in progress.
- English medium only. Multilingual support is identified as future work.
- Mathematics chapters are excluded due to equation rendering issues in NCERT PDFs.
- Evaluation against quantitative NLP metrics (BLEU, ROUGE-L) on the full benchmark is still in progress; current validation is primarily functional and qualitative.
- GPU acceleration for the build pipeline is dependent on a correctly configured CUDA-enabled `llama-cpp-python` build; CPU fallback works but is significantly slower for the knowledge graph construction step.

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

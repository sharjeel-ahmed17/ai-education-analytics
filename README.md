# EduCopilot — AI Co-Pilot for Education Analytics

A production-grade decision support platform combining deep predictive modeling with Human-in-the-Loop controls. Identifies at-risk students, explains predictions via SHAP, and generates intervention reports — all with an analyst review gate.

## Features

- **Multimodal Data Ingestion** — CSV (grades, attendance), PDF (report cards), TXT (counselor notes)
- **Deep Learning Risk Prediction** — Keras ANN trained on 9 student features, exported to ONNX for inference
- **Explainable AI (XAI)** — Per-student SHAP feature attributions and global importance rankings
- **LLM Reasoning & RAG** — LangChain chains synthesize quantitative predictions with qualitative notes via Qdrant vector search
- **Human-in-the-Loop** — Review queue for analysts to Approve, Reject, or Modify AI recommendations with a full audit trail
- **Report Generation** — Export finalized reports as PDF (ReportLab) or Word DOCX (python-docx)
- **Streamlit Dashboard** — 5 interactive pages for the full workflow

## Architecture

```
Data Ingestion (CSV/PDF/TXT)
        │
        ├─► Tabular Preprocessor ──► ANN Model ──► SHAP Explainer
        │                                │
        ├─► Text Loader ──► Embeddings ──► Qdrant Vector DB
        │                                         │
        ▼                                         ▼
    Relational DB (SQLite/PostgreSQL)     LangChain RAG
                                              │
                                              ▼
                                     HITL Review Queue
                                              │
                                              ▼
                                    Report Export (PDF/DOCX)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML | TensorFlow/Keras, ONNX, scikit-learn |
| XAI | SHAP KernelExplainer |
| LLM | LangChain, OpenAI GPT-4o-mini / Groq Llama-3.1 |
| Embeddings | Cohere / OpenAI (fallback: mock encoder) |
| Vector Store | Qdrant (in-memory or remote) |
| Database | SQLAlchemy + SQLite (dev) / PostgreSQL (prod) |
| Reports | ReportLab (PDF), python-docx (DOCX) |
| Language | Python 3.12+ |

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone <repo-url>
cd ai-education
uv sync
```

### Configuration

Copy `.env.example` to `.env` and configure API keys (all are optional — the app falls back to mock/offline modes):

```env
# Database (defaults to SQLite if not set)
DATABASE_URL=

# LLM Providers
OPENAI_API_KEY=
COHERE_API_KEY=
GROQ_API_KEY=

# Vector Store
QDRANT_URL=
QDRANT_API_KEY=
```

### Run

```bash
streamlit run src/edu_copilot/app.py
```

### Seed Sample Data

```bash
uv run python scripts/seed_sample_data.py
```

## Project Structure

```
src/edu_copilot/
├── app.py                  # Streamlit entry point
├── config.py               # Pydantic settings (env vars)
├── data/
│   ├── schemas.py          # Pydantic data models
│   ├── loaders/            # CSV, PDF, TXT loaders
│   └── preprocessing/      # Feature preprocessing
├── db/
│   ├── models.py           # SQLAlchemy ORM models
│   └── session.py          # DB session management
├── models/
│   ├── ann_model.py        # Keras ANN definition
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   └── export_onnx.py      # ONNX export
├── xai/
│   ├── shap_explainer.py   # SHAP explanations
│   ├── confidence.py       # Confidence scoring
│   └── feature_importance.py
├── llm/
│   ├── providers.py        # LLM provider abstraction
│   ├── embeddings.py       # Embedding models
│   ├── chains/             # LangChain chains
│   └── prompts/            # Prompt templates
├── vectorstore/
│   └── qdrant_client.py    # Qdrant vector DB client
├── hitl/
│   ├── review_queue.py     # HITL review logic
│   └── workflow.py         # Approval workflow
├── reports/
│   ├── pdf_report.py       # ReportLab PDF export
│   ├── docx_report.py      # python-docx export
│   └── report_data.py      # Report data model
└── ui/pages/
    ├── 1_upload_data.py
    ├── 2_predictions.py
    ├── 3_hitl_review.py
    ├── 4_explainability.py
    └── 5_reports.py
```

## Tests

```bash
uv run pytest
```

- Unit: `tests/unit/` (ANN model, HITL workflow, XAI)
- Integration: `tests/integration/` (end-to-end pipeline)

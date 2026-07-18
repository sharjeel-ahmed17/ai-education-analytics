# System Architecture: AI Co-Pilot for Education Analytics

This document details the production-grade architecture of the AI Co-Pilot, showing the relationships between data ingestion, predictive modeling, database storage, and LLM reasoning.

## 1. Component Diagram

```mermaid
graph TD
    subgraph Client Layer
        UI[Streamlit Web UI]
    end

    subgraph Data Ingestion & Preprocessing
        CSV[Student Tabular CSV] --> TabPrep[Tabular Preprocessor]
        PDF[Student PDFs] --> PDFLoad[pypdf Loader] --> TextSplit[Text Preprocessor]
        TXT[Teacher Notes TXT] --> TextLoad[Text Loader] --> TextSplit
    end

    subgraph Storage Layer
        DB[(Relational DB: PostgreSQL / SQLite)]
        QDRANT[(Qdrant Vector Database)]
    end

    subgraph Prediction & XAI
        ANN[ANN Model: Keras / ONNX]
        SHAP[SHAP Explainer]
        CONF[Confidence Calculator]
    end

    subgraph LLM & RAG Orchestration
        LC[LangChain LLM Chains]
        EMB[Cohere/OpenAI Embeddings]
    end

    subgraph Document Generation
        REP[Report Exporters: reportlab & python-docx]
    end

    %% Data Flows
    TabPrep -->|Clean Features| ANN
    TabPrep -->|Save Records| DB
    TextSplit -->|Embeddings via EMB| QDRANT
    
    ANN -->|Probability| CONF
    ANN -->|Attributions| SHAP
    CONF -->|Scores| DB
    SHAP -->|Values| DB
    
    QDRANT -->|Retrieved Chunks| LC
    ANN -->|Model outputs| LC
    LC -->|Report Drafts| UI
    
    UI -->|HITL Approval/Edit| DB
    DB -->|Approved Reports| REP
    REP -->|Download PDF/DOCX| UI
```

## 2. Subsystem Definitions

### Data Modalities & Loaders
- **Tabular Data**: Student demographic and quantitative grades preprocessed using standard scaling and categorical encoding. Saves baseline entries in the Relational DB.
- **PDF & Unstructured Text**: Parsed via `pypdf` or direct streams, split into overlapping chunks, and indexed in Qdrant with student-level metadata tags.

### ML & Explainability Subsystem
- **ANN Predictor**: Keras dense network with BatchNormalization and Dropout predicting classification probabilities. Running via `.keras` or `.onnx` runtimes.
- **SHAP (SHapley Additive exPlanations)**: KernelExplainer using a synthesized k-means reference background to map feature-level contributions for single student predictions.
- **Confidence Metrics**: Normalized certainty score mapped using distance from the threshold boundary.

### Database Layer
- **Relational DB**: SQLAlchemy schema storing `students`, `student_tabular_data`, `prediction_records`, `review_records` (HITL audit logs), and `report_records`.
- **Vector Database**: Qdrant vector database (running in-memory locally) holding chunked notes mapped to `student_id` fields.

### Orchestration & Document Output
- **LangChain Chains**: Orchestrates summarization of qualitative files, reasoning over quantitative predictions, and final document drafting.
- **Exporters**: Converts structured Pydantic objects into ReportLab tables (PDF) and python-docx elements (Word).

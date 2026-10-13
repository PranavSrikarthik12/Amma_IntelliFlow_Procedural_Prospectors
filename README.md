# Amma IntelliFlow — Supply Chain Intelligence

> A multi-agent AI platform that analyzes invoice and disbursement data to detect anomalies, explain root causes, simulate what-if scenarios, and recommend optimizations in real time.

**Team:** Procedural Prospectors · **Event:** Techathon 2.0

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-green)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [The Agents](#the-agents)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Configuration](#configuration)
9. [Usage Guide](#usage-guide)
10. [API Reference](#api-reference)
11. [Sample Data](#sample-data)
12. [Deployment](#deployment)
13. [Limitations and Roadmap](#limitations-and-roadmap)
14. [Contributing](#contributing)

---

## Overview

Small and medium suppliers often wait on invoice approvals and fund disbursements without knowing *why* payments stall. **Amma IntelliFlow** turns raw CSV exports from ERP and payment systems into answers: what is going wrong, why, what to do about it, and how confident the system is in that advice.

Users upload an **invoice report** or a **disbursement report**. A pipeline of cooperating agents parses the data, extracts key metrics, flags anomalies, and indexes a text summary into a vector store. Users can then ask natural-language questions in a chat interface. Each question flows through a chain of specialised agents (intent classification, retrieval, root-cause reasoning, counterfactual simulation, action recommendation, and a decision-confidence guardrail), and every step is logged for full transparency.

The platform is framed around two report types:

| Report type    | What it covers                                                              | Headline metric          |
| -------------- | --------------------------------------------------------------------------- | ------------------------ |
| `invoice`      | Supplier invoice payment flows and approval/rejection behaviour             | Invoice rejection rate   |
| `disbursement` | Supplier fund disbursement timelines and cash-flow bottlenecks              | Disbursement delay (days) |

The legacy values `authorization` and `settlement` are still accepted by the upload endpoint and are silently remapped to `invoice` and `disbursement`.

---

## Key Features

- **CSV report ingestion** with automatic type conversion. Text values such as `"$1,500.50"`, `"5%"` or `"3.5 hours"` are parsed into numbers, and date columns are detected automatically.
- **Automatic metric extraction.** Mean, min, max, sum and count are computed for every numeric column, then mapped onto standardised metric names.
- **Anomaly detection** with severity levels (`high`, `medium`, `low`).
- **Retrieval-Augmented Generation (RAG).** Report summaries are embedded with `all-MiniLM-L6-v2` and stored in a FAISS index; questions retrieve the most relevant context.
- **Intent-aware multi-agent chat pipeline** with root-cause analysis, recommended actions and priorities.
- **Counterfactual simulation** that proposes directional "what if we changed X" scenarios with impact, risk and confidence.
- **Decision-confidence guardrail** that scores each recommendation (0–1), assigns a risk level and flags when human review is required before acting.
- **Full agent audit trail.** Every agent call is logged with timestamp, input, output and metadata, and is viewable in the UI.
- **Works without an LLM key.** If `OPENAI_API_KEY` is absent or the call fails, the reasoning agent falls back to deterministic, intent-based logic, so the demo never breaks.
- **Interactive dashboard** with Chart.js visualisations and dynamic alert thresholds.
- **Docker-ready** for one-command deployment.

---

## System Architecture

```
                         ┌────────────────────────────┐
   CSV upload ─────────▶ │  1. Report Understanding   │
 (invoice / disbursement)│     (pandas metric extract)│
                         └──────────────┬─────────────┘
                                        │ summary
                         ┌──────────────▼─────────────┐
                         │  2. Anomaly Detection      │──▶ alerts / upload response
                         └──────────────┬─────────────┘
                                        │
                         ┌──────────────▼─────────────┐
                         │  RAG: embed summary        │
                         │  (MiniLM → FAISS index)    │
                         └──────────────┬─────────────┘
                                        │
   ───────────────────────  indexed knowledge  ───────────────────────
                                        │
   Chat query ──▶ 0. Intent Classification
                         │
                         ▼
                  RAG Retrieval (top-k context)
                         │
                         ▼
                  3. Root Cause Reasoning  (OpenAI or deterministic fallback)
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
   5. Counterfactual Simulation   4. Action Recommendation
              │                      │
              └──────────┬───────────┘
                         ▼
              6. Decision Confidence Guardrail
                         │
                         ▼
        Response: intent · analysis · actions · counterfactual · confidence
                         │
                         ▼
              Agent Logger (audit trail → /agents page)
```

Two pipelines run on the backend:

1. **Upload pipeline** (`POST /upload`): validate → save → Report Understanding → Anomaly Detection → cache summary → embed into FAISS.
2. **Chat pipeline** (`POST /chat`): classify intent → retrieve context → root cause → counterfactual → recommendations → confidence assessment.

---

## The Agents

| #   | Agent                          | File                                      | Responsibility |
| --- | ------------------------------ | ----------------------------------------- | -------------- |
| 0   | Intent Classification          | `agents/intent_classifier.py`             | Keyword-based routing of a query into an intent such as `AUTH_TRENDS`, `SETTLEMENT_ANALYSIS`, `PERFORMANCE_REPORT`, `OPTIMIZATION_RECOMMENDATION` or `GENERAL`. |
| 1   | Report Understanding           | `agents/report_understanding.py`          | Loads the CSV, parses dates, converts text-numeric columns, computes per-column statistics, maps columns to standard metrics, and builds a text summary for embedding. |
| 2   | Anomaly Detection              | `agents/anomaly_detection.py`             | Applies thresholds to extracted metrics and emits anomalies with type, severity and message. |
| 3   | Root Cause Reasoning           | `agents/root_cause_reasoning.py`          | Uses OpenAI (`gpt-4.1-mini`, temperature 0.2) with retrieved context; falls back to deterministic intent-based reasoning when no key is set. |
| 4   | Action Recommendation          | `agents/action_recommendation.py`         | Turns the root-cause explanation into actions with impact, priority, owner and confidence. |
| 5   | Counterfactual Simulation      | `agents/counterfactual_simulation.py`     | Produces explainable "what-if" scenarios with impact, risk and confidence. Directional only; it does not predict the future. |
| 6   | Decision Confidence Guardrail  | `agents/decision_confidence.py`           | Multi-factor scoring (intent stability, anomaly strength, root-cause depth, counterfactual support, historical accuracy) mapped to `HIGH` / `MEDIUM-HIGH` / `MEDIUM` / `MEDIUM-LOW` / `LOW`, plus escalation to human review. |
| –   | Agent Logger                   | `agents/agent_logger.py`                  | In-memory audit log of every agent step (last 50 entries served). |
| –   | RAG Embed / Retrieve           | `rag/embed.py`, `rag/retrieve.py`         | Tagged embeddings in a FAISS `IndexFlatL2` (384 dimensions) and top-k retrieval. |

### Confidence scoring bands

| Score        | Level         | Human review | Guidance                                              |
| ------------ | ------------- | ------------ | ----------------------------------------------------- |
| ≥ 0.80       | `HIGH`        | No           | Suitable for automated execution with monitoring      |
| 0.65 – 0.79  | `MEDIUM-HIGH` | No           | Proceed with enhanced monitoring and post-review      |
| 0.50 – 0.64  | `MEDIUM`      | Yes          | Human review recommended before execution             |
| 0.35 – 0.49  | `MEDIUM-LOW`  | Yes          | Human approval required                               |
| < 0.35       | `LOW`         | Yes          | Manual investigation required; do not automate        |

---

## Tech Stack

**Backend**
- Python 3.11, Flask 3.0, Flask-CORS
- pandas, NumPy, scikit-learn
- sentence-transformers (`all-MiniLM-L6-v2`), PyTorch, FAISS (CPU)
- OpenAI Python SDK (optional, for LLM reasoning)
- python-dotenv, gunicorn

**Frontend**
- Server-rendered Jinja2 templates (HTML/CSS/JavaScript)
- Chart.js for charts
- Bootstrap 5 and Bootstrap Icons on selected pages

**Infrastructure**
- Docker (`python:3.11` base image)

---

## Project Structure

```
Amma_IntelliFlow_Procedural_Prospectors/
├── backend/
│   ├── app.py                         # Flask app, routes, upload + chat pipelines
│   ├── agents/
│   │   ├── intent_classifier.py
│   │   ├── report_understanding.py
│   │   ├── anomaly_detection.py
│   │   ├── root_cause_reasoning.py
│   │   ├── action_recommendation.py
│   │   ├── counterfactual_simulation.py
│   │   ├── decision_confidence.py
│   │   └── agent_logger.py
│   ├── rag/
│   │   ├── embed.py                   # SentenceTransformer + FAISS indexing
│   │   └── retrieve.py                # top-k context retrieval
│   └── utils/
│       ├── helpers.py
│       └── metrics.py
├── frontend/
│   └── templates/
│       ├── index.html                 # Landing page (problem, agents, pipeline)
│       ├── upload.html                # Report upload
│       ├── dashboard.html             # Charts + chat interface
│       ├── agents.html                # Agent workflow / audit log
│       ├── alerts.html                # Alerts view
│       └── settings.html              # Settings
├── synthetic_erp_invoices.csv         # Sample dataset (200 invoices)
├── Dockerfile
├── requirements.txt
├── runtime.txt                        # python-3.11.5
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python **3.11**
- `pip`
- (Optional) An OpenAI API key for LLM-powered root-cause reasoning
- (Optional) Docker
- Internet access on first run so `all-MiniLM-L6-v2` can be downloaded from Hugging Face

### 1. Clone the repository

```bash
git clone https://github.com/PranavSrikarthik12/Amma_IntelliFlow_Procedural_Prospectors.git
cd Amma_IntelliFlow_Procedural_Prospectors
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` pins the full PyTorch and CUDA stack, so the install is large. On a CPU-only machine you can install a CPU build of PyTorch first to save space.

### 4. Configure environment variables

Create a `.env` file in the project root (see [Configuration](#configuration)).

### 5. Run the app

```bash
python backend/app.py
```

The server starts on **http://localhost:7860** by default.

> Run the command from the project root as shown. `app.py` imports `agents` and `rag` as top-level packages, so it expects to be launched as `backend/app.py`.

### Run with Docker

```bash
docker build -t amma-intelliflow .
docker run -p 7860:7860 --env-file .env amma-intelliflow
```

---

## Configuration

| Variable          | Required | Default | Description |
| ----------------- | -------- | ------- | ----------- |
| `OPENAI_API_KEY`  | No       | –       | Enables LLM-based root-cause reasoning. Without it, deterministic fallback logic is used. |
| `PORT`            | No       | `7860`  | Port the Flask server listens on. |

Example `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
PORT=7860
```

`.env` is git-ignored. Never commit secrets.

---

## Usage Guide

1. **Open the landing page** at `/` for an overview of the problem, the agents and the pipeline.
2. **Upload a report** at `/upload`. Choose the report type (`invoice` or `disbursement`) and select a `.csv` file. The sample file `synthetic_erp_invoices.csv` works well for the `invoice` type.
3. **Review anomalies** returned after upload and on `/alerts`.
4. **Open the dashboard** at `/dashboard` to see rejection-rate and delay charts. With no uploads yet, illustrative sample data is shown.
5. **Ask questions** in the dashboard chat, for example:
   - *"Why are invoice rejections rising?"*
   - *"What is causing disbursement delays?"*
   - *"Recommend optimizations for supplier payments."*
6. **Inspect the agent trail** at `/agents` to see every agent's input and output for each request.

---

## API Reference

### Pages

| Route        | Description              |
| ------------ | ------------------------ |
| `GET /`      | Landing page             |
| `GET /dashboard` | Dashboard and chat   |
| `GET /upload`    | Upload page          |
| `GET /agents`    | Agent workflow view  |
| `GET /alerts`    | Alerts               |
| `GET /settings`  | Settings             |

### `POST /upload`

Upload a CSV report. Content type: `multipart/form-data`.

| Field  | Type   | Description |
| ------ | ------ | ----------- |
| `type` | string | `invoice` or `disbursement` (also accepts `authorization` / `settlement`) |
| `file` | file   | A `.csv` file |

**Response `200`**

```json
{
  "message": "Invoice report processed successfully",
  "anomalies": [
    { "type": "NO_ANOMALY", "severity": "low", "message": "No significant anomalies detected" }
  ],
  "filename": "synthetic_erp_invoices.csv",
  "report_type": "invoice",
  "cached_count": 1
}
```

Errors return `400` (missing type/file, non-CSV, invalid type) or `500` (processing failure).

### `POST /chat`

Run the agentic analysis pipeline on a natural-language question.

**Request**

```json
{
  "query": "Why are invoice rejections rising?",
  "show_counterfactual": true
}
```

**Response `200`**

```json
{
  "intent": "GENERAL",
  "analysis": "…root-cause explanation…",
  "actions": [
    { "action": "…", "impact": "…", "priority": "High", "owner": "…", "confidence": "0.85" }
  ],
  "counterfactual": {
    "scenario": "…", "impact": "…", "risk": "…", "confidence": "Medium", "generated_at": "…"
  },
  "decision_confidence": {
    "confidence_score": 0.55,
    "confidence_level": "MEDIUM",
    "human_review_required": true,
    "recommendation": "Human review recommended before execution",
    "risk_level": "LOW",
    "risk_flags": ["…"],
    "rationale": ["…"]
  }
}
```

### `GET /chart-data`

Returns data for the dashboard charts. Response keys are `authorization` (invoice rejection rate, `unit: "%"`) and `settlement` (disbursement delay, `unit: "days"`). Each contains `avg_*`, `threshold`, `data_points`, `labels`, `has_data`, `domain_label` and `unit`. Thresholds are dynamic: 1.2× the invoice average and 1.25× the disbursement average.

### `GET /agent-logs`

Returns the most recent agent execution log entries (timestamp, agent, input, output, metadata).

### `POST /clear-logs`

Clears the agent log. Useful between demos.

### `GET /debug/cache`

Shows how many reports are cached per type and which metrics each contains.

---

## Sample Data

`synthetic_erp_invoices.csv` contains **200 synthetic ERP invoices** with the following columns:

| Column           | Description |
| ---------------- | ----------- |
| `invoice_id`     | Unique invoice identifier (e.g. `INV-1001`) |
| `supplier_name`  | Supplier (e.g. Orion Logistics, Nova Supply Co., Vertex Components) |
| `invoice_amount` | Invoice value |
| `invoice_date`   | Date the invoice was issued |
| `due_date`       | Payment due date |
| `payment_date`   | Actual payment date |
| `status`         | `Paid`, `Rejected`, `Delayed`, … |
| `rejection_flag` | `1` if rejected, otherwise `0` |
| `delay_days`     | Days paid after the due date (negative means early) |
| `risk_score`     | Risk score between 0 and 1 |

---

## Deployment

The app listens on `0.0.0.0` and honours the `PORT` environment variable, so it runs on container platforms such as Hugging Face Spaces (default port 7860), Render, Railway or any Docker host. The `runtime.txt` file pins Python `3.11.5` for buildpack-based platforms. `gunicorn` is included in the requirements for production serving.

> Uploaded files are written to `backend/data/<report_type>/`, and the FAISS index, report cache and agent logs are held **in memory**. They reset when the process restarts.

---

## Limitations and Roadmap

Being upfront about where the project stands:

- **State is in-memory.** The report cache, FAISS index and agent logs are not persisted. Adding a database and a persisted vector index is a natural next step.
- **Supply-chain domain mapping is a thin layer.** The upload and chat routes inject supply-chain-finance context and remap legacy payment terms, but several agents (intent keywords, anomaly thresholds, recommendation text, counterfactual scenarios, and the root-cause prompt) still use payment-network vocabulary such as authorization, settlement and issuer. Fully native supply-chain rules and prompts are a planned improvement.
- **Heuristic components.** Intent classification is keyword-based, and counterfactual figures are illustrative templates rather than model-derived forecasts.
- **Metric fallbacks.** When expected columns are missing, report understanding falls back to proxy or default values, which should be reviewed before drawing conclusions.
- **Anomalies are not yet fed into the confidence guardrail** from the chat route, which lowers confidence scores slightly.
- **No authentication.** Intended as a demo/prototype; add authentication and rate limiting before any production use.

**Roadmap ideas:** persistent storage, native supply-chain anomaly rules, learned intent classification, historical-accuracy tracking for the guardrail, role-based access, and live ERP connectors.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a pull request

---

## Acknowledgements

Built by **Team Procedural Prospectors** for **Techathon 2.0**. Forked from [sarvesh-112/Amma_IntelliFlow_Procedural_Prospectors](https://github.com/sarvesh-112/Amma_IntelliFlow_Procedural_Prospectors).

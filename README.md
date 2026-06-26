# OCRScore

A web-based toolkit for benchmarking OCR engines at scale. Upload PDFs, run them through multiple OCR systems, inspect results down to the individual character outline, and get reproducible accuracy scores.

Supports **Tesseract**, **GCP Document AI**, **AWS Textract**, **olmOCR**, and **DeepSeek-OCR** — with a plugin system for adding more.

---

## Features

- **Multi-engine OCR** — Run the same PDF through every engine. Compare results side by side.
- **PDF viewer with overlay layers** — Toggle per-engine character outlines, word-level color coding (green = correct, red = wrong), reading order numbers.
- **Granular evaluation** — Scores at character, word, line, paragraph, and table level. CER, WER, precision/recall/F1, GriTS for tables.
- **Automatic ground truth** — Consensus Entropy builds ground truth from multiple engine outputs. Manually correct through the UI.
- **Semantic scoring** — Fluency, grammaticality, and semantic similarity metrics catch what edit distance misses.
- **Novel metrics** — Imagination Rate (hallucination detection), Confidence Calibration Error, Noise Sensitivity Index.
- **Statistical rigor** — Bootstrap confidence intervals on every metric. Know when differences are significant.
- **Reproducible runs** — Content-addressable storage, run hash dedup, immutable results, raw output preservation.
- **Full API** — REST endpoints for documents, runs, scores, ground truth, reports, batch processing, and auth.
- **Batch processing** — Upload hundreds of PDFs, run across engines, export aggregate reports.
- **Role-based auth** — JWT + API keys with admin/reviewer/viewer roles.
- **Script-aware** — Latin, CJK, Arabic, Devanagari, Thai, Hebrew, Cyrillic, and more. Handwriting detection and filtering.
- **Exportable** — CSV, JSON, and self-contained HTML reports.

---

## Quickstart

### Option A: Docker Compose (recommended)

```bash
# Start PostgreSQL, backend, and frontend
docker compose up -d

# Backend:   http://localhost:8000/api/v1/health
# Frontend:  http://localhost:5173
# API docs:  http://localhost:8000/api/v1/docs
```

### Option B: Manual setup

**Prerequisites:**
- Python 3.12+
- Node.js 18+
- PostgreSQL 16 (or use Docker for just the database: `docker compose up -d db`)
- Tesseract (optional — for the Tesseract engine)

```bash
# 1. Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Database (if not running via Docker)
# Make sure PostgreSQL is running and create the database:
# createdb ocrscore

# 3. Run migrations
alembic upgrade head

# 4. Start the backend
uvicorn backend.main:app --reload --port 8000

# 5. Frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

### First run

1. Open the frontend at http://localhost:5173
2. Upload a PDF via the **PDFs** page
3. Click **Process** and select an OCR engine
4. Once the run completes, open the **Viewer** to inspect results
5. Navigate to **Evaluation** to see per-page and aggregate scores

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│  React + TypeScript + Vite + Tailwind + PDF.js   │
│  ┌──────┐ ┌──────┐ ┌────────┐ ┌──────────────┐ │
│  │ PDF   │ │ Overlay│ │Eval   │ │ Dashboard    │ │
│  │Viewer │ │Layers  │ │Charts │ │ + Reports    │ │
│  └──────┘ └──────┘ └────────┘ └──────────────┘ │
└──────────────────┬──────────────────────────────┘
                   │ HTTP + WebSocket
┌──────────────────▼──────────────────────────────┐
│                   Backend                        │
│  FastAPI + SQLAlchemy + asyncpg + Celery-style   │
│  ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐│
│  │Routers │ │Engine  │ │Evalua-  │ │ Auth     ││
│  │(REST)  │ │Plugin  │ │tion     │ │ (JWT)    ││
│  │  + WS  │ │System  │ │Pipeline │ │          ││
│  └────────┘ └────────┘ └─────────┘ └──────────┘│
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              Storage Layer                       │
│  PostgreSQL (JSONB) + Content-Addressable FS      │
└─────────────────────────────────────────────────┘
```

### Project structure

```
OCRScore/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── database.py                # Async SQLAlchemy engine
│   ├── settings.py                # Environment config
│   ├── storage.py                 # Content-addressable file store
│   ├── mock_engine.py             # Synthetic engine for testing
│   ├── run_orchestrator.py        # Run lifecycle management
│   ├── ground_truth_manager.py    # GT versioning + CRUD
│   ├── report_generator.py        # CSV/JSON/HTML exports
│   ├── batch_processor.py         # Batch run orchestration
│   ├── websocket_manager.py       # WebSocket connection handler
│   ├── models/                    # SQLAlchemy ORM models
│   ├── schemas/                   # Pydantic v2 API schemas
│   ├── engine/                    # OCREngine ABC + registry
│   ├── engines/                   # Engine implementations
│   │   ├── tesseract.py           # Tesseract OCR
│   │   ├── gcp_document_ai.py     # GCP Document AI
│   │   ├── textract.py            # AWS Textract
│   │   ├── vlm_olmocr.py          # olmOCR
│   │   └── vlm_deepseek.py        # DeepSeek-OCR
│   ├── routers/                   # FastAPI route handlers
│   ├── evaluation/                # Scoring + metrics
│   │   ├── scoring.py             # CER/WER/F1
│   │   ├── consensus.py           # Consensus Entropy GT
│   │   ├── bootstrap.py           # Confidence intervals
│   │   ├── table_scoring.py       # GriTS metrics
│   │   ├── semantic.py            # Fluency/grammaticality
│   │   ├── novel_metrics.py       # Imagination Rate, CCE, NSI
│   │   ├── script_aware.py        # Per-script evaluation
│   │   └── handwriting.py         # Handwriting detection
│   ├── alignment/                 # Sequence + spatial alignment
│   ├── auth/                      # JWT + API key auth
│   └── alembic/                   # Database migrations
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Route definitions
│   │   ├── main.tsx               # Entry point
│   │   ├── pages/                 # Route-level pages
│   │   │   ├── Dashboard.tsx
│   │   │   ├── PdfList.tsx
│   │   │   ├── PdfViewer.tsx      # PDF.js + overlay layers
│   │   │   ├── Evaluation.tsx     # Scores + confusion matrices
│   │   │   ├── EngineList.tsx
│   │   │   └── Reports.tsx
│   │   ├── components/
│   │   │   ├── PdfViewer/         # Viewer components
│   │   │   ├── ScoreChart.tsx     # Per-page score bar chart
│   │   │   ├── ConfusionMatrix.tsx# Canvas heatmap
│   │   │   ├── GeometricHeatmap.tsx # Spatial error heatmap
│   │   │   ├── RunProgress.tsx    # Progress bar + WS status
│   │   │   ├── BatchDialog.tsx    # Batch processing modal
│   │   │   └── RunComparison.tsx  # Side-by-side engine scores
│   │   ├── lib/
│   │   │   ├── api.ts             # API client
│   │   │   ├── types.ts           # TypeScript interfaces
│   │   │   └── websocket.ts       # WebSocket hook
│   └── ...
│
├── docker-compose.yml             # Postgres + backend + frontend
├── Dockerfile
├── pyproject.toml
└── ruff.toml
```

---

## OCR Engines

| Engine | Type | Config | Status |
|--------|------|--------|--------|
| **Tesseract** | Local (pytesseract) | `lang`, `psm`, `oem`, `dpi` | ✅ Production |
| **GCP Document AI** | Cloud API | `processor_id`, `project_id`, `location` | ✅ Production |
| **AWS Textract** | Cloud API | `region`, `access_key_id`, `secret_access_key` | ✅ Production |
| **olmOCR** | VLM API | `api_url`, `model`, `api_key`, `dpi` | ✅ Experimental |
| **DeepSeek-OCR** | VLM API | `api_key`, `model`, `dpi` | ✅ Experimental |
| **Mock Engine** | Synthetic | `seed` | ✅ Testing/CI |

Adding a new engine: create a file in `backend/engines/` that extends `OCREngine` and implements `process_pdf()` + `normalize()`. Register it with `EngineRegistry.register()`.

---

## API Overview

| Category | Endpoints |
|----------|-----------|
| **Documents** | `POST/GET/DELETE /api/v1/documents` — Upload, list, soft-delete PDFs |
| **Runs** | `POST/GET/DELETE /api/v1/runs` — Create, list, cancel OCR runs |
| **Results** | `GET /api/v1/runs/{id}/results` — Per-page normalized OCR output |
| **Compare** | `GET /api/v1/runs/{id}/results/{page}/compare` — Multi-engine aligned grid |
| **Scores** | `GET /api/v1/runs/{id}/scores` — Aggregate + per-page breakdown |
| **Ground Truth** | `POST/GET/PUT/DELETE /api/v1/ground-truth` — Versioned GT CRUD |
| **Engines** | `GET /api/v1/engines` — List registered engines + config schemas |
| **Reports** | `GET /api/v1/reports/summary` — Aggregate stats, rankings, exports |
| **Batch** | `POST/GET /api/v1/batch` — Multi-PDF × multi-engine processing |
| **Comparison** | `GET /api/v1/comparison/runs` — Cross-run side-by-side scores |
| **Auth** | `POST /api/v1/auth/login` — JWT login, API key management |
| **WebSocket** | `WS /api/v1/ws/runs/{id}` — Real-time run progress |

Full OpenAPI docs at `/api/v1/docs` when the backend is running.

---

## Evaluation Metrics

| Metric | Level | Range | Description |
|--------|-------|-------|-------------|
| **CER** | Character | 0–1 | Character Error Rate (edit distance / reference length) |
| **WER** | Word | 0–1 | Word Error Rate with I/D/S breakdown |
| **Precision / Recall / F1** | Char + Word | 0–1 | Per-character and per-word classification metrics |
| **GriTS_Top** | Table | 0–1 | Grid Table Similarity — topological structure accuracy |
| **GriTS_Con** | Table | 0–1 | Grid Table Similarity — cell content accuracy |
| **GriTS_Loc** | Table | 0–1 | Grid Table Similarity — cell bounding box IoU |
| **Semantic Plausibility** | Document | 0–1 | Fluency + grammaticality composite (no reference needed) |
| **Semantic Similarity** | Document | 0–1 | N-gram cosine similarity against reference |
| **Imagination Rate** | Word | 0–1 | Fraction of OCR words not in reference (hallucination) |
| **CCE** | Word | 0–1 | Confidence Calibration Error (Brier score) |
| **NSI** | Document | — | Noise Sensitivity Index (CER degradation across DPI) |
| **Bootstrap CI** | All | 95% | Confidence intervals via percentile bootstrap |

All metrics include 95% bootstrap confidence intervals and per-page breakdowns.

---

## Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest backend/test_integration.py         # End-to-end pipeline
pytest backend/evaluation/test_scoring.py  # CER/WER/scoring
pytest backend/alignment/test_aligner.py   # Sequence alignment
pytest backend/alignment/test_clustering.py# Spatial clustering

# Code quality
ruff check backend/
basedpyright backend/
```

**751 tests** across all modules. CI-ready with a mock engine that requires zero external dependencies.

---

## Configuration

Configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://ocrscore:ocrscore@localhost:5432/ocrscore` | PostgreSQL connection |
| `STORAGE_PATH` | `./store` | Content-addressable file storage root |
| `JWT_SECRET` | (auto-generated) | JWT signing key |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |

---

## License

MIT

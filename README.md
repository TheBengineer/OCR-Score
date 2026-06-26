# OCRScore

A web-based evaluation toolkit for benchmarking OCR engines — GCP Document AI, AWS Textract, Tesseract, and more.

Upload PDFs, run them through multiple OCR systems in parallel, inspect the results side by side down to the individual character outline, and generate reproducible, granular accuracy scores.

## Goals

- **Pluggable OCR modules** — Add new engines with minimal friction. Each engine's raw output is preserved, and a normalizer converts it to a shared schema for comparison.
- **Granular evaluation** — Score OCR accuracy from the character level up through words, lines, paragraphs, and table structure.
- **Consensus-based ground truth** — Align outputs across engines to build ground truth automatically, then refine it through the UI.
- **Reproducibility** — Every run's config, raw outputs, and computed scores are stored and traceable.
- **Rich visual debugging** — View the original PDF with toggleable overlay layers per OCR engine, showing bounding boxes, character outlines, and confidence.

## Roadmap

### Phase 1 — Foundation
Project scaffold, data model, database schema, API skeleton, PDF upload and storage.

### Phase 2 — OCR Engine Integration
Plugin interface definition, engine modules for Tesseract, GCP Document AI, and AWS Textract, output normalization.

### Phase 3 — Evaluation Pipeline
Sequence alignment, character/word/table-level scoring, semantic grammaticality scoring, combined weighted score.

### Phase 4 — Frontend Viewer
React app shell, PDF.js-based document viewer with SVG overlay system (toggleable layers, character outlines, zoom/pan).

### Phase 5 — Ground Truth & Consensus
Consensus builder from multi-engine alignment, ground truth editing UI, manual correction workflow.

### Phase 6 — Reports & Dashboard
Aggregate rankings across all PDFs, per-engine score breakdowns, exportable reports (CSV/JSON/HTML), confusion matrices.

---

*Built with FastAPI, React, Python, and PostgreSQL.*

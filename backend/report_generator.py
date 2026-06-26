"""Report generation — summary statistics, engine rankings, and CSV/JSON/HTML export.

All functions accept an ``AsyncSession`` from SQLAlchemy and perform their own
queries against the existing OCR data models.

.. note::
   Generated reports are **not persisted** — they are created on demand and
   served directly to the client.
"""

from __future__ import annotations

import csv
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from backend.models.engine import OCREngine
from backend.models.enums import RunStatus
from backend.models.page_result import PageResult
from backend.models.pdf import PDF
from backend.models.run import OCRRun

# ── Public API ─────────────────────────────────────────────────────────────────


async def compute_summary_statistics(db: AsyncSession) -> dict[str, Any]:
    """Aggregate summary statistics across all completed runs.

    Returns
    -------
    dict
        ``{
            "total_pdfs": …,
            "total_runs": …,
            "completed_runs": …,
            "avg_cer": …,
            "avg_wer": …,
            "best_engine": {"id": …, "avg_cer": …} | None,
            "pages_evaluated": …,
        }``
    """
    # ── Simple counts ────────────────────────────────────────────────────
    pdf_count = (
        await db.execute(select(func.count(PDF.id)).where(PDF.deleted_at.is_(None)))
    ).scalar() or 0

    total_runs = (await db.execute(select(func.count(OCRRun.id)))).scalar() or 0

    completed_runs = (
        await db.execute(
            select(func.count(OCRRun.id)).where(OCRRun.status == RunStatus.COMPLETED),
        )
    ).scalar() or 0

    pages_eval = (
        await db.execute(
            select(func.count(PageResult.id))
            .join(OCRRun)
            .where(OCRRun.status == RunStatus.COMPLETED),
        )
    ).scalar() or 0

    # ── Score summaries for completed runs ────────────────────────────────
    result = await db.execute(
        select(OCRRun)
        .options(joinedload(OCRRun.score_summaries), joinedload(OCRRun.engine))
        .where(OCRRun.status == RunStatus.COMPLETED),
    )
    runs = list(result.scalars().unique().all())

    cer_values: list[float] = []
    wer_values: list[float] = []
    engine_cers: dict[str, list[float]] = {}

    for run in runs:
        engine_slug = run.engine.slug if run.engine else str(run.engine_id)
        for ss in run.score_summaries:
            cer = _safe_float(ss.breakdown, ("character", "cer"))
            wer = _safe_float(ss.breakdown, ("word", "wer"))
            if cer is not None:
                cer_values.append(cer)
                engine_cers.setdefault(engine_slug, []).append(cer)
            if wer is not None:
                wer_values.append(wer)

    avg_cer = _avg(cer_values)
    avg_wer = _avg(wer_values)

    best_engine: dict[str, Any] | None = None
    if engine_cers:
        best_slug = min(engine_cers, key=lambda s: _avg(engine_cers[s]))
        best_engine = {
            "id": best_slug,
            "avg_cer": round(_avg(engine_cers[best_slug]), 4),
        }

    return {
        "total_pdfs": pdf_count,
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "avg_cer": round(avg_cer, 4),
        "avg_wer": round(avg_wer, 4),
        "best_engine": best_engine,
        "pages_evaluated": pages_eval,
    }


async def aggregate_engine_rankings(db: AsyncSession) -> list[dict[str, Any]]:
    """Rank engines by their average CER across all completed runs.

    Returns
    -------
    list[dict]
        Each entry: ``{
            "engine": "gcp-document-ai",
            "display_name": "GCP Document AI",
            "avg_cer": …,
            "avg_wer": …,
            "avg_f1": …,
            "runs": …,
        }``
        Sorted ascending by ``avg_cer`` (best first).
    """
    result = await db.execute(
        select(OCREngine).options(
            joinedload(OCREngine.runs).joinedload(OCRRun.score_summaries),
        ),
    )
    engines = list(result.scalars().unique().all())

    rankings: list[dict[str, Any]] = []
    for engine in engines:
        cer_vals: list[float] = []
        wer_vals: list[float] = []
        f1_vals: list[float] = []
        completed_count = 0

        for run in engine.runs:
            if run.status != RunStatus.COMPLETED:
                continue
            completed_count += 1
            for ss in run.score_summaries:
                if ss.breakdown is None:
                    continue
                cer = _safe_float(ss.breakdown, ("character", "cer"))
                wer = _safe_float(ss.breakdown, ("word", "wer"))
                f1 = (
                    _safe_float(ss.breakdown, ("character", "f1"))
                    or _safe_float(ss.breakdown, ("character", "accuracy"))
                    or _safe_float(ss.breakdown, ("word", "f1"))
                )
                if cer is not None:
                    cer_vals.append(cer)
                if wer is not None:
                    wer_vals.append(wer)
                if f1 is not None:
                    f1_vals.append(f1)

        if completed_count == 0:
            continue

        rankings.append({
            "engine": engine.slug,
            "display_name": engine.display_name,
            "avg_cer": round(_avg(cer_vals), 4),
            "avg_wer": round(_avg(wer_vals), 4),
            "avg_f1": round(_avg(f1_vals), 4),
            "runs": completed_count,
        })

    rankings.sort(key=lambda r: r["avg_cer"])
    return rankings


async def generate_csv_report(
    run_ids: list[uuid.UUID],
    db: AsyncSession,
    output_path: str,
) -> str:
    """Generate a CSV report with one row per run per page.

    Columns: ``run_id, engine, pdf, page, cer, wer, char_f1, word_f1``
    """
    runs = await _load_runs_with_details(db, run_ids)
    path = Path(output_path)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "engine",
            "pdf",
            "page",
            "cer",
            "wer",
            "char_f1",
            "word_f1",
        ])

        for run in runs:
            engine_slug = run.engine.slug if run.engine else str(run.engine_id)
            pdf_filename = (
                run.pdf.original_filename if run.pdf else str(run.pdf_id)
            )

            cer_val, wer_val, char_f1_val, word_f1_val = _extract_summary_scores(
                run.score_summaries,
            )

            for pr in run.page_results:
                writer.writerow([
                    str(run.id),
                    engine_slug,
                    pdf_filename,
                    pr.page_number,
                    _fmt(cer_val),
                    _fmt(wer_val),
                    _fmt(char_f1_val),
                    _fmt(word_f1_val),
                ])

    return str(path)


async def generate_json_report(
    run_ids: list[uuid.UUID],
    db: AsyncSession,
) -> dict[str, Any]:
    """Generate a full JSON data dump for the given runs.

    Includes run details, per-page scores, and engine summaries.
    """
    runs = await _load_runs_with_details(db, run_ids)

    run_data: list[dict[str, Any]] = []
    for run in runs:
        engine_slug = run.engine.slug if run.engine else str(run.engine_id)
        pdf_filename = (
            run.pdf.original_filename if run.pdf else str(run.pdf_id)
        )
        scores = _make_summary_dict(run.score_summaries)

        pages: list[dict[str, Any]] = [
            {"page": pr.page_number} for pr in run.page_results
        ]

        run_data.append({
            "run_id": str(run.id),
            "engine": engine_slug,
            "pdf": pdf_filename,
            "status": run.status.value,
            "pages": pages,
            "page_count": len(pages),
            "scores": scores,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": (
                run.completed_at.isoformat() if run.completed_at else None
            ),
        })

    engine_summaries = await aggregate_engine_rankings(db)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": run_data,
        "engine_summaries": engine_summaries,
    }


async def generate_html_report(
    run_ids: list[uuid.UUID],
    db: AsyncSession,
    output_path: str,
) -> str:
    """Generate a self-contained HTML report with inline CSS."""
    json_data = await generate_json_report(run_ids, db)
    html = _render_html_report(json_data)
    path = Path(output_path)
    path.write_text(html, encoding="utf-8")
    return str(path)


# ── Internal helpers ────────────────────────────────────────────────────────


async def _load_runs_with_details(
    db: AsyncSession,
    run_ids: list[uuid.UUID],
) -> list[Any]:
    """Load runs with their PDF, engine, score_summaries, and page_results eagerly."""
    result = await db.execute(
        select(OCRRun)
        .options(
            joinedload(OCRRun.pdf),
            joinedload(OCRRun.engine),
            joinedload(OCRRun.score_summaries),
            joinedload(OCRRun.page_results),
        )
        .where(OCRRun.id.in_(run_ids))
        .order_by(OCRRun.created_at),
    )
    return list(result.scalars().unique().all())


def _safe_float(d: Any, keys: tuple[str, ...]) -> float | None:
    """Safely drill into a nested dict and extract a float."""
    try:
        for k in keys:
            d = d[k]  # type: ignore[operator]
        return float(d)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float:
    """Arithmetic mean, returning 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _fmt(value: float | None) -> str:
    """Format a number for CSV output (4 decimal places)."""
    return f"{value:.4f}" if value is not None else ""


def _extract_summary_scores(
    summaries: list[Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Pull CER, WER, char F1, word F1 from the first ScoreSummary found."""
    for ss in summaries:
        if ss.breakdown is None:
            continue
        cer = _safe_float(ss.breakdown, ("character", "cer"))
        wer = _safe_float(ss.breakdown, ("word", "wer"))
        char_f1 = (
            _safe_float(ss.breakdown, ("character", "f1"))
            or _safe_float(ss.breakdown, ("character", "accuracy"))
        )
        word_f1 = _safe_float(ss.breakdown, ("word", "f1"))
        return cer, wer, char_f1, word_f1
    return None, None, None, None


def _make_summary_dict(summaries: list[Any]) -> dict[str, Any] | None:
    """Convert the first ScoreSummary into a plain dict."""
    for ss in summaries:
        return {
            "overall_score": ss.overall_score,
            "breakdown": ss.breakdown,
        }
    return None


def _render_html_report(data: dict[str, Any]) -> str:
    """Render a self-contained HTML report page."""
    runs = data.get("runs", [])
    engine_summaries = data.get("engine_summaries", [])

    # ── Build summary section ─────────────────────────────────────────
    total_runs = len(runs)
    total_pages = sum(r.get("page_count", 0) for r in runs)
    completed = [r for r in runs if r.get("status") == "completed"]

    cer_vals = []
    wer_vals = []
    f1_vals = []
    for r in completed:
        s = r.get("scores")
        if s and s.get("breakdown"):
            b = s["breakdown"]
            cer = _safe_float(b, ("character", "cer"))
            wer = _safe_float(b, ("word", "wer"))
            f1 = (
                _safe_float(b, ("character", "f1"))
                or _safe_float(b, ("character", "accuracy"))
                or _safe_float(b, ("word", "f1"))
            )
            if cer is not None:
                cer_vals.append(cer)
            if wer is not None:
                wer_vals.append(wer)
            if f1 is not None:
                f1_vals.append(f1)

    avg_cer = _avg(cer_vals)
    avg_wer = _avg(wer_vals)
    avg_f1 = _avg(f1_vals)

    # ── Build rows ────────────────────────────────────────────────────
    run_rows = ""
    for r in runs:
        s = r.get("scores")
        cer = wer = char_f1 = word_f1 = "—"
        if s and s.get("breakdown"):
            cer = _fmt(_safe_float(s["breakdown"], ("character", "cer")))
            wer = _fmt(_safe_float(s["breakdown"], ("word", "wer")))
            char_f1 = (
                _fmt(
                    _safe_float(s["breakdown"], ("character", "f1"))
                    or _safe_float(s["breakdown"], ("character", "accuracy")),
                )
            )
            word_f1 = _fmt(_safe_float(s["breakdown"], ("word", "f1")))

        run_rows += f"""\
            <tr>
              <td>{r["run_id"][:8]}…</td>
              <td>{r.get("engine", "?")}</td>
              <td>{r.get("pdf", "?")}</td>
              <td>{r.get("status", "?")}</td>
              <td>{r.get("page_count", 0)}</td>
              <td>{cer}</td>
              <td>{wer}</td>
              <td>{char_f1}</td>
              <td>{word_f1}</td>
            </tr>"""

    engine_rows = ""
    for e in engine_summaries:
        engine_rows += f"""\
            <tr>
              <td>{e.get("engine", "?")}</td>
              <td>{e.get("display_name", "?")}</td>
              <td>{_fmt(e.get("avg_cer"))}</td>
              <td>{_fmt(e.get("avg_wer"))}</td>
              <td>{_fmt(e.get("avg_f1"))}</td>
              <td>{e.get("runs", 0)}</td>
            </tr>"""

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCRScore Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         color: #1e293b; background: #f8fafc; padding: 2rem; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
              gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 0.75rem;
           padding: 1.25rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;
                 color: #64748b; margin-bottom: 0.25rem; }}
  .card-value {{ font-size: 1.5rem; font-weight: 700; }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin: 1.5rem 0 0.75rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border: 1px solid #e2e8f0; border-radius: 0.5rem; overflow: hidden;
           box-shadow: 0 1px 2px rgba(0,0,0,0.04); margin-bottom: 1.5rem; }}
  th, td {{ text-align: left; padding: 0.6rem 0.75rem; font-size: 0.85rem; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569;
        border-bottom: 1px solid #e2e8f0; }}
  td {{ border-bottom: 1px solid #f1f5f9; }}
  tr:last-child td {{ border-bottom: none; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 0.8rem;
             margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }}
</style>
</head>
<body>
<h1>OCRScore Report</h1>
<p class="subtitle">Generated {data.get("generated_at", "?")}</p>

<div class="summary">
  <div class="card"><div class="card-label">Runs</div><div class="card-value">{total_runs}</div></div>
  <div class="card"><div class="card-label">Pages</div><div class="card-value">{total_pages}</div></div>
  <div class="card"><div class="card-label">Avg CER</div><div class="card-value">{_fmt(avg_cer)}</div></div>
  <div class="card"><div class="card-label">Avg WER</div><div class="card-value">{_fmt(avg_wer)}</div></div>
  <div class="card"><div class="card-label">Avg F1</div><div class="card-value">{_fmt(avg_f1)}</div></div>
</div>

<h2>Engine Rankings</h2>
<table>
<thead><tr>
  <th>Engine</th><th>Name</th><th>Avg CER</th><th>Avg WER</th><th>Avg F1</th><th>Runs</th>
</tr></thead>
<tbody>
{engine_rows or '<tr><td colspan="6" style="text-align:center;color:#94a3b8;">No engine data</td></tr>'}
</tbody>
</table>

<h2>Runs</h2>
<table>
<thead><tr>
  <th>Run ID</th><th>Engine</th><th>PDF</th><th>Status</th><th>Pages</th>
  <th>CER</th><th>WER</th><th>Char F1</th><th>Word F1</th>
</tr></thead>
<tbody>
{run_rows or '<tr><td colspan="9" style="text-align:center;color:#94a3b8;">No run data</td></tr>'}
</tbody>
</table>

<div class="footer">OCRScore &mdash; OCR Evaluation Report</div>
</body>
</html>"""

    return html

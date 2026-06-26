"""Tests for the report generator — summary statistics, engine rankings, and
CSV/JSON/HTML export."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.enums import RunStatus
from backend.report_generator import (
    aggregate_engine_rankings,
    compute_summary_statistics,
    generate_csv_report,
    generate_html_report,
    generate_json_report,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def db() -> AsyncMock:
    """Mock async database session."""
    return AsyncMock()


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.uuid4()


# ── Mock helpers ──────────────────────────────────────────────────────────


def _mock_count(value: int) -> MagicMock:
    """Mock returned by ``db.execute()`` for COUNT queries (``.scalar()``)."""
    m = MagicMock()
    m.scalar.return_value = value
    return m


def _mock_orm_list(items: list) -> MagicMock:
    """Mock returned by ``db.execute()`` for ORM queries that return a list.

    Supports ``.scalars().unique().all()`` and ``.scalars().all()``.
    """
    m = MagicMock()
    scalars_m = MagicMock()
    scalars_m.all.return_value = items
    scalars_m.unique.return_value = scalars_m
    m.scalars.return_value = scalars_m
    return m


def _mock_orm_single(item) -> MagicMock:
    """Mock returned by ``db.execute()`` for ORM queries returning a single row."""
    m = MagicMock()
    scalars_m = MagicMock()
    scalars_m.one_or_none.return_value = item
    scalars_m.unique.return_value = scalars_m
    m.scalars.return_value = scalars_m
    return m


def _make_mock_engine(
    slug: str = "gcp-document-ai",
    display_name: str = "GCP Document AI",
) -> MagicMock:
    eng = MagicMock()
    eng.slug = slug
    eng.display_name = display_name
    eng.runs = []
    return eng


def _make_mock_pdf(original_filename: str = "report.pdf") -> MagicMock:
    pdf = MagicMock()
    pdf.original_filename = original_filename
    return pdf


def _make_mock_score_summary(
    overall_score: float = 0.95,
    cer: float = 0.021,
    wer: float = 0.042,
    f1: float = 0.97,
) -> MagicMock:
    ss = MagicMock()
    ss.overall_score = overall_score
    ss.breakdown = {
        "character": {"cer": cer, "f1": f1},
        "word": {"wer": wer, "f1": f1},
    }
    return ss


def _make_mock_page_result(page_number: int = 1) -> MagicMock:
    pr = MagicMock()
    pr.page_number = page_number
    return pr


def _make_mock_run(
    run_id: uuid.UUID | None = None,
    engine: MagicMock | None = None,
    pdf: MagicMock | None = None,
    status: RunStatus = RunStatus.COMPLETED,
    score_summaries: list | None = None,
    page_results: list | None = None,
) -> MagicMock:
    run = MagicMock()
    run.id = run_id or uuid.uuid4()
    run.pdf_id = uuid.uuid4()
    run.engine_id = uuid.uuid4()
    run.status = status
    run.engine = engine or _make_mock_engine()
    run.pdf = pdf or _make_mock_pdf()
    run.score_summaries = score_summaries or [_make_mock_score_summary()]
    run.page_results = page_results or [_make_mock_page_result(1)]
    run.created_at = datetime.now(UTC)
    run.completed_at = datetime.now(UTC)
    return run


# ══════════════════════════════════════════════════════════════════════════
# compute_summary_statistics
# ══════════════════════════════════════════════════════════════════════════


class TestComputeSummaryStatistics:
    """Aggregate summary across all completed runs."""

    async def test_summary_statistics(self, db) -> None:
        engine = _make_mock_engine(slug="gcp-document-ai", display_name="GCP Document AI")
        ss = _make_mock_score_summary(cer=0.021, wer=0.042, f1=0.97)
        run = _make_mock_run(engine=engine, score_summaries=[ss])

        db.execute.side_effect = [
            _mock_count(10),        # PDF count
            _mock_count(30),        # total runs
            _mock_count(25),        # completed runs
            _mock_count(100),       # pages evaluated
            _mock_orm_list([run]),  # completed runs with summaries
        ]

        result = await compute_summary_statistics(db)

        assert result["total_pdfs"] == 10
        assert result["total_runs"] == 30
        assert result["completed_runs"] == 25
        assert result["pages_evaluated"] == 100
        assert result["avg_cer"] == 0.021
        assert result["avg_wer"] == 0.042
        assert result["best_engine"] is not None
        assert result["best_engine"]["id"] == "gcp-document-ai"
        assert result["best_engine"]["avg_cer"] == 0.021

    async def test_summary_no_completed_runs(self, db) -> None:
        db.execute.side_effect = [
            _mock_count(5),
            _mock_count(10),
            _mock_count(0),
            _mock_count(0),
            _mock_orm_list([]),
        ]

        result = await compute_summary_statistics(db)

        assert result["total_pdfs"] == 5
        assert result["completed_runs"] == 0
        assert result["avg_cer"] == 0.0
        assert result["avg_wer"] == 0.0
        assert result["best_engine"] is None

    async def test_summary_averages_multiple_runs(self, db) -> None:
        engine = _make_mock_engine(slug="tesseract", display_name="Tesseract")
        run1 = _make_mock_run(engine=engine, score_summaries=[_make_mock_score_summary(cer=0.05, wer=0.08)])
        run2 = _make_mock_run(engine=engine, score_summaries=[_make_mock_score_summary(cer=0.03, wer=0.06)])

        db.execute.side_effect = [
            _mock_count(2),
            _mock_count(2),
            _mock_count(2),
            _mock_count(10),
            _mock_orm_list([run1, run2]),
        ]

        result = await compute_summary_statistics(db)

        assert result["avg_cer"] == pytest.approx(0.04, abs=0.001)  # (0.05 + 0.03) / 2
        assert result["avg_wer"] == pytest.approx(0.07, abs=0.001)  # (0.08 + 0.06) / 2


# ══════════════════════════════════════════════════════════════════════════
# aggregate_engine_rankings
# ══════════════════════════════════════════════════════════════════════════


class TestAggregateEngineRankings:
    """Engines ranked by average CER."""

    async def test_engine_rankings(self, db) -> None:
        gcp = _make_mock_engine(slug="gcp-document-ai", display_name="GCP Document AI")
        aws = _make_mock_engine(slug="aws-textract", display_name="AWS Textract")

        gcp_run = _make_mock_run(
            engine=gcp,
            score_summaries=[_make_mock_score_summary(cer=0.021, wer=0.042, f1=0.97)],
        )
        aws_run = _make_mock_run(
            engine=aws,
            score_summaries=[_make_mock_score_summary(cer=0.035, wer=0.055, f1=0.94)],
        )

        gcp.runs = [gcp_run]
        aws.runs = [aws_run]

        db.execute.side_effect = [
            _mock_orm_list([gcp, aws]),
        ]

        result = await aggregate_engine_rankings(db)

        assert len(result) == 2
        assert result[0]["engine"] == "gcp-document-ai"  # best first
        assert result[0]["avg_cer"] == 0.021
        assert result[0]["avg_f1"] == 0.97
        assert result[0]["runs"] == 1
        assert result[1]["engine"] == "aws-textract"
        assert result[1]["avg_cer"] == 0.035

    async def test_engine_rankings_no_completed(self, db) -> None:
        engine = _make_mock_engine(slug="tesseract", display_name="Tesseract")
        engine.runs = [_make_mock_run(engine=engine, status=RunStatus.FAILED)]

        db.execute.side_effect = [
            _mock_orm_list([engine]),
        ]

        result = await aggregate_engine_rankings(db)

        assert len(result) == 0  # no completed runs → no rankings

    async def test_engine_rankings_empty(self, db) -> None:
        db.execute.side_effect = [
            _mock_orm_list([]),
        ]

        result = await aggregate_engine_rankings(db)

        assert result == []


# ══════════════════════════════════════════════════════════════════════════
# generate_csv_report
# ══════════════════════════════════════════════════════════════════════════


class TestGenerateCsvReport:
    """CSV export one row per run per page."""

    async def test_csv_export(self, db, run_id) -> None:
        engine = _make_mock_engine(slug="gcp-document-ai")
        pdf = _make_mock_pdf(original_filename="test.pdf")
        ss = _make_mock_score_summary(cer=0.021, wer=0.042, f1=0.97)
        pr1 = _make_mock_page_result(1)
        pr2 = _make_mock_page_result(2)

        run = _make_mock_run(
            run_id=run_id,
            engine=engine,
            pdf=pdf,
            score_summaries=[ss],
            page_results=[pr1, pr2],
        )

        db.execute.side_effect = [
            _mock_orm_list([run]),  # _load_runs_with_details
        ]

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result_path = await generate_csv_report([run_id], db, output_path)

            assert Path(result_path).exists()
            content = Path(result_path).read_text()

            # Check headers
            assert "run_id,engine,pdf,page,cer,wer,char_f1,word_f1" in content

            # Check data rows (2 pages → 2 rows)
            lines = content.strip().split("\n")
            assert len(lines) >= 3  # header + 2 data rows
            assert "gcp-document-ai" in lines[1]
            assert "test.pdf" in lines[1]
            assert "0.0210" in lines[1]
        finally:
            Path(output_path).unlink(missing_ok=True)

    async def test_csv_empty(self, db) -> None:
        db.execute.side_effect = [
            _mock_orm_list([]),
        ]

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result_path = await generate_csv_report([], db, output_path)

            content = Path(result_path).read_text()
            # Only headers
            assert "run_id,engine,pdf,page,cer,wer,char_f1,word_f1" in content
            lines = content.strip().split("\n")
            assert len(lines) == 1  # header only
        finally:
            Path(output_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# generate_json_report
# ══════════════════════════════════════════════════════════════════════════


class TestGenerateJsonReport:
    """JSON data dump."""

    async def test_json_export(self, db, run_id) -> None:
        engine = _make_mock_engine(slug="gcp-document-ai", display_name="GCP Document AI")
        pdf = _make_mock_pdf(original_filename="test.pdf")
        ss = _make_mock_score_summary(cer=0.021, wer=0.042, f1=0.97)
        pr1 = _make_mock_page_result(1)
        pr2 = _make_mock_page_result(2)

        run = _make_mock_run(
            run_id=run_id,
            engine=engine,
            pdf=pdf,
            score_summaries=[ss],
            page_results=[pr1, pr2],
        )
        engine.runs = [run]

        # JSON calls _load_runs_with_details first, then aggregate_engine_rankings
        # _load_runs_with_details: one db call
        # aggregate_engine_rankings: one db call (load engines)
        db.execute.side_effect = [
            _mock_orm_list([run]),        # _load_runs_with_details
            _mock_orm_list([engine]),      # aggregate_engine_rankings
        ]

        result = await generate_json_report([run_id], db)

        # Top-level structure
        assert "generated_at" in result
        assert "runs" in result
        assert "engine_summaries" in result

        # Run entries
        assert len(result["runs"]) == 1
        r = result["runs"][0]
        assert r["run_id"] == str(run_id)
        assert r["engine"] == "gcp-document-ai"
        assert r["pdf"] == "test.pdf"
        assert r["status"] == "completed"
        assert r["page_count"] == 2
        assert len(r["pages"]) == 2

        # Scores
        assert r["scores"] is not None
        assert r["scores"]["overall_score"] == 0.95
        assert r["scores"]["breakdown"]["character"]["cer"] == 0.021

        # Engine summaries
        assert len(result["engine_summaries"]) == 1
        assert result["engine_summaries"][0]["engine"] == "gcp-document-ai"

    async def test_json_empty(self, db) -> None:
        db.execute.side_effect = [
            _mock_orm_list([]),
            _mock_orm_list([]),
        ]

        result = await generate_json_report([], db)

        assert result["runs"] == []
        assert result["engine_summaries"] == []


# ══════════════════════════════════════════════════════════════════════════
# generate_html_report
# ══════════════════════════════════════════════════════════════════════════


class TestGenerateHtmlReport:
    """Self-contained HTML report."""

    async def test_html_export(self, db, run_id) -> None:
        engine = _make_mock_engine(slug="gcp-document-ai", display_name="GCP Document AI")
        pdf = _make_mock_pdf(original_filename="test.pdf")
        ss = _make_mock_score_summary(cer=0.021, wer=0.042, f1=0.97)
        pr1 = _make_mock_page_result(1)

        run = _make_mock_run(
            run_id=run_id,
            engine=engine,
            pdf=pdf,
            score_summaries=[ss],
            page_results=[pr1],
        )
        engine.runs = [run]

        db.execute.side_effect = [
            _mock_orm_list([run]),        # _load_runs_with_details
            _mock_orm_list([engine]),      # aggregate_engine_rankings
        ]

        with NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            result_path = await generate_html_report([run_id], db, output_path)

            assert Path(result_path).exists()
            content = Path(result_path).read_text(encoding="utf-8")

            # Valid HTML structure
            assert "<!DOCTYPE html>" in content
            assert "</html>" in content
            assert "<style>" in content

            # Contains data
            assert "OCRScore Report" in content
            assert "gcp-document-ai" in content
            assert "Engine Rankings" in content
            assert "Avg F1" in content
        finally:
            Path(output_path).unlink(missing_ok=True)

    async def test_html_empty(self, db) -> None:
        db.execute.side_effect = [
            _mock_orm_list([]),
            _mock_orm_list([]),
        ]

        with NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            result_path = await generate_html_report([], db, output_path)

            content = Path(result_path).read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "No run data" in content or "No engine data" in content
        finally:
            Path(output_path).unlink(missing_ok=True)

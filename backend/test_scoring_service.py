"""Tests for the scoring service — compute_run_scores, compute_page_scores,
compare_engines, and their error paths."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.evaluation.scoring_service import (
    compare_engines,
    compute_page_scores,
    compute_run_scores,
)
from backend.models.enums import RunStatus
from backend.models.ground_truth import GTPageResult
from backend.models.page_result import PageResult
from backend.models.run import OCRRun

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def db() -> AsyncMock:
    """Mock async database session."""
    return AsyncMock()


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def gt_version_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def pdf_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def engine_id() -> uuid.UUID:
    return uuid.uuid4()


def _mock_scalar_result(data):
    """Helper: create a mock that returns data from .scalars().one_or_none() etc."""
    m = MagicMock()
    scalars_m = MagicMock()
    scalars_m.one_or_none.return_value = data
    scalars_m.all.return_value = data if isinstance(data, list) else [data]
    # .unique().all() is called on the scalars mock in scoring_service
    scalars_m.unique.return_value = scalars_m
    m.scalars.return_value = scalars_m
    return m


def _make_mock_run(
    rid: uuid.UUID,
    status: RunStatus = RunStatus.COMPLETED,
) -> MagicMock:
    r = MagicMock(spec=OCRRun)
    r.id = rid
    r.status = status
    return r


def _make_mock_page_result(
    page_number: int,
    text: str = "hello world",
) -> MagicMock:
    pr = MagicMock(spec=PageResult)
    pr.page_number = page_number
    pr.data = {
        "blocks": [
            {
                "type": "text",
                "lines": [
                    {
                        "words": [
                            {"text": w, "bbox": [0, 0, 10, 5], "confidence": 0.9}
                            for w in text.split()
                        ],
                    },
                ],
            },
        ],
    }
    return pr


def _make_mock_gt_page(
    page_number: int,
    text: str = "hello world",
) -> MagicMock:
    gp = MagicMock(spec=GTPageResult)
    gp.page_number = page_number
    gp.data = {
        "blocks": [
            {
                "type": "text",
                "lines": [
                    {
                        "words": [
                            {"text": w, "bbox": [0, 0, 10, 5], "confidence": 1.0}
                            for w in text.split()
                        ],
                    },
                ],
            },
        ],
    }
    return gp


# ── compute_run_scores ────────────────────────────────────────────────────


class TestComputeRunScores:
    """Scores computed for a completed run."""

    async def test_compute_run_scores_perfect(self, db, run_id, gt_version_id) -> None:
        run = _make_mock_run(run_id)
        db.execute.side_effect = [
            _mock_scalar_result(run),  # load run
            _mock_scalar_result([_make_mock_page_result(1)]),  # run pages
            _mock_scalar_result([_make_mock_gt_page(1)]),  # gt pages
        ]

        result = await compute_run_scores(run_id, gt_version_id, db)

        assert result["run_id"] == str(run_id)
        assert result["gt_version_id"] == str(gt_version_id)
        assert result["overall"] is not None
        assert result["overall"]["cer"] == 0.0
        assert result["overall"]["wer"] == 0.0
        assert result["overall"]["char_f1"] == 1.0
        assert result["overall"]["word_f1"] == 1.0
        assert result["pages"] == 1
        assert result["evaluated_pages"] == 1

    async def test_compute_run_scores_with_errors(
        self, db, run_id, gt_version_id,
    ) -> None:
        run = _make_mock_run(run_id)
        # Run has "helpo" vs GT "hello" on first page
        run_page = _make_mock_page_result(1, text="helpo world")
        gt_page = _make_mock_gt_page(1, text="hello world")
        db.execute.side_effect = [
            _mock_scalar_result(run),
            _mock_scalar_result([run_page]),
            _mock_scalar_result([gt_page]),
        ]

        result = await compute_run_scores(run_id, gt_version_id, db)

        assert result["overall"] is not None
        assert result["overall"]["cer"] > 0.0  # "helpo" vs "hello" has errors
        assert result["overall"]["wer"] > 0.0  # word-level mismatch
        assert result["overall"]["char_f1"] < 1.0

    async def test_compute_run_scores_not_completed(
        self, db, run_id, gt_version_id,
    ) -> None:
        run = _make_mock_run(run_id, status=RunStatus.RUNNING)
        db.execute.side_effect = [
            _mock_scalar_result(run),
        ]

        result = await compute_run_scores(run_id, gt_version_id, db)

        assert result["overall"] is None
        assert "not 'completed'" in (result.get("message") or "")

    async def test_empty_run(self, db, run_id, gt_version_id) -> None:
        """No page results → empty scores."""
        run = _make_mock_run(run_id)
        db.execute.side_effect = [
            _mock_scalar_result(run),
            _mock_scalar_result([]),  # no run pages
            _mock_scalar_result([_make_mock_gt_page(1)]),  # gt pages exist
        ]

        result = await compute_run_scores(run_id, gt_version_id, db)

        assert result["overall"] is not None
        assert result["overall"]["cer"] == 0.0
        assert result["overall"]["wer"] == 0.0
        assert result["pages"] == 0

    async def test_invalid_run(self, db, run_id, gt_version_id) -> None:
        """Run not found → error."""
        db.execute.side_effect = [
            _mock_scalar_result(None),  # no run found
        ]

        with pytest.raises(ValueError, match="not found"):
            await compute_run_scores(run_id, gt_version_id, db)


# ── compute_page_scores ───────────────────────────────────────────────────


class TestComputePageScores:
    """Single page scores."""

    async def test_compute_page_scores(self, db, run_id, gt_version_id) -> None:
        run = _make_mock_run(run_id)
        db.execute.side_effect = [
            _mock_scalar_result(run),  # load run
            _mock_scalar_result(_make_mock_page_result(1)),  # page result
            _mock_scalar_result(_make_mock_gt_page(1)),  # GT page
        ]

        result = await compute_page_scores(run_id, 1, gt_version_id, db)

        assert result["page"] == 1
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["char_f1"] == 1.0
        assert result["word_f1"] == 1.0

    async def test_compute_page_scores_run_not_completed(
        self, db, run_id, gt_version_id,
    ) -> None:
        run = _make_mock_run(run_id, status=RunStatus.FAILED)
        db.execute.side_effect = [
            _mock_scalar_result(run),
        ]

        result = await compute_page_scores(run_id, 1, gt_version_id, db)

        assert result["cer"] is None
        assert "not 'completed'" in (result.get("message") or "")

    async def test_compute_page_scores_page_not_found(
        self, db, run_id, gt_version_id,
    ) -> None:
        run = _make_mock_run(run_id)
        db.execute.side_effect = [
            _mock_scalar_result(run),
            _mock_scalar_result(None),  # page not found
        ]

        result = await compute_page_scores(run_id, 99, gt_version_id, db)

        assert result["cer"] is None
        assert "not found" in (result.get("message") or "")

    async def test_compute_page_scores_gt_not_found(
        self, db, run_id, gt_version_id,
    ) -> None:
        run = _make_mock_run(run_id)
        db.execute.side_effect = [
            _mock_scalar_result(run),
            _mock_scalar_result(_make_mock_page_result(1)),
            _mock_scalar_result(None),  # GT page not found
        ]

        result = await compute_page_scores(run_id, 1, gt_version_id, db)

        assert result["cer"] is None
        assert "Ground truth page" in (result.get("message") or "")


# ── compare_engines ───────────────────────────────────────────────────────


class TestCompareEngines:
    """Cross-engine comparison."""

    async def test_compare_engines(
        self, db, pdf_id, gt_version_id, engine_id,
    ) -> None:
        other_engine_id = uuid.uuid4()
        run_id_1 = uuid.uuid4()
        run_id_2 = uuid.uuid4()

        run_1 = _make_mock_run(run_id_1)
        run_2 = _make_mock_run(run_id_2)

        db.execute.side_effect = [
            # GT data (first call for compare_engines -> internal _build_gt_data)
            _mock_scalar_result([_make_mock_gt_page(1)]),
            # Query for engine 1
            _mock_scalar_result(run_1),
            # run data for engine 1
            _mock_scalar_result([_make_mock_page_result(1)]),
            # Query for engine 2
            _mock_scalar_result(run_2),
            # run data for engine 2
            _mock_scalar_result([_make_mock_page_result(1)]),
        ]

        result = await compare_engines(
            pdf_id=pdf_id,
            engine_ids=[engine_id, other_engine_id],
            gt_version_id=gt_version_id,
            db=db,
        )

        assert result["pdf_id"] == str(pdf_id)
        assert result["gt_version_id"] == str(gt_version_id)
        assert len(result["engines"]) == 2
        for entry in result["engines"]:
            assert entry["scores"] is not None
            assert entry["scores"]["cer"] == 0.0
            assert entry["scores"]["char_f1"] == 1.0

    async def test_compare_engines_missing_engine(
        self, db, pdf_id, gt_version_id, engine_id,
    ) -> None:
        db.execute.side_effect = [
            # GT data
            _mock_scalar_result([_make_mock_gt_page(1)]),
            # No completed run for this engine
            _mock_scalar_result(None),
        ]

        result = await compare_engines(
            pdf_id=pdf_id,
            engine_ids=[engine_id],
            gt_version_id=gt_version_id,
            db=db,
        )

        assert len(result["engines"]) == 1
        assert result["engines"][0]["scores"] is None
        assert "No completed run" in (result["engines"][0].get("message") or "")

    async def test_compare_engines_runs_not_found(
        self, db, pdf_id, gt_version_id,
    ) -> None:
        """Empty engine_ids → no engine results."""
        db.execute.side_effect = [
            _mock_scalar_result([_make_mock_gt_page(1)]),  # GT data
        ]
        result = await compare_engines(
            pdf_id=pdf_id,
            engine_ids=[],  # no engines to compare
            gt_version_id=gt_version_id,
            db=db,
        )

        assert result["pdf_id"] == str(pdf_id)
        assert len(result["engines"]) == 0

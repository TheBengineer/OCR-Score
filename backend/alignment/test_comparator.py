"""Tests for the multi-engine comparison module.

Covers :func:`align_multiple_engine_pages` / :func:`build_comparison_grid`
from :mod:`backend.alignment.comparator`.
"""

from __future__ import annotations

from backend.alignment.comparator import (
    _compute_consensus_for_position,
    _compute_position_entropy,
    _extract_flat_words,
    align_multiple_engine_pages,
    build_comparison_grid,
)  # fmt: skip

# ── Flat word extraction tests ────────────────────────────────────────────────


class TestExtractFlatWords:
    """Word extraction from various page result formats."""

    def test_flat_word_list(self) -> None:
        """``{"words": [...]}`` extracted directly."""
        data = {"words": [{"text": "hello"}, {"text": "world"}]}
        words = _extract_flat_words(data)
        assert len(words) == 2
        assert words[0]["text"] == "hello"

    def test_canonical_jsonb(self) -> None:
        """Canonical ``{"data": {"blocks": [...]}}`` format."""
        data = {
            "data": {
                "blocks": [
                    {
                        "lines": [
                            {
                                "words": [
                                    {"text": "Hello", "confidence": 0.95},
                                ],
                            },
                        ],
                    },
                ],
            },
        }
        words = _extract_flat_words(data)
        assert len(words) == 1
        assert words[0]["text"] == "Hello"

    def test_partial_jsonb_no_data_key(self) -> None:
        """Partial JSONB with ``{"blocks": [...]}`` at top level."""
        data = {
            "blocks": [
                {
                    "lines": [
                        {
                            "words": [
                                {"text": "foo"},
                                {"text": "bar"},
                            ],
                        },
                    ],
                },
            ],
        }
        words = _extract_flat_words(data)
        assert len(words) == 2
        assert words[0]["text"] == "foo"

    def test_empty_input(self) -> None:
        """Empty dict → empty list."""
        assert _extract_flat_words({}) == []

    def test_none_words(self) -> None:
        """Explicit ``None`` words → fall through to JSONB paths."""
        assert _extract_flat_words({"words": None}) == []

    def test_empty_word_list(self) -> None:
        """Empty word list → fall through to JSONB paths."""
        assert _extract_flat_words({"words": []}) == []


# ── Consensus helper tests ────────────────────────────────────────────────────


class TestComputeConsensusForPosition:
    """Majority-vote consensus computation."""

    def test_all_agree(self) -> None:
        """All engines agree → consensus is that text."""
        position = {
            "eng_a": {"text": "Hello", "confidence": 0.9},
            "eng_b": {"text": "Hello", "confidence": 0.8},
            "eng_c": {"text": "Hello", "confidence": 0.85},
        }
        text, conf = _compute_consensus_for_position(position)
        assert text == "Hello"
        # Consensus confidence = average of engine confidences for "Hello"
        assert 0.8 <= conf <= 1.0

    def test_majority_wins(self) -> None:
        """2 of 3 agree → majority wins."""
        position = {
            "eng_a": {"text": "Hello", "confidence": 0.9},
            "eng_b": {"text": "Hello", "confidence": 0.8},
            "eng_c": {"text": "World", "confidence": 0.95},
        }
        text, _conf = _compute_consensus_for_position(position)
        assert text == "Hello"

    def test_tie_confidence_weighted(self) -> None:
        """Tie broken by confidence-weighted vote."""
        position = {
            "eng_a": {"text": "Hello", "confidence": 0.5},
            "eng_b": {"text": "World", "confidence": 0.95},
            "eng_c": {"text": "World", "confidence": 0.9},
        }
        text, _conf = _compute_consensus_for_position(position)
        # All three present, "World" has 2 votes (0.95+0.9 avg > 0.5)
        assert text == "World"

    def test_all_gaps(self) -> None:
        """All None → None consensus."""
        position = {"eng_a": None, "eng_b": None}
        text, conf = _compute_consensus_for_position(position)
        assert text is None
        assert conf == 0.0

    def test_some_gaps(self) -> None:
        """Some None entries → only non-None engines vote."""
        position = {
            "eng_a": {"text": "Hello", "confidence": 0.9},
            "eng_b": None,
            "eng_c": {"text": "Hello", "confidence": 0.85},
        }
        text, _conf = _compute_consensus_for_position(position)
        assert text == "Hello"


class TestComputePositionEntropy:
    """Per-position entropy computation."""

    def test_all_same(self) -> None:
        """All engines have the same text → entropy 0."""
        entropy = _compute_position_entropy(["Hello", "Hello", "Hello"])
        assert entropy == 0.0

    def test_all_different(self) -> None:
        """Every engine has a different text → max entropy (1.0)."""
        entropy = _compute_position_entropy(["Hello", "World", "Foo"])
        assert entropy == 1.0

    def test_some_gaps(self) -> None:
        """Gaps are excluded from entropy computation."""
        entropy = _compute_position_entropy(["Hello", None, "Hello"])
        # 2 non-None: both "Hello" → entropy 0
        assert entropy == 0.0

    def test_fewer_than_two_non_null(self) -> None:
        """0 or 1 non-None entries → entropy 0."""
        assert _compute_position_entropy([None, None]) == 0.0
        assert _compute_position_entropy(["Hello", None]) == 0.0
        assert _compute_position_entropy(["Hello"]) == 0.0
        assert _compute_position_entropy([]) == 0.0

    def test_two_different(self) -> None:
        """2 engines with different text → entropy 1.0."""
        entropy = _compute_position_entropy(["A", "B"])
        assert entropy == 1.0


# ── Multi-engine alignment tests ──────────────────────────────────────────────


class TestAlignMultipleEnginePages:
    """Main alignment tests."""

    def test_align_two_engines(self) -> None:
        """Two identical engine outputs → all matches, zero entropy."""
        engine_a = {
            "engine": "tesseract",
            "words": [
                {"text": "Hello", "bbox": [0, 0, 50, 20], "confidence": 0.95},
                {"text": "World", "bbox": [50, 0, 100, 20], "confidence": 0.98},
            ],
        }
        engine_b = {
            "engine": "gcp-document-ai",
            "words": [
                {"text": "Hello", "bbox": [0, 0, 50, 20], "confidence": 0.93},
                {"text": "World", "bbox": [50, 0, 100, 20], "confidence": 0.96},
            ],
        }

        result = align_multiple_engine_pages([engine_a, engine_b])

        assert result["num_positions"] == 2
        assert len(result["aligned_words"]) == 2
        assert result["consensus_entropy"] == 0.0
        for pos in result["aligned_words"]:
            for eng_name in ("tesseract", "gcp-document-ai"):
                eng_data = pos["engines"][eng_name]
                assert eng_data["status"] == "match"
                assert eng_data["text"] is not None

    def test_align_three_engines(self) -> None:
        """Three engine outputs → correct number of aligned positions."""
        engine_a = {
            "engine": "tesseract",
            "words": [
                {"text": "The", "confidence": 0.95},
                {"text": "quick", "confidence": 0.95},
                {"text": "fox", "confidence": 0.95},
            ],
        }
        engine_b = {
            "engine": "gcp-document-ai",
            "words": [
                {"text": "The", "confidence": 0.90},
                {"text": "quick", "confidence": 0.90},
                {"text": "fox", "confidence": 0.90},
            ],
        }
        engine_c = {
            "engine": "aws-textract",
            "words": [
                {"text": "The", "confidence": 0.85},
                {"text": "quick", "confidence": 0.85},
                {"text": "fox", "confidence": 0.85},
            ],
        }

        result = align_multiple_engine_pages([engine_a, engine_b, engine_c])

        assert result["num_positions"] == 3
        assert len(result["aligned_words"]) == 3
        assert result["consensus_entropy"] == 0.0
        assert len(result["engines"]) == 3
        assert result["engines"] == ["tesseract", "gcp-document-ai", "aws-textract"]

    def test_align_with_consensus(self) -> None:
        """Correct consensus and status assignment when engines disagree."""
        engine_a = {
            "engine": "tesseract",
            "words": [
                {"text": "Hello", "confidence": 0.95},
                {"text": "World", "confidence": 0.95},
            ],
        }
        engine_b = {
            "engine": "gcp-document-ai",
            "words": [
                {"text": "Hello", "confidence": 0.90},
                {"text": "Wrld", "confidence": 0.80},
            ],
        }
        engine_c = {
            "engine": "aws-textract",
            "words": [
                {"text": "Hello", "confidence": 0.85},
                {"text": "World", "confidence": 0.85},
            ],
        }

        result = align_multiple_engine_pages([engine_a, engine_b, engine_c])

        # Position 0: "Hello" (unanimous).
        first_pos = result["aligned_words"][0]
        assert first_pos["consensus"] == "Hello"
        assert first_pos["engines"]["tesseract"]["status"] == "match"
        assert first_pos["engines"]["gcp-document-ai"]["status"] == "match"
        assert first_pos["engines"]["aws-textract"]["status"] == "match"

        # Position 1: "World" (2 of 3).
        second_pos = result["aligned_words"][1]
        assert second_pos["consensus"] == "World"
        assert second_pos["engines"]["tesseract"]["status"] == "match"
        assert second_pos["engines"]["gcp-document-ai"]["status"] == "wrong"
        assert second_pos["engines"]["aws-textract"]["status"] == "match"

    def test_compare_diff_lengths(self) -> None:
        """Engines with different word counts → gaps tracked."""
        engine_a = {
            "engine": "tesseract",
            "words": [
                {"text": "The", "confidence": 0.95},
                {"text": "quick", "confidence": 0.95},
                {"text": "brown", "confidence": 0.95},
                {"text": "fox", "confidence": 0.95},
            ],
        }
        engine_b = {
            "engine": "gcp-document-ai",
            "words": [
                {"text": "The", "confidence": 0.90},
                {"text": "fox", "confidence": 0.90},
            ],
        }

        result = align_multiple_engine_pages([engine_a, engine_b])

        # Engine B has gaps for "quick" and "brown" (or similar alignment).
        gap_positions = [
            pos for pos in result["aligned_words"]
            if pos["engines"]["gcp-document-ai"]["status"] == "missing"
        ]
        assert len(gap_positions) >= 1

    def test_empty_input(self) -> None:
        """No engines → empty response."""
        result = align_multiple_engine_pages([])

        assert result["aligned_words"] == []
        assert result["engines"] == []
        assert result["num_positions"] == 0
        assert result["consensus_entropy"] == 0.0

    def test_single_engine(self) -> None:
        """Single engine → basic alignment, zero entropy."""
        engine = {
            "engine": "tesseract",
            "words": [{"text": "hello", "confidence": 0.95}],
        }

        result = align_multiple_engine_pages([engine])

        assert result["num_positions"] == 1
        assert result["consensus_entropy"] == 0.0
        pos = result["aligned_words"][0]
        assert pos["consensus"] == "hello"
        assert pos["engines"]["tesseract"]["status"] == "match"

    def test_extras_tracked(self) -> None:
        """Extra words in non-reference engines recorded as extras."""
        engine_a = {
            "engine": "tesseract",
            "words": [
                {"text": "Hello", "confidence": 0.95},
            ],
        }
        engine_b = {
            "engine": "gcp-document-ai",
            "words": [
                {"text": "Hello", "confidence": 0.90},
                {"text": "Extra", "confidence": 0.70},
            ],
        }

        result = align_multiple_engine_pages([engine_a, engine_b])

        extras = result.get("extras", {})
        assert len(extras.get("gcp-document-ai", [])) >= 1
        assert extras["gcp-document-ai"][0]["text"] == "Extra"
        # Engine stats should reflect the extra.
        assert result["stats"]["engine_stats"]["gcp-document-ai"]["extra"] >= 1

    def test_hierarchical_jsonb_input(self) -> None:
        """Canonical JSONB format is handled correctly."""
        engine = {
            "engine": "tesseract",
            "data": {
                "blocks": [
                    {
                        "type": "text",
                        "bbox": [0, 0, 200, 50],
                        "confidence": 0.95,
                        "order": 0,
                        "lines": [
                            {
                                "text": "Hello World",
                                "bbox": [0, 0, 200, 50],
                                "confidence": 0.95,
                                "order": 0,
                                "words": [
                                    {"text": "Hello", "bbox": [0, 0, 50, 20],
                                     "confidence": 0.95, "order": 0, "chars": []},
                                    {"text": "World", "bbox": [50, 0, 100, 20],
                                     "confidence": 0.95, "order": 1, "chars": []},
                                ],
                            },
                        ],
                    },
                ],
            },
        }

        result = align_multiple_engine_pages([engine])

        assert result["num_positions"] == 2
        assert result["aligned_words"][0]["consensus"] == "Hello"
        assert result["aligned_words"][1]["consensus"] == "World"

    def test_engine_id_fallback(self) -> None:
        """Fallback for missing 'engine' key."""
        engine = {"engine_id": "custom-slug", "words": [{"text": "test"}]}
        result = align_multiple_engine_pages([engine])
        assert result["engines"] == ["custom-slug"]

    def test_ref_has_no_words(self) -> None:
        """Reference engine with no words → empty alignment."""
        engine_a = {"engine": "tesseract", "words": []}
        engine_b = {"engine": "gcp", "words": [{"text": "hello"}]}
        result = align_multiple_engine_pages([engine_a, engine_b])
        assert result["num_positions"] == 0
        assert result["aligned_words"] == []

    def test_entropy_positive_when_disagreement(self) -> None:
        """Disagreement among engines → entropy > 0."""
        engine_a = {"engine": "a", "words": [{"text": "Hello"}]}
        engine_b = {"engine": "b", "words": [{"text": "World"}]}
        engine_c = {"engine": "c", "words": [{"text": "Hello"}]}

        result = align_multiple_engine_pages([engine_a, engine_b, engine_c])
        # 2 "Hello", 1 "World" → some disagreement
        assert result["consensus_entropy"] > 0.0


# ── Comparison grid builder tests ─────────────────────────────────────────────


class TestBuildComparisonGrid:
    """Grid builder tests."""

    def test_build_comparison_grid(self) -> None:
        """Grid has correct top-level structure."""
        aligned = {
            "aligned_words": [
                {
                    "position": 0,
                    "consensus": "Hello",
                    "consensus_confidence": 0.95,
                    "engines": {
                        "tesseract": {"text": "Hello", "confidence": 0.95, "status": "match"},
                    },
                },
            ],
            "engines": ["tesseract"],
            "consensus_entropy": 0.0,
            "num_positions": 1,
            "extras": {},
            "stats": {
                "total_words": 1,
                "engine_stats": {
                    "tesseract": {"match": 1, "wrong": 0, "missing": 0, "extra": 0},
                },
            },
        }

        grid = build_comparison_grid(
            aligned,
            page_number=2,
            dimensions={"width": 612.0, "height": 792.0},
        )

        assert grid["page_number"] == 2
        assert grid["dimensions"]["width"] == 612.0
        assert grid["dimensions"]["height"] == 792.0
        assert grid["engines"] == ["tesseract"]
        assert "alignment" in grid
        assert "aligned_words" in grid["alignment"]
        assert grid["consensus_entropy"] == 0.0
        assert "stats" in grid
        assert grid["stats"]["total_words"] == 1

    def test_empty_aligned_data(self) -> None:
        """Empty aligned data → empty grid."""
        aligned = {
            "aligned_words": [],
            "engines": [],
            "consensus_entropy": 0.0,
            "num_positions": 0,
            "extras": {},
            "stats": {"total_words": 0, "engine_stats": {}},
        }

        grid = build_comparison_grid(aligned)

        assert grid["page_number"] == 1
        assert grid["alignment"]["aligned_words"] == []
        assert grid["engines"] == []

    def test_default_dimensions(self) -> None:
        """No dimensions provided → zero defaults."""
        aligned = {
            "aligned_words": [],
            "engines": [],
            "consensus_entropy": 0.0,
            "num_positions": 0,
            "extras": {},
            "stats": {"total_words": 0, "engine_stats": {}},
        }

        grid = build_comparison_grid(aligned)

        assert grid["dimensions"]["width"] == 0.0
        assert grid["dimensions"]["height"] == 0.0

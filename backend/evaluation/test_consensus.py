"""Tests for the consensus entropy auto-GT module — alignment, entropy,
weighted voting, ground truth building, and validation."""

from __future__ import annotations

import math

from backend.evaluation.consensus import (
    _align_engine_outputs,
    _align_word_sequences,
    _majority_vote,
    _validate_candidate_gt,
    build_ground_truth,
    compute_confidence_weighted_consensus,
    compute_consensus_entropy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Consensus Entropy
# ═══════════════════════════════════════════════════════════════════════════════


class TestConsensusEntropy:
    """Core Consensus Entropy computation."""

    def test_ce_identical_texts(self) -> None:
        """All engines produce identical text → CE = 0.0."""
        texts = [
            "hello world foo bar",
            "hello world foo bar",
            "hello world foo bar",
        ]
        ce = compute_consensus_entropy(texts)
        assert ce == 0.0

    def test_ce_completely_different(self) -> None:
        """All engines produce completely different text → CE ≈ 1.0."""
        texts = [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
        ]
        ce = compute_consensus_entropy(texts)
        # With 3 engines and 3 word positions, each position has 3 different
        # words → H_per_pos = log2(3) ≈ 1.585, avg = 1.585, max = 1.585 → 1.0
        assert math.isclose(ce, 1.0, rel_tol=1e-3)

    def test_ce_partial_agreement(self) -> None:
        """2/3 engines agree → medium CE (0.2-0.6)."""
        texts = [
            "hello world",
            "hello moon",
            "hello world",
        ]
        ce = compute_consensus_entropy(texts)
        # Position 0: "hello" × 3 → H = 0.0
        # Position 1: "world" × 2, "moon" × 1 → H ≈ 0.918
        # avg = 0.459, norm = 0.459 / 1.585 ≈ 0.29
        assert 0.2 < ce < 0.6

    def test_ce_two_engines_full_agreement(self) -> None:
        """Two engines with identical text → CE = 0.0."""
        texts = ["hello world", "hello world"]
        ce = compute_consensus_entropy(texts)
        assert ce == 0.0

    def test_ce_two_engines_complete_disagreement(self) -> None:
        """Two engines with completely different text → CE = 1.0."""
        texts = ["hello", "world"]
        ce = compute_consensus_entropy(texts)
        assert math.isclose(ce, 1.0, rel_tol=1e-3)

    def test_ce_single_engine(self) -> None:
        """Single engine → CE = 0.0 (not meaningful)."""
        texts = ["hello world"]
        ce = compute_consensus_entropy(texts)
        assert ce == 0.0

    def test_ce_empty_input(self) -> None:
        """Empty list → CE = 0.0."""
        ce = compute_consensus_entropy([])
        assert ce == 0.0

    def test_ce_different_lengths(self) -> None:
        """Texts with different word counts → CNW alignment handles gaps."""
        texts = [
            "the quick brown fox",
            "the brown fox",
            "the quick brown fox jumps",
        ]
        ce = compute_consensus_entropy(texts)
        # Should produce a finite value in [0, 1].
        assert 0.0 <= ce <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence-weighted consensus
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceWeightedConsensus:
    """Confidence-weighted consensus computation."""

    def test_confidence_weighted_consensus(self) -> None:
        """Confidence weighting changes vote outcome when equal-weight tie."""
        outputs = [
            {
                "engine": "tesseract",
                "words": [{"text": "hello", "confidence": 0.9}],
            },
            {
                "engine": "gcp",
                "words": [{"text": "world", "confidence": 0.3}],
            },
            {
                "engine": "textract",
                "words": [{"text": "world", "confidence": 0.95}],
            },
        ]
        result = compute_confidence_weighted_consensus(outputs)
        # Equal weight: "hello"=1, "world"=2 → world wins (tie already broken)
        # But the agreement_flags should be False since weighted==equal here.
        assert len(result["consensus_words"]) == 1
        assert result["consensus_words"][0]["text"] == "world"
        # Confidence-weighted and equal-weight agree for this simple case.
        assert result["agreement_flags"] == [False]

    def test_confidence_weighted_tiebreaker(self) -> None:
        """2-2 equal-weight tie → confidence-weighted breaks it."""
        outputs = [
            {
                "engine": "tesseract",
                "words": [{"text": "hello", "confidence": 0.3}],
            },
            {
                "engine": "gcp",
                "words": [{"text": "hello", "confidence": 0.2}],
            },
            {
                "engine": "textract",
                "words": [{"text": "world", "confidence": 0.95}],
            },
            {
                "engine": "azure",
                "words": [{"text": "world", "confidence": 0.9}],
            },
        ]
        result = compute_confidence_weighted_consensus(outputs)
        # Equal weight: "hello"=2, "world"=2 → tie
        # Confidence: hello=0.5, world=1.85 → "world" wins
        assert len(result["consensus_words"]) == 1
        assert result["consensus_words"][0]["text"] == "world"

    def test_confidence_weighted_no_confidence(self) -> None:
        """When no confidence provided, defaults to 1.0 for each engine."""
        outputs = [
            {"engine": "e1", "words": [{"text": "hello"}]},
            {"engine": "e2", "words": [{"text": "world"}]},
        ]
        result = compute_confidence_weighted_consensus(outputs)
        # With confidence=1.0 for both and equal weight tie, weighted picks one.
        # Both have confidence=1.0, confidence_sums: hello=1.0, world=1.0.
        # max picks the first with highest value (hello).
        assert len(result["consensus_words"]) == 1

    def test_confidence_weighted_empty(self) -> None:
        """Empty engine outputs → no positions."""
        result = compute_confidence_weighted_consensus([])
        assert result["num_positions"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# build_ground_truth
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildGroundTruth:
    """Full ground truth builder — routing and edge cases."""

    def test_build_ground_truth_low_entropy(self) -> None:
        """Low CE → auto-accept with source='auto_consensus'."""
        outputs = [
            {"engine": "tesseract", "words": [{"text": "hello"}, {"text": "world"}]},
            {"engine": "gcp", "words": [{"text": "hello"}, {"text": "world"}]},
            {"engine": "textract", "words": [{"text": "hello"}, {"text": "world"}]},
        ]
        result = build_ground_truth(outputs)
        assert result["source"] == "auto_consensus"
        assert result["consensus_entropy"] == 0.0
        assert result["needs_review"] is False
        assert len(result["pages"]) == 1
        assert len(result["pages"][0]["blocks"]) == 1
        assert result["warnings"] == []

    def test_build_ground_truth_medium_entropy(self) -> None:
        """Medium CE → auto-accept but flag for review."""
        outputs = [
            {"engine": "tesseract", "words": [{"text": "hello"}, {"text": "world"}]},
            {"engine": "gcp", "words": [{"text": "hello"}, {"text": "moon"}]},
            {"engine": "textract", "words": [{"text": "hello"}, {"text": "world"}]},
        ]
        result = build_ground_truth(outputs)
        assert result["source"] == "auto_consensus"
        assert result["needs_review"] is True
        assert len(result["pages"]) == 1
        # Should have a warning about moderate CE.
        assert any("Moderate CE" in w for w in result["warnings"])

    def test_build_ground_truth_high_entropy(self) -> None:
        """High CE → reject, return empty GT (source=None, no pages)."""
        outputs = [
            {"engine": "tesseract", "words": [{"text": "hello"}]},
            {"engine": "gcp", "words": [{"text": "world"}]},
            {"engine": "textract", "words": [{"text": "foo"}]},
        ]
        result = build_ground_truth(outputs)
        assert result["source"] is None
        assert result["pages"] == []
        assert result["needs_review"] is True
        assert any("High CE" in w for w in result["warnings"])

    def test_empty_input(self) -> None:
        """No engines → empty GT with appropriate warning."""
        result = build_ground_truth([])
        assert result["source"] is None
        assert result["pages"] == []
        assert any("No engine outputs" in w for w in result["warnings"])

    def test_single_engine(self) -> None:
        """Single engine → its output is GT with needs_review=True."""
        outputs = [
            {"engine": "tesseract", "words": [{"text": "hello"}, {"text": "world"}]},
        ]
        result = build_ground_truth(outputs)
        assert result["source"] == "auto_consensus"
        assert result["pages"] != []
        assert result["needs_review"] is True
        assert any("Single engine" in w for w in result["warnings"])

    def test_config_thresholds(self) -> None:
        """Custom thresholds affect routing."""
        outputs = [
            {"engine": "e1", "words": [{"text": "hello"}]},
            {"engine": "e2", "words": [{"text": "world"}]},
        ]
        # With two different words, CE = 1.0.
        # Default thresholds: high=0.6 → CE >= 0.6 → rejected.
        default_result = build_ground_truth(outputs)
        assert default_result["source"] is None

        # Override ce_threshold_high to 1.5 so CE falls below it → low.
        custom_result = build_ground_truth(
            outputs,
            config={"ce_threshold_low": 1.5, "ce_threshold_high": 1.5},
        )
        # CE is still 1.0, but now thresholds are both 1.5 so CE < low → auto-accept.
        assert custom_result["source"] == "auto_consensus"


# ═══════════════════════════════════════════════════════════════════════════════
# _align_engine_outputs
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlignEngineOutputs:
    """Multi-engine word-level alignment."""

    def test_alignment_multiple_engines(self) -> None:
        """Three engines with slightly different texts → aligned."""
        outputs = [
            {
                "engine": "tesseract",
                "words": [
                    {"text": "hello", "confidence": 0.9},
                    {"text": "world", "confidence": 0.85},
                ],
            },
            {
                "engine": "gcp",
                "words": [
                    {"text": "hello", "confidence": 0.95},
                    {"text": "earth", "confidence": 0.8},
                ],
            },
            {
                "engine": "textract",
                "words": [
                    {"text": "hello", "confidence": 0.88},
                    {"text": "world", "confidence": 0.92},
                ],
            },
        ]
        result = _align_engine_outputs(outputs)
        assert result["num_engines"] == 3
        assert result["num_positions"] == 2
        assert len(result["aligned_words"]) == 2
        # First position: all engines have "hello".
        pos0 = result["aligned_words"][0]
        assert pos0[0]["text"] == "hello"
        assert pos0[1]["text"] == "hello"
        assert pos0[2]["text"] == "hello"
        # Second position: tesseract="world", gcp="earth", textract="world".
        pos1 = result["aligned_words"][1]
        assert pos1[0]["text"] == "world"
        assert pos1[1]["text"] == "earth"
        assert pos1[2]["text"] == "world"

    def test_alignment_empty(self) -> None:
        """Empty engine list → empty alignment."""
        result = _align_engine_outputs([])
        assert result["num_positions"] == 0

    def test_alignment_no_words(self) -> None:
        """Engine with empty word list → no positions."""
        outputs = [
            {"engine": "tesseract", "words": []},
            {"engine": "gcp", "words": [{"text": "hello"}]},
        ]
        result = _align_engine_outputs(outputs)
        assert result["num_positions"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _majority_vote
# ═══════════════════════════════════════════════════════════════════════════════


class TestMajorityVote:
    """Majority vote at aligned positions."""

    def test_majority_vote(self) -> None:
        """Three engines, two agree → that text wins."""
        aligned = [
            [
                {"text": "hello", "confidence": 0.9},
                {"text": "hello", "confidence": 0.95},
                {"text": "world", "confidence": 0.85},
            ],
        ]
        result = _majority_vote(aligned)
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_majority_vote_tie(self) -> None:
        """2-2 tie → confidence-weighted breaks it."""
        aligned = [
            [
                {"text": "hello", "confidence": 0.3},
                {"text": "hello", "confidence": 0.2},
                {"text": "world", "confidence": 0.95},
                {"text": "world", "confidence": 0.9},
            ],
        ]
        result = _majority_vote(aligned)
        assert len(result) == 1
        # world has total confidence 1.85, hello has 0.5 → world wins
        assert result[0]["text"] == "world"

    def test_majority_vote_unanimous(self) -> None:
        """All engines agree → that text wins."""
        aligned = [
            [
                {"text": "hello", "confidence": 0.9},
                {"text": "hello", "confidence": 0.95},
                {"text": "hello", "confidence": 0.85},
            ],
        ]
        result = _majority_vote(aligned)
        assert result[0]["text"] == "hello"
        assert result[0]["vote_count"] == 3

    def test_majority_vote_empty(self) -> None:
        """No words → empty result."""
        result = _majority_vote([])
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateCandidateGT:
    """Ground truth validation."""

    def test_validate_candidate_gt_valid(self) -> None:
        """Valid GT data passes all checks."""
        gt = {
            "source": "auto_consensus",
            "source_config": {
                "engines_used": ["tesseract", "gcp"],
                "ce_threshold_low": 0.2,
                "ce_threshold_high": 0.6,
            },
            "consensus_entropy": 0.05,
            "pages": [
                {
                    "blocks": [
                        {
                            "type": "text",
                            "bbox": [0.0, 0.0, 0.0, 0.0],
                            "confidence": 1.0,
                            "order": 0,
                            "lines": [],
                        }
                    ],
                    "tables": [],
                }
            ],
            "needs_review": False,
            "warnings": [],
        }
        assert _validate_candidate_gt(gt) is True

    def test_validate_candidate_gt_invalid_source(self) -> None:
        """Unknown source → invalid."""
        gt = {
            "source": "invalid_source",
            "consensus_entropy": 0.0,
            "pages": [{"blocks": [], "tables": []}],
            "needs_review": False,
            "warnings": [],
        }
        assert _validate_candidate_gt(gt) is False

    def test_validate_candidate_gt_entropy_out_of_range(self) -> None:
        """Entropy outside [0, 1] → invalid."""
        gt = {
            "source": "auto_consensus",
            "consensus_entropy": 1.5,
            "pages": [{"blocks": [], "tables": []}],
            "needs_review": False,
            "warnings": [],
        }
        assert _validate_candidate_gt(gt) is False

    def test_validate_candidate_gt_missing_pages(self) -> None:
        """Missing pages key → invalid."""
        gt = {
            "source": "auto_consensus",
            "consensus_entropy": 0.0,
            "needs_review": False,
            "warnings": [],
        }
        assert _validate_candidate_gt(gt) is False

    def test_validate_candidate_gt_page_missing_blocks(self) -> None:
        """Page missing 'blocks' key → invalid."""
        gt = {
            "source": "auto_consensus",
            "consensus_entropy": 0.0,
            "pages": [{"tables": []}],
            "needs_review": False,
            "warnings": [],
        }
        assert _validate_candidate_gt(gt) is False

    def test_validate_candidate_gt_source_none_valid(self) -> None:
        """source=None is valid (high-entropy rejection case)."""
        gt = {
            "source": None,
            "consensus_entropy": 0.8,
            "pages": [],
            "needs_review": True,
            "warnings": ["High entropy"],
        }
        assert _validate_candidate_gt(gt) is True

    def test_validate_candidate_gt_needs_review_not_bool(self) -> None:
        """needs_review must be bool."""
        gt = {
            "source": "auto_consensus",
            "consensus_entropy": 0.0,
            "pages": [{"blocks": [], "tables": []}],
            "needs_review": "yes",
            "warnings": [],
        }
        assert _validate_candidate_gt(gt) is False

    def test_validate_candidate_gt_warnings_not_list(self) -> None:
        """warnings must be a list."""
        gt = {
            "source": "auto_consensus",
            "consensus_entropy": 0.0,
            "pages": [{"blocks": [], "tables": []}],
            "needs_review": False,
            "warnings": "not a list",
        }
        assert _validate_candidate_gt(gt) is False

    def test_validate_candidate_gt_empty_pages(self) -> None:
        """Empty pages list is valid (high-entropy case)."""
        gt = {
            "source": None,
            "consensus_entropy": 0.9,
            "pages": [],
            "needs_review": True,
            "warnings": ["High CE"],
        }
        assert _validate_candidate_gt(gt) is True


# ═══════════════════════════════════════════════════════════════════════════════
# _align_word_sequences (internal helper)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlignWordSequences:
    """Multi-sequence word alignment helper."""

    def test_align_word_sequences_identical(self) -> None:
        """Identical sequences → each word aligns perfectly."""
        seqs = [
            ["hello", "world"],
            ["hello", "world"],
            ["hello", "world"],
        ]
        aligned = _align_word_sequences(seqs)
        assert len(aligned) == 2
        assert aligned[0] == ["hello", "hello", "hello"]
        assert aligned[1] == ["world", "world", "world"]

    def test_align_word_sequences_with_gaps(self) -> None:
        """Shorter sequences get gaps in aligned positions."""
        seqs = [
            ["the", "quick", "brown", "fox"],
            ["the", "brown", "fox"],
        ]
        aligned = _align_word_sequences(seqs)
        # "the" matches, "quick" is deleted in engine1, "brown" matches, "fox" matches.
        assert len(aligned) == 4
        # Position 0: both have "the"
        assert aligned[0] == ["the", "the"]
        # Position 1: engine0 has "quick", engine1 has gap (None)
        assert aligned[1] == ["quick", None]
        # Remaining positions match
        assert aligned[3] == ["fox", "fox"]

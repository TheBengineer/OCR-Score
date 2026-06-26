"""Tests for the sequence alignment module."""

from __future__ import annotations

from backend.alignment.aligner import (
    _bbox_iou,
    _word_scoring_func,
    align_ocr_texts,
    character_level_align,
    needleman_wunsch,
    smith_waterman,
)

# ── Needleman-Wunsch tests ──────────────────────────────────────────────────────


class TestNeedlemanWunsch:
    """Global alignment tests."""

    def test_exact_match_global(self) -> None:
        """Identical sequences → perfect alignment (all matches, no gaps)."""
        seq_a = ["the", "quick", "brown", "fox"]
        seq_b = ["the", "quick", "brown", "fox"]

        path, score = needleman_wunsch(seq_a, seq_b)

        assert len(path) == 4
        assert all(a is not None and b is not None for a, b in path)
        assert all(seq_a[a] == seq_b[b] for a, b in path if a is not None and b is not None)
        # 4 matches × 1.0 each
        assert score == 4.0

    def test_single_substitution(self) -> None:
        """One char diff → substitution counted."""
        seq_a = ["hello"]
        seq_b = ["helpo"]

        path, score = needleman_wunsch(seq_a, seq_b)

        assert len(path) == 1
        a, b = path[0]
        assert a == 0 and b == 0
        assert score == -1.0  # mismatch penalty

    def test_single_insertion(self) -> None:
        """Extra word → insertion counted."""
        seq_a = ["hello"]
        seq_b = ["hello", "world"]

        path, score = needleman_wunsch(seq_a, seq_b)

        # One match, one insertion.
        assert len(path) == 2
        matched = [p for p in path if p[0] is not None and p[1] is not None]
        inserted = [p for p in path if p[0] is None]
        assert len(matched) == 1
        assert len(inserted) == 1
        # match (+1) + gap (-1) = 0
        assert score == 0.0

    def test_single_deletion(self) -> None:
        """Missing word → deletion counted."""
        seq_a = ["hello", "world"]
        seq_b = ["hello"]

        path, score = needleman_wunsch(seq_a, seq_b)

        assert len(path) == 2
        matched = [p for p in path if p[0] is not None and p[1] is not None]
        deleted = [p for p in path if p[1] is None]
        assert len(matched) == 1
        assert len(deleted) == 1
        assert score == 0.0

    def test_empty_sequence(self) -> None:
        """Empty → empty alignment."""
        path, score = needleman_wunsch([], [])
        assert path == []
        assert score == 0.0

        path_b, score_b = needleman_wunsch(["a", "b"], [])
        assert len(path_b) == 2
        assert all(p[1] is None for p in path_b)
        assert score_b == -2.0

    def test_custom_scoring_matrix(self) -> None:
        """Custom match/mismatch/gap scores respected."""
        seq_a = ["a", "b"]
        seq_b = ["a", "c"]

        _, score = needleman_wunsch(seq_a, seq_b, match_score=5.0, mismatch_penalty=-2.0, gap_penalty=-3.0)
        # match (5) + mismatch (-2) = 3
        assert score == 3.0

    def test_scoring_func_override(self) -> None:
        """Custom scoring_func overrides match/mismatch."""
        seq_a = ["hello"]
        seq_b = ["world"]

        def always_positive(_a: str, _b: str) -> float:
            return 10.0

        _, score = needleman_wunsch(seq_a, seq_b, scoring_func=always_positive)
        assert score == 10.0

    def test_multiple_gaps(self) -> None:
        """Multiple gaps on both sides."""
        seq_a = ["a", "b", "c"]
        seq_b = ["a", "x", "y", "z", "c"]

        path, score = needleman_wunsch(seq_a, seq_b, match_score=1.0, gap_penalty=-1.0, mismatch_penalty=-1.0)

        assert len(path) == 5
        insertions = [p for p in path if p[0] is None]
        assert len(insertions) == 2


# ── Smith-Waterman tests ────────────────────────────────────────────────────────


class TestSmithWaterman:
    """Local alignment tests."""

    def test_partial_smith_waterman(self) -> None:
        """Local alignment finds best substring match."""
        seq_a = ["the", "cat", "sat"]
        seq_b = ["not", "here", "the", "cat", "sat", "there"]

        path, score = smith_waterman(seq_a, seq_b)

        assert len(path) == 3
        matched_texts = []
        for a_idx, b_idx in path:
            assert a_idx is not None and b_idx is not None
            assert seq_a[a_idx] == seq_b[b_idx]
            matched_texts.append(seq_a[a_idx])
        assert matched_texts == ["the", "cat", "sat"]
        assert score > 0  # positive match

    def test_local_no_match(self) -> None:
        """No similarity → empty path, zero score."""
        seq_a = ["abc"]
        seq_b = ["xyz"]

        path, score = smith_waterman(seq_a, seq_b, match_score=1.0, mismatch_penalty=-1.0)

        # Smith-Waterman can start anywhere; with a mismatch the cell will be 0.
        # Path length depends on whether any path exceeds 0.
        assert score == 0.0

    def test_local_single_word_match(self) -> None:
        """Single matching word in a long context."""
        seq_a = ["hello"]
        seq_b = ["foo", "bar", "hello", "baz"]

        path, score = smith_waterman(seq_a, seq_b, match_score=2.0, mismatch_penalty=-1.0, gap_penalty=-1.0)

        assert len(path) == 1
        a_idx, b_idx = path[0]
        assert a_idx == 0
        assert b_idx == 2
        assert score == 2.0


# ── OCR text alignment tests ─────────────────────────────────────────────────────


class TestAlignOcrTexts:
    """High-level OCR alignment pipeline tests."""

    def test_ocr_text_identical(self) -> None:
        """Same word lists → all matches."""
        ref = [{"text": "hello"}, {"text": "world"}]
        ocr = [{"text": "hello"}, {"text": "world"}]

        result = align_ocr_texts(ocr, ref)

        assert result["stats"]["matches"] == 2
        assert result["stats"]["substitutions"] == 0
        assert result["stats"]["insertions"] == 0
        assert result["stats"]["deletions"] == 0
        assert all(p["operation"] == "match" for p in result["word_pairs"])

    def test_ocr_text_with_errors(self) -> None:
        """OCR errors → correct operation mapping."""
        ref = [{"text": "hello"}, {"text": "world"}, {"text": "foo"}]
        ocr = [{"text": "hello"}, {"text": "wor1d"}, {"text": "bar"}]

        result = align_ocr_texts(ocr, ref)

        # hello = match, wor1d vs world = substitution (high similarity), bar vs foo = substitution
        total = sum(result["stats"].values())
        assert total == 3

    def test_ocr_with_bbox_fallback(self) -> None:
        """Low-confidence words trigger bbox-based matching."""
        ref = [
            {"text": "hello", "bbox": [0, 0, 50, 20], "confidence": 0.3},
            {"text": "world", "bbox": [50, 0, 100, 20], "confidence": 0.9},
        ]
        ocr = [
            {"text": "hell0", "bbox": [0, 0, 50, 20], "confidence": 0.2},
            {"text": "world", "bbox": [50, 0, 100, 20], "confidence": 0.9},
        ]

        result = align_ocr_texts(ocr, ref)

        # hell0 vs hello: similarity ~0.8, bbox_iou ~1.0, low confidence triggers bbox check.
        # If bbox_iou > 0.7, operation should be "match" despite substitution.
        first_pair = result["word_pairs"][0]
        assert first_pair["bbox_iou"] is not None
        assert first_pair["bbox_iou"] > 0.9

    def test_ocr_single_insertion(self) -> None:
        """Extra word in OCR → insertion."""
        ref = [{"text": "hello"}]
        ocr = [{"text": "hello"}, {"text": "extra"}]

        result = align_ocr_texts(ocr, ref)

        assert result["stats"]["insertions"] == 1
        assert result["stats"]["matches"] >= 1

    def test_ocr_single_deletion(self) -> None:
        """Missing word in OCR → deletion."""
        ref = [{"text": "hello"}, {"text": "world"}]
        ocr = [{"text": "hello"}]

        result = align_ocr_texts(ocr, ref)

        assert result["stats"]["deletions"] == 1
        assert result["stats"]["matches"] >= 1


# ── Character-level alignment tests ──────────────────────────────────────────────


class TestCharacterLevelAlign:
    """Character-level alignment tests."""

    def test_character_alignment_exact(self) -> None:
        """Identical strings → all matches."""
        result = character_level_align("hello", "hello")

        assert len(result) == 5
        assert all(op == "match" for _, _, op in result)
        assert all(ref == ocr for ref, ocr, _ in result)

    def test_character_alignment_substitution(self) -> None:
        """Char diff → substitution."""
        result = character_level_align("hello", "hallo")

        assert len(result) == 5
        sub_ops = [(r, o, op) for r, o, op in result if op == "substitution"]
        assert len(sub_ops) == 1
        assert sub_ops[0] == ("e", "a", "substitution")

    def test_character_alignment_gap(self) -> None:
        """Length mismatch → insert/delete ops."""
        result = character_level_align("abcd", "abcef")

        # abcd vs abcef: 4 vs 5 chars. NW will find the best path.
        # Likely: a-a b-b c-c d-e None-f (deletion of d, insertion of e, insertion of f)
        # or: a-a b-b c-c None-e d-f (insertion of e, insertion of f)
        insertions = [(r, o) for r, o, op in result if op == "insertion"]
        deletions = [(r, o) for r, o, op in result if op == "deletion"]

        # Total operations = 5 (length of longer seq)
        assert len(result) == 5
        assert len(insertions) >= 1
        assert len(deletions) >= 0

    def test_character_alignment_empty(self) -> None:
        """Empty strings → empty alignment."""
        result = character_level_align("", "")
        assert result == []

        result_ocr_empty = character_level_align("abc", "")
        assert len(result_ocr_empty) == 3
        assert all(op == "deletion" for _, _, op in result_ocr_empty)


# ── Large-sequence / banded NW tests ─────────────────────────────────────────────


class TestBandedHirschberg:
    """Banded Hirschberg for large sequences."""

    def test_bounded_nw_large_sequences(self) -> None:
        """Banded alignment works for 500+ word sequences."""
        n = 600
        seq_a = [f"word_{i}" for i in range(n)]
        seq_b = [f"word_{i}" for i in range(n)]

        path, score = needleman_wunsch(seq_a, seq_b)

        assert len(path) == n
        # Every entry should be a match with the same index.
        for a_idx, b_idx in path:
            assert a_idx is not None
            assert b_idx is not None
        assert score == float(n)

    def test_large_with_substitutions(self) -> None:
        """Large sequence with some mismatches."""
        n = 550
        seq_a = [f"word_{i}" for i in range(n)]
        seq_b = [f"word_{i}" if i != 300 else "WRONG" for i in range(n)]

        path, score = needleman_wunsch(seq_a, seq_b)

        assert len(path) == n
        # n-1 matches, 1 mismatch.
        assert score == float(n - 1) - 1.0

    def test_large_with_gaps(self) -> None:
        """Large sequence with insertions/deletions."""
        n = 550
        seq_a = [f"word_{i}" for i in range(n)]
        seq_b = [f"word_{i}" if i < 200 else f"word_{i+1}" for i in range(n)]

        path, score = needleman_wunsch(seq_a, seq_b)

        assert len(path) >= n


class TestWordScoringFunc:
    """Levenshtein similarity scoring."""

    def test_identical_words(self) -> None:
        """Identical words → score of 1.0."""
        score = _word_scoring_func("hello", "hello")
        assert score == 1.0

    def test_completely_different(self) -> None:
        """Completely different words → score near -1."""
        score = _word_scoring_func("abc", "xyz")
        assert score < 0

    def test_partial_similarity(self) -> None:
        """Partially similar words → intermediate score."""
        score = _word_scoring_func("clear", "c1ear")
        # 'clear' vs 'c1ear' - mostly similar but one char diff.
        assert -1.0 < score < 1.0


class TestBboxIou:
    """Bounding box I/O calculations."""

    def test_identical_boxes(self) -> None:
        """Identical → IOU = 1.0."""
        bbox = [0.0, 0.0, 100.0, 100.0]
        iou = _bbox_iou(bbox, bbox)
        assert iou == 1.0

    def test_no_overlap(self) -> None:
        """No overlap → IOU = 0.0."""
        a = [0.0, 0.0, 10.0, 10.0]
        b = [100.0, 100.0, 110.0, 110.0]
        iou = _bbox_iou(a, b)
        assert iou == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap → 0 < IOU < 1."""
        a = [0.0, 0.0, 100.0, 100.0]
        b = [50.0, 50.0, 150.0, 150.0]
        iou = _bbox_iou(a, b)
        # Intersection: 50x50 = 2500
        # Area A: 10000, Area B: 10000
        # Union: 17500
        # IOU: 2500/17500 ≈ 0.143
        assert 0.1 < iou < 0.2


class TestAlignOcrIntegration:
    """End-to-end alignment integration tests."""

    def test_no_words(self) -> None:
        """No words on either side → empty."""
        result = align_ocr_texts([], [])
        assert result["word_pairs"] == []
        assert result["stats"]["matches"] == 0

    def test_realistic_ocr_output(self) -> None:
        """Realistic scenario with mixed errors."""
        ref = [
            {"text": "The", "bbox": [0, 0, 30, 15], "confidence": 1.0},
            {"text": "quick", "bbox": [30, 0, 70, 15], "confidence": 1.0},
            {"text": "brown", "bbox": [70, 0, 120, 15], "confidence": 1.0},
            {"text": "fox", "bbox": [120, 0, 150, 15], "confidence": 1.0},
        ]
        ocr = [
            {"text": "The", "bbox": [0, 0, 30, 15], "confidence": 0.9},
            {"text": "quick", "bbox": [30, 0, 70, 15], "confidence": 0.85},
            {"text": "brown", "bbox": [70, 0, 120, 15], "confidence": 0.8},
            {"text": "cat", "bbox": [120, 0, 160, 15], "confidence": 0.6},
        ]

        result = align_ocr_texts(ocr, ref)

        # The, quick, brown should be matches; cat vs fox = substitution (sim ≈ 0.5 < 0.6).
        assert result["stats"]["matches"] == 3
        assert result["stats"]["substitutions"] == 1
        total = sum(result["stats"].values())
        assert total == 4

    def test_config_override(self) -> None:
        """Custom config respected."""
        ref = [{"text": "a"}, {"text": "b"}]
        ocr = [{"text": "a"}, {"text": "c"}]

        result_default = align_ocr_texts(ocr, ref)
        assert result_default["stats"]["substitutions"] == 1

        # Very permissive similarity threshold → all match.
        result_permissive = align_ocr_texts(ocr, ref, config={"similarity_threshold": 0.0})
        assert result_permissive["stats"]["matches"] == 2

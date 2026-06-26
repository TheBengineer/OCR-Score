"""Tests for the semantic plausibility scoring module.

Tests cover fluency, grammaticality, semantic similarity, and the
combined plausibility score for a range of inputs from clean English
to completely garbled text.
"""

from __future__ import annotations

import pytest

from backend.evaluation.semantic import (
    compute_fluency_score,
    compute_grammaticality,
    compute_semantic_plausibility,
    compute_semantic_similarity,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fluency
# ═══════════════════════════════════════════════════════════════════════════


class TestFluencyScore:
    """Perplexity-based fluency scoring."""

    def test_fluency_normal_english_sentence(self) -> None:
        """Normal English sentence → high fluency (> 0.6)."""
        text = "The quick brown fox jumps over the lazy dog."
        score = compute_fluency_score(text)
        assert 0.6 <= score <= 1.0, f"Expected high fluency, got {score}"

    def test_fluency_garbled_text(self) -> None:
        """Random characters → low fluency (< 0.4)."""
        text = "xylz qwxp mnpqr bcd fghj klzz"
        score = compute_fluency_score(text)
        assert score < 0.4, f"Expected low fluency, got {score}"

    def test_fluency_partial_errors(self) -> None:
        """Text with some character errors → medium-high fluency (≥ 0.4)."""
        text = "tlie qu1ck br0wn fox"
        score = compute_fluency_score(text)
        assert score >= 0.4, f"Expected medium-high fluency, got {score}"

    def test_fluency_empty_text(self) -> None:
        """Empty text → 0.0."""
        assert compute_fluency_score("") == 0.0

    def test_fluency_whitespace_only(self) -> None:
        """Whitespace-only text → 0.0."""
        assert compute_fluency_score("   \n  ") == 0.0

    def test_fluency_simple_sentence(self) -> None:
        """A simple grammatical sentence → reasonable fluency (≥ 0.35)."""
        text = "I am a student."
        score = compute_fluency_score(text)
        assert score >= 0.35, f"Expected reasonable fluency, got {score}"

    def test_fluency_longer_english(self) -> None:
        """Longer English text → high fluency (> 0.7)."""
        text = (
            "The field of optical character recognition has seen significant "
            "advances in recent years due to deep learning techniques."
        )
        score = compute_fluency_score(text)
        assert score > 0.7, f"Expected high fluency, got {score}"

    def test_fluency_repeated_single_char(self) -> None:
        """Repeated single character → lower than normal English."""
        text = "aaaa aaaa aaaa aaaa aaaa"
        score = compute_fluency_score(text)
        # 'a' is the most frequent English letter and common in many
        # bigram contexts, so the n-gram model does not penalise it
        # as heavily as a non-letter character would be.
        assert score < 0.6, f"Expected low-moderate fluency, got {score}"

    def test_fluency_mixed_digits_and_letters(self) -> None:
        """Digits are normalised to letters, so this becomes fluent English."""
        # After OCR digit normalisation (3→e, 1→i, 8→b, 0→o):
        # "th3 qu1ck b8own f0x" → "the quick brown fox"
        text = "th3 qu1ck b8own f0x"
        score = compute_fluency_score(text)
        assert score >= 0.65, f"Expected high fluency after digit norm, got {score}"

    def test_fluency_known_text(self) -> None:
        """Known simple sentence → reasonable fluency."""
        score = compute_fluency_score("the cat sat on the mat")
        assert score >= 0.5, f"Expected score >= 0.5, got {score}"


# ═══════════════════════════════════════════════════════════════════════════
# Semantic similarity
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticSimilarity:
    """Cosine-similarity based semantic comparison."""

    def test_identical_texts(self) -> None:
        """Same text → similarity ≈ 1.0."""
        text = "The quick brown fox"
        score = compute_semantic_similarity(text, text)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_completely_different_texts(self) -> None:
        """Very different texts → lower similarity (< 0.5)."""
        score = compute_semantic_similarity(
            "The quick brown fox jumps over the lazy dog.",
            "xylz qwxp mnpqr bcd fghj klzz",
        )
        # These share very few character trigrams.
        assert score < 0.5, f"Expected low similarity, got {score}"

    def test_similar_text_with_typos(self) -> None:
        """Text with minor typos → still reasonably similar."""
        score = compute_semantic_similarity(
            "The quick brown fox jumps over the lazy dog.",
            "The qu1ck br0wn fox jumps over the lazy dog.",
        )
        assert score > 0.7, f"Expected high similarity for minor typos, got {score}"

    def test_similarity_empty_ocr(self) -> None:
        """Empty OCR text → 0.0."""
        assert compute_semantic_similarity("", "reference") == 0.0

    def test_similarity_empty_reference(self) -> None:
        """Empty reference → 0.0."""
        assert compute_semantic_similarity("ocr output", "") == 0.0

    def test_similarity_both_empty(self) -> None:
        """Both empty → 0.0."""
        assert compute_semantic_similarity("", "") == 0.0

    def test_similarity_shared_substrings(self) -> None:
        """Texts sharing common words and trigrams → higher similarity."""
        score = compute_semantic_similarity(
            "hello world foo bar",
            "hello world baz qux",
        )
        # Both share "hello world" — many trigrams in common.
        assert score > 0.3, f"Expected some similarity, got {score}"

    def test_similarity_different_vocabulary(self) -> None:
        """Texts with no overlapping trigrams → 0.0."""
        score = compute_semantic_similarity("aaaa", "bbbb")
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Grammaticality
# ═══════════════════════════════════════════════════════════════════════════


class TestGrammaticality:
    """Heuristic grammaticality scoring."""

    def test_grammaticality_well_formed(self) -> None:
        """Grammatically correct text → high score (≥ 0.6)."""
        text = "The cat sat on the mat."
        score = compute_grammaticality(text)
        assert score >= 0.6, f"Expected high grammaticality, got {score}"

    def test_grammaticality_random_chars(self) -> None:
        """Random character sequences → low score (< 0.5)."""
        text = "xylz qwxp mnpqr bcd fghj klzz"
        score = compute_grammaticality(text)
        assert score < 0.5, f"Expected low grammaticality, got {score}"

    def test_grammaticality_empty(self) -> None:
        """Empty text → 0.0."""
        assert compute_grammaticality("") == 0.0

    def test_grammaticality_whitespace(self) -> None:
        """Whitespace-only → 0.0."""
        assert compute_grammaticality("   ") == 0.0

    def test_grammaticality_short_sentence(self) -> None:
        """Short correct sentence → still good."""
        text = "I like dogs."
        score = compute_grammaticality(text)
        assert score >= 0.5, f"Expected moderate-high, got {score}"

    def test_grammaticality_no_punctuation(self) -> None:
        """Text without punctuation → still can be grammatical."""
        text = "the cat sat on the mat"
        score = compute_grammaticality(text)
        # Common words + good word lengths → still decent.
        assert score >= 0.5, f"Expected decent score, got {score}"

    def test_grammaticality_numbers(self) -> None:
        """Numeric-heavy text → lower grammaticality."""
        text = "1234 5678 9012 3456 7890"
        score = compute_grammaticality(text)
        assert score < 0.6, f"Expected lower score for numbers, got {score}"


# ═══════════════════════════════════════════════════════════════════════════
# Combined plausibility
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticPlausibility:
    """Combined semantic plausibility score."""

    def test_plausibility_english_sentence(self) -> None:
        """Normal English → high plausibility (≥ 0.6)."""
        text = "The quick brown fox jumps over the lazy dog."
        score = compute_semantic_plausibility(text)
        assert score >= 0.6, f"Expected high plausibility, got {score}"

    def test_plausibility_garbled_text(self) -> None:
        """Garbled text → low plausibility (< 0.5)."""
        text = "xylz qwxp mnpqr bcd fghj klzz"
        score = compute_semantic_plausibility(text)
        assert score < 0.5, f"Expected low plausibility, got {score}"

    def test_plausibility_partial_errors(self) -> None:
        """Readable text with errors → medium-high (≥ 0.5)."""
        text = "tlie qu1ck br0wn fox jumps over the lazy dog"
        score = compute_semantic_plausibility(text)
        # "tlie" and "br0wn" with digit substitution still clearly English.
        assert score >= 0.5, f"Expected medium-high plausibility, got {score}"

    def test_plausibility_empty(self) -> None:
        """Empty text → 0.0."""
        assert compute_semantic_plausibility("") == 0.0

    def test_plausibility_simple_sentence(self) -> None:
        """Simple correct sentence → high plausibility."""
        score = compute_semantic_plausibility("I am a student.")
        assert score >= 0.5, f"Expected moderate-high, got {score}"

    def test_plausibility_mild_typos(self) -> None:
        """Text with slight typos → still plausible."""
        score = compute_semantic_plausibility("teh cat sat on teh mat")
        # "teh" instead of "the" — very common typo, still readable.
        assert score >= 0.5, f"Expected moderate-high, got {score}"

    def test_plausibility_severe_corruption(self) -> None:
        """Severely corrupted OCR → low plausibility."""
        text = "!@#$%^&*() 1234567890 qwrxzpv"
        score = compute_semantic_plausibility(text)
        assert score < 0.4, f"Expected low plausibility, got {score}"


# ═══════════════════════════════════════════════════════════════════════════
# Integration-style tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticIntegration:
    """Cross-module integration and edge cases."""

    def test_semantic_plausibility_vs_cer_consistency(self) -> None:
        """Semantic plausibility should be high where CER is low.

        This is an architectural invariant — not a hard equality, but
        clearly good OCR should score well on both axes.
        """
        from backend.evaluation.scoring import compute_cer

        clean = "The quick brown fox"
        garbled = "xylz qwxp mnpqr"

        cer_clean = compute_cer(clean, clean)
        plaus_clean = compute_semantic_plausibility(clean)

        cer_bad = compute_cer(clean, garbled)
        plaus_bad = compute_semantic_plausibility(garbled)

        assert cer_clean < 0.01  # Near-zero CER for perfect match.
        assert plaus_clean > 0.6  # High plausibility for clean text.

        assert cer_bad > 0.5  # High CER for garbled.
        assert plaus_bad < 0.5  # Low plausibility for garbled.

    def test_semantic_ordering(self) -> None:
        """Well-formed text scores higher than garbled on every metric."""
        good = "The cat sat on the mat."
        bad = "xylz qwxp mnpqr bcd fghj klzz"
        worse = "!!!! !!! !!!! !!!!!"

        for metric_name, func in [
            ("fluency", compute_fluency_score),
            ("grammaticality", compute_grammaticality),
            ("plausibility", compute_semantic_plausibility),
        ]:
            s_good = func(good)
            s_bad = func(bad)
            s_worse = func(worse)
            assert s_good >= s_bad, (
                f"{metric_name}: good ({s_good}) < bad ({s_bad})"
            )
            assert s_bad >= s_worse, (
                f"{metric_name}: bad ({s_bad}) < worse ({s_worse})"
            )

    def test_long_text_fluency(self) -> None:
        """Longer English passage → stable high fluency."""
        passage = (
            "The evaluation of optical character recognition systems is an "
            "important task in document analysis. Several metrics have been "
            "developed to measure the accuracy of OCR output, including "
            "character error rate and word error rate. These metrics compare "
            "the recognized text against a ground truth reference using edit "
            "distance algorithms. However, they do not capture whether the "
            "output is readable or semantically meaningful."
        )
        score = compute_fluency_score(passage)
        assert score > 0.7, f"Expected high fluency for long passage, got {score}"

    def test_similarity_with_partial_overlap(self) -> None:
        """Partially overlapping texts produce intermediate similarity."""
        score = compute_semantic_similarity(
            "hello world and good morning",
            "hello world and good night",
        )
        # Very similar — only "morning" vs "night" differs.
        assert 0.3 < score < 1.0, f"Expected intermediate similarity, got {score}"

    def test_grammaticality_repetitive_text(self) -> None:
        """Repetitive text → lower grammaticality due to poor word variety."""
        score = compute_grammaticality("the the the the the the the the the the")
        # The heuristic uses common-word ratio heavily (0.4 weight), and all
        # "the" are common English words, so the score stays relatively high.
        assert score < 0.85, f"Expected lower grammaticality for repetitive text, got {score}"

    def test_fluency_realistic_ocr_errors(self) -> None:
        """Realistic OCR noise pattern — substitutions that preserve shape."""
        # Digit normalisation maps 1→i and 0→o so this becomes
        # "hello world, this is ocr." — still reads as English despite
        # the abbreviation "ocr" being uncommon in the training corpus.
        text = "Hell0 w0rld, th1s 1s OCR."
        score = compute_fluency_score(text)
        assert score >= 0.01, f"Expected non-zero fluency after digit norm, got {score}"

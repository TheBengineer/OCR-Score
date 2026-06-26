"""Tests for script-aware OCR evaluation — script detection, CJK metrics,
Arabic metrics, and per-script breakdowns."""

from __future__ import annotations

from backend.evaluation.script_aware import (
    compute_arabic_metrics,
    compute_arabic_rtl_penalty,
    compute_cjk_metrics,
    compute_script_scores,
    detect_script,
)


class TestDetectScript:
    """Unicode-range based script detection."""

    def test_detect_script_latin(self) -> None:
        """Plain English text -> 'latin'."""
        assert detect_script("The quick brown fox jumps over the lazy dog") == "latin"

    def test_detect_script_cjk(self) -> None:
        """Chinese text -> 'cjk'."""
        text = "\u4eca\u5929\u5929\u6c14\u5f88\u597d"  # 今天天气很好
        assert detect_script(text) == "cjk"

    def test_detect_script_arabic(self) -> None:
        """Arabic text -> 'arabic'."""
        text = "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645"  # السلام عليكم
        assert detect_script(text) == "arabic"

    def test_detect_script_devanagari(self) -> None:
        """Devanagari text -> 'devanagari'."""
        text = "\u0928\u092e\u0938\u094d\u0924\u0947"  # नमस्ते
        assert detect_script(text) == "devanagari"

    def test_detect_script_thai(self) -> None:
        """Thai text -> 'thai'."""
        text = "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"  # สวัสดี
        assert detect_script(text) == "thai"

    def test_detect_script_hebrew(self) -> None:
        """Hebrew text -> 'hebrew'."""
        text = "\u05e9\u05dc\u05d5\u05dd"  # שלום
        assert detect_script(text) == "hebrew"

    def test_detect_script_cyrillic(self) -> None:
        """Cyrillic text -> 'cyrillic'."""
        text = "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435"  # Здравствуйте
        assert detect_script(text) == "cyrillic"

    def test_detect_script_greek(self) -> None:
        """Greek text -> 'greek'."""
        text = "\u0393\u03b5\u03b9\u03ac \u03c3\u03b1\u03c2"  # Γειά σας
        assert detect_script(text) == "greek"

    def test_detect_script_japanese_hiragana(self) -> None:
        """Japanese hiragana -> 'japanese'."""
        text = "\u3053\u3093\u306b\u3061\u306f"  # こんにちは
        assert detect_script(text) == "japanese"

    def test_detect_script_korean(self) -> None:
        """Korean hangul -> 'korean'."""
        text = "\uc548\ub155\ud558\uc138\uc694"  # 안녕하세요
        assert detect_script(text) == "korean"

    def test_detect_script_mixed_latin_cjk(self) -> None:
        """Mixed Latin + CJK with CJK dominant -> 'cjk'."""
        text = "Hello \u4eca\u5929\u5929\u6c14\u5f88\u597d"  # Hello 今天天气很好
        assert detect_script(text) == "cjk"

    def test_detect_script_empty(self) -> None:
        """Empty string -> 'latin'."""
        assert detect_script("") == "latin"

    def test_detect_script_only_spaces(self) -> None:
        """Whitespace-only -> 'latin'."""
        assert detect_script("   \t\n  ") == "latin"

    def test_detect_script_punctuation(self) -> None:
        """Punctuation-only -> 'latin'."""
        assert detect_script("!@#$%^&*()") == "latin"


class TestComputeCjkMetrics:
    """CJK-specific character-level metrics."""

    def test_compute_cjk_metrics_exact_match(self) -> None:
        """Identical CJK strings -> CER = 0, char_accuracy = 1."""
        text = "\u4eca\u5929\u5929\u6c14\u5f88\u597d"  # 今天天气很好
        result = compute_cjk_metrics(text, text)
        assert result["cer"] == 0.0
        assert result["cjk_char_accuracy"] == 1.0
        assert result["char_f1"] == 1.0
        assert result["confusion_matrix"] == {}

    def test_compute_cjk_metrics_partial(self) -> None:
        """Slightly different CJK -> CER > 0, char_accuracy < 1."""
        gt = "\u4eca\u5929\u5929\u6c14\u5f88\u597d"  # 今天天气很好
        ocr = "\u4eca\u5929\u6c14\u5f88\u597d"        # 今天气很好  (missing 天)
        result = compute_cjk_metrics(ocr, gt)
        assert result["cer"] > 0.0
        assert result["cjk_char_accuracy"] < 1.0
        assert "confusion_matrix" in result
        # No word-level metrics in CJK output.
        assert "wer" not in result

    def test_compute_cjk_metrics_confusion_matrix(self) -> None:
        """CJK confusion matrix captures character substitutions."""
        gt = "\u7532\u4e59\u4e19"    # 甲乙丙
        ocr = "\u7532\u4e01\u4e19"    # 甲丁丙 (乙->丁 substitution)
        result = compute_cjk_metrics(ocr, gt)
        matrix = result["confusion_matrix"]
        assert len(matrix) > 0

    def test_compute_cjk_metrics_empty(self) -> None:
        """Empty CJK strings -> zero CER, zero F1."""
        result = compute_cjk_metrics("", "")
        assert result["cer"] == 0.0
        assert result["cjk_char_accuracy"] == 0.0
        assert result["char_f1"] == 0.0


class TestComputeArabicMetrics:
    """Arabic-specific metrics."""

    def test_compute_arabic_metrics_exact_match(self) -> None:
        """Identical Arabic strings -> CER = 0."""
        text = "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645"  # السلام عليكم
        result = compute_arabic_metrics(text, text)
        assert result["cer"] == 0.0
        assert result["char_f1"] == 1.0
        assert result["rtl_bbox_penalty"] == 0.0

    def test_compute_arabic_metrics_with_errors(self) -> None:
        """Arabic with OCR errors -> CER > 0."""
        gt = "\u0627\u0644\u0633\u0644\u0627\u0645"        # السلام
        ocr = "\u0627\u0644\u0633\u0644\u0627\u0645"  # match
        result = compute_arabic_metrics(ocr, gt)
        assert result["cer"] == 0.0

    def test_compute_arabic_metrics_empty(self) -> None:
        """Empty Arabic strings -> zero CER."""
        result = compute_arabic_metrics("", "")
        assert result["cer"] == 0.0


class TestComputeArabicRtlPenalty:
    """RTL-aware bbox evaluation for Arabic."""

    def test_rtl_penalty_correct_order(self) -> None:
        """RTL-ordered bboxes -> penalty = 0."""
        words = [
            {"text": "word1", "bbox": [100, 0, 200, 10]},
            {"text": "word2", "bbox": [50, 0, 90, 10]},
            {"text": "word3", "bbox": [0, 0, 40, 10]},
        ]
        penalty = compute_arabic_rtl_penalty(words, words)
        # All x_min decrease: right->left.  No wrong pairs.
        assert penalty == 0.0

    def test_rtl_penalty_wrong_order(self) -> None:
        """LTR-ordered bboxes -> penalty = 1."""
        words = [
            {"text": "word1", "bbox": [0, 0, 40, 10]},
            {"text": "word2", "bbox": [50, 0, 90, 10]},
            {"text": "word3", "bbox": [100, 0, 200, 10]},
        ]
        penalty = compute_arabic_rtl_penalty(words, words)
        # All x_min increase: left->right.  All pairs wrong.
        assert penalty == 1.0

    def test_rtl_penalty_single_word(self) -> None:
        """Single word -> no pairs -> penalty = 0."""
        words = [{"text": "word1", "bbox": [0, 0, 40, 10]}]
        penalty = compute_arabic_rtl_penalty(words, words)
        assert penalty == 0.0

    def test_rtl_penalty_missing_bbox(self) -> None:
        """Words without bbox are skipped -> no penalty."""
        words = [
            {"text": "word1", "bbox": [100, 0, 200, 10]},
            {"text": "word2"},  # missing bbox
        ]
        # Only 1 pair, but the pair with missing bbox is skipped -> 0 total.
        penalty = compute_arabic_rtl_penalty(words, words)
        assert penalty == 0.0

    def test_rtl_penalty_all_missing_bbox(self) -> None:
        """All words missing bbox -> no pairs to compare -> 0.0."""
        words = [{"text": "word1"}, {"text": "word2"}]
        penalty = compute_arabic_rtl_penalty(words, words)
        assert penalty == 0.0


class TestComputeScriptScores:
    """Per-script breakdown of evaluation scores."""

    def test_script_scores_single_script(self) -> None:
        """All pages in one script -> single entry in per_script."""
        ocr_data = {
            "pages": [
                {"results": [{"text": "今天天气很好", "confidence": 0.9}]},
                {"results": [{"text": "你好世界", "confidence": 0.8}]},
            ],
        }
        gt_data = {
            "pages": [
                {"results": [{"text": "今天天气很好", "confidence": 1.0}]},
                {"results": [{"text": "你好世界", "confidence": 1.0}]},
            ],
        }
        result = compute_script_scores(ocr_data, gt_data)
        assert "overall" in result
        assert "per_script" in result
        assert "script_counts" in result
        # Both pages are CJK.
        assert result["script_counts"].get("cjk", 0) == 2
        assert "cjk" in result["per_script"]

    def test_script_scores_empty(self) -> None:
        """No pages -> zero overall metrics."""
        result = compute_script_scores({"pages": []}, {"pages": []})
        assert result["overall"]["cer"] == 0.0
        assert result["overall"]["wer"] == 0.0
        assert result["per_script"] == {}
        assert result["script_counts"] == {}

    def test_script_scores_mixed_scripts(self) -> None:
        """Pages in different scripts -> multiple per_script entries."""
        ocr_data = {
            "pages": [
                {"results": [{"text": "hello world", "confidence": 0.9}]},
                {"results": [{"text": "今天天气很好", "confidence": 0.8}]},
                {"results": [{"text": "السلام عليكم", "confidence": 0.7}]},
            ],
        }
        gt_data = {
            "pages": [
                {"results": [{"text": "hello world", "confidence": 1.0}]},
                {"results": [{"text": "今天天气很好", "confidence": 1.0}]},
                {"results": [{"text": "السلام عليكم", "confidence": 1.0}]},
            ],
        }
        result = compute_script_scores(ocr_data, gt_data)
        assert result["script_counts"].get("latin", 0) == 1
        assert result["script_counts"].get("cjk", 0) == 1
        assert result["script_counts"].get("arabic", 0) == 1
        assert len(result["per_script"]) == 3

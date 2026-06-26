"""Tests for handwriting detection and tagging in OCR evaluation."""

from __future__ import annotations

from backend.evaluation.handwriting import (
    filter_handwriting_from_scoring,
    tag_handwriting_regions,
)


class TestTagHandwritingRegions:
    """Handwriting detection via heuristics."""

    def test_tag_handwriting_low_confidence_irregular_spacing(self) -> None:
        """Low confidence + irregular spacing -> tagged as handwriting."""
        output = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "bbox": [0, 0, 100, 10], "confidence": 0.3},
                        {"text": "world", "bbox": [5, 0, 105, 10], "confidence": 0.4},
                        {"text": "test", "bbox": [200, 0, 210, 10], "confidence": 0.9},
                    ],
                },
            ],
        }
        result = tag_handwriting_regions(output)
        assert result["has_handwriting"] is True
        # Low confidence words should be tagged.
        tagged = [w for w in result["pages"][0]["results"] if w.get("handwriting")]
        assert len(tagged) >= 2  # at least the low-conf words
        # High confidence word should NOT be tagged.
        high_conf = [w for w in result["pages"][0]["results"] if w["text"] == "test"]
        assert high_conf[0].get("handwriting") is not True

    def test_tag_handwriting_very_low_confidence(self) -> None:
        """Very low confidence (< 0.3) -> tagged even without spacing issues."""
        output = {
            "pages": [
                {
                    "results": [
                        {"text": "unclear", "bbox": [0, 0, 50, 10], "confidence": 0.15},
                    ],
                },
            ],
        }
        result = tag_handwriting_regions(output)
        assert result["pages"][0]["results"][0]["handwriting"] is True

    def test_tag_handwriting_engine_metadata(self) -> None:
        """Engine metadata indicating handwriting -> all tagged."""
        output = {
            "handwriting": True,
            "pages": [
                {
                    "results": [
                        {"text": "hello", "bbox": [0, 0, 30, 10], "confidence": 0.95},
                        {"text": "world", "bbox": [30, 0, 60, 10], "confidence": 0.92},
                    ],
                },
            ],
        }
        result = tag_handwriting_regions(output)
        assert result["has_handwriting"] is True
        for word in result["pages"][0]["results"]:
            assert word["handwriting"] is True

    def test_tag_handwriting_engine_metadata_string(self) -> None:
        """Engine metadata with writing_style='handwriting' -> all tagged."""
        output = {
            "writing_style": "handwriting",
            "pages": [
                {
                    "results": [
                        {"text": "hello", "confidence": 0.9},
                    ],
                },
            ],
        }
        result = tag_handwriting_regions(output)
        assert result["has_handwriting"] is True
        assert result["pages"][0]["results"][0]["handwriting"] is True

    def test_no_handwriting(self) -> None:
        """Clean text with high confidence -> nothing tagged."""
        output = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "bbox": [0, 0, 30, 10], "confidence": 0.95},
                        {"text": "world", "bbox": [30, 0, 60, 10], "confidence": 0.92},
                    ],
                },
            ],
        }
        result = tag_handwriting_regions(output)
        assert result["has_handwriting"] is False
        for word in result["pages"][0]["results"]:
            assert word.get("handwriting") is not True

    def test_tag_handwriting_hierarchical_data(self) -> None:
        """Handwriting tagging works with hierarchical block/line/word structure."""
        output = {
            "pages": [
                {
                    "data": {
                        "blocks": [
                            {
                                "type": "text",
                                "lines": [
                                    {
                                        "words": [
                                            {"text": "clear", "bbox": [0, 0, 30, 10], "confidence": 0.95},
                                            {"text": "text", "bbox": [30, 0, 60, 10], "confidence": 0.92},
                                        ],
                                    },
                                ],
                            },
                            {
                                "type": "text",
                                "lines": [
                                    {
                                        "words": [
                                            {"text": "messy", "bbox": [100, 0, 200, 10], "confidence": 0.25},
                                            {"text": "scratch", "bbox": [105, 0, 210, 10], "confidence": 0.4},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
            ],
        }
        result = tag_handwriting_regions(output)
        page = result["pages"][0]
        data = page["data"]
        # First block (high conf) -> no handwriting.
        block0_words = data["blocks"][0]["lines"][0]["words"]
        assert block0_words[0].get("handwriting") is not True
        assert block0_words[1].get("handwriting") is not True
        # Second block (low conf + irregular spacing) -> handwriting.
        block1_words = data["blocks"][1]["lines"][0]["words"]
        assert block1_words[0]["handwriting"] is True

    def test_tag_handwriting_no_confidence_field(self) -> None:
        """Words without confidence field -> not tagged as handwriting."""
        output = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "bbox": [0, 0, 30, 10]},
                    ],
                },
            ],
        }
        result = tag_handwriting_regions(output)
        assert result["has_handwriting"] is False
        assert result["pages"][0]["results"][0].get("handwriting") is not True


class TestFilterHandwritingFromScoring:
    """Separation of handwriting and text for scoring."""

    def test_filter_handwriting(self) -> None:
        """Tagged handwriting regions excluded from text scoring data."""
        page_data = {
            "results": [
                {"text": "hello", "handwriting": True, "confidence": 0.3},
                {"text": "world", "handwriting": False, "confidence": 0.95},
                {"text": "test", "handwriting": False, "confidence": 0.92},
            ],
        }
        text_data, hw_data = filter_handwriting_from_scoring(page_data)
        assert len(text_data["results"]) == 2  # non-handwriting words
        assert len(hw_data["results"]) == 1  # handwriting word
        assert text_data["results"][0]["text"] == "world"
        assert hw_data["results"][0]["text"] == "hello"

    def test_filter_no_handwriting(self) -> None:
        """No handwriting tags -> all go to text_data."""
        page_data = {
            "results": [
                {"text": "hello", "confidence": 0.95},
                {"text": "world", "confidence": 0.92},
            ],
        }
        text_data, hw_data = filter_handwriting_from_scoring(page_data)
        assert len(text_data["results"]) == 2
        assert len(hw_data["results"]) == 0

    def test_filter_hierarchical_data(self) -> None:
        """Filtering works with block/line/word hierarchy."""
        page_data = {
            "data": {
                "blocks": [
                    {
                        "type": "text",
                        "lines": [
                            {
                                "words": [
                                    {"text": "hello", "handwriting": True, "confidence": 0.3},
                                    {"text": "world", "handwriting": False, "confidence": 0.95},
                                ],
                            },
                        ],
                    },
                ],
            },
        }
        text_data, hw_data = filter_handwriting_from_scoring(page_data)
        text_block = text_data["data"]["blocks"][0]
        assert len(text_block["lines"][0]["words"]) == 1
        assert text_block["lines"][0]["words"][0]["text"] == "world"

        hw_block = hw_data["data"]["blocks"][0]
        assert len(hw_block["lines"][0]["words"]) == 1
        assert hw_block["lines"][0]["words"][0]["text"] == "hello"

    def test_filter_all_handwriting(self) -> None:
        """All words are handwriting -> text_data has no results."""
        page_data = {
            "results": [
                {"text": "hello", "handwriting": True, "confidence": 0.3},
                {"text": "world", "handwriting": True, "confidence": 0.4},
            ],
        }
        text_data, hw_data = filter_handwriting_from_scoring(page_data)
        assert len(text_data["results"]) == 0
        assert len(hw_data["results"]) == 2

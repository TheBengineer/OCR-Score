"""Novel OCR evaluation metrics — Imagination Rate (hallucination detection),
Confidence Calibration Error (CCE), and Noise Sensitivity Index (NSI).

Validates against existing CER/WER metrics to provide richer evaluation of
OCR engine quality.

Typical usage::

    from backend.evaluation.novel_metrics import (
        compute_imagination_rate,
        compute_confidence_calibration_error,
        compute_noise_sensitivity_index,
    )

    # Imagination: what fraction of OCR words are hallucinations?
    ir = compute_imagination_rate(ocr_words, ref_words)

    # Calibration: does confidence match accuracy?
    cce = compute_confidence_calibration_error(ocr_words, ref_words)

    # Noise sensitivity: how does CER degrade with DPI?
    nsi = compute_noise_sensitivity_index(engine, "doc.pdf", [72, 150, 300])
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.alignment.aligner import align_ocr_texts
from backend.evaluation.scoring import compute_cer, compute_wer

# ── Novel metrics ─────────────────────────────────────────────────────────────


def compute_imagination_rate(
    ocr_words: list[dict],
    reference_words: list[dict],
) -> float:
    """Rate of hallucinated text — words in OCR output not found in reference.

    Uses the existing word-aligner to classify OCR output words:

    - Words aligned as ``"insertions"`` are hallucinations.
    - IR = number of insertions / total OCR words.

    Args:
        ocr_words: List of word dicts from OCR output, each containing
            ``"text"`` and optionally ``"confidence"`` and ``"bbox"``.
        reference_words: List of word dicts from ground truth (same format).

    Returns:
        Imagination Rate in ``[0.0, 1.0]``:

        - ``0.0`` = no hallucination (all OCR words are grounded in reference).
        - ``1.0`` = every OCR word is hallucinated.
        - Returns ``0.0`` when *ocr_words* is empty.
    """
    if not ocr_words:
        return 0.0

    alignment = align_ocr_texts(ocr_words, reference_words)
    insertions = alignment["stats"].get("insertions", 0)
    return insertions / len(ocr_words)


def compute_confidence_calibration_error(
    ocr_words: list[dict],
    reference_words: list[dict],
) -> float:
    """Confidence Calibration Error — Brier-score-derived metric.

    Measures how well an OCR engine's confidence scores match its actual
    correctness.  For each word::

        correctness_i = 1.0  if the word was correctly recognised (match)
        correctness_i = 0.0  otherwise (substitution or insertion)

    CCE = mean((confidence_i - correctness_i)^2)

    An engine with high confidence on correct words **and** low confidence
    on wrong words will have a low CCE (well calibrated).  An engine that
    is always 90% confident regardless of correctness will have high CCE.

    Args:
        ocr_words: List of word dicts from OCR output.  Each dict *must*
            contain ``"text"`` and ``"confidence"`` (``float`` in ``[0, 1]``).
        reference_words: List of word dicts from ground truth.

    Returns:
        CCE in ``[0.0, 1.0]``:

        - ``0.0`` = perfect calibration.
        - ``1.0`` = worst possible calibration.
        - Returns ``0.0`` when *ocr_words* is empty.
    """
    if not ocr_words:
        return 0.0

    alignment = align_ocr_texts(ocr_words, reference_words)

    # Build ocr_idx → correctness mapping from the alignment pairs.
    ocr_correctness: dict[int, float] = {}
    for pair in alignment["word_pairs"]:
        ocr_idx = pair["ocr_idx"]
        if ocr_idx is not None:
            ocr_correctness[ocr_idx] = 1.0 if pair["operation"] == "match" else 0.0

    total_sq_error = 0.0
    for idx, word in enumerate(ocr_words):
        confidence = word.get("confidence", 0.0)
        correctness = ocr_correctness.get(idx, 0.0)
        total_sq_error += (confidence - correctness) ** 2

    return total_sq_error / len(ocr_words)


def compute_noise_sensitivity_index(
    engine: Callable[[str, int], list[dict]] | None = None,
    pdf_path: str | None = None,
    resolutions: list[int] | None = None,
    *,
    _test_texts: dict[int, list[dict]] | None = None,
    _test_reference: list[dict] | None = None,
) -> dict:
    """Noise Sensitivity Index — CER degradation across DPI values.

    Evaluates how OCR quality degrades as the input resolution decreases.
    Computes CER at each requested DPI and fits a linear regression to
    determine the degradation slope.

    When called without a real engine (e.g. in unit tests), pass
    ``_test_texts`` (a ``{dpi: [word_dict, ...]}`` mapping) and
    ``_test_reference`` (the reference word list at the highest quality).

    Args:
        engine: A callable ``engine(pdf_path, dpi) -> list[word_dict]``
            that runs OCR at a given resolution.
        pdf_path: Path to the PDF document to evaluate.
        resolutions: List of DPI values to test (e.g. ``[72, 150, 300]``).
        _test_texts: Internal — pre-computed OCR word lists per resolution
            for unit testing without a real engine.
        _test_reference: Internal — reference word list for testing.

    Returns:
        Dict with:

        - ``cer_at_<dpi>`` — CER at each tested resolution.
        - ``degradation_slope`` — Linear regression slope of CER vs DPI.
          A positive slope means CER increases (worsens) at lower resolution.
    """
    if _test_texts is not None and _test_reference is not None:
        ocr_texts = _test_texts
        ref_words = _test_reference
    elif engine is not None and pdf_path is not None and resolutions is not None:
        ocr_texts = {}
        for dpi in resolutions:
            ocr_texts[dpi] = engine(pdf_path, dpi)
        ref_words = engine(pdf_path, max(resolutions))
    else:
        return {
            "error": "Provide either (engine, pdf_path, resolutions) or "
            "(_test_texts, _test_reference)."
        }

    res_list = sorted(ocr_texts.keys())
    cers: list[float] = []
    ref_text = " ".join(w["text"] for w in ref_words)

    for dpi in res_list:
        ocr_words = ocr_texts[dpi]
        hyp_text = " ".join(w["text"] for w in ocr_words)
        cers.append(compute_cer(ref_text, hyp_text))

    # Linear regression of CER vs DPI.
    slope = _linregress_slope([float(r) for r in res_list], cers)

    result: dict[str, Any] = {
        "degradation_slope": slope,
    }
    for dpi, cer in zip(res_list, cers, strict=True):
        result[f"cer_at_{dpi}"] = cer

    return result


# ── Combined computation and validation ──────────────────────────────────────


def _extract_words_from_dict(data: dict) -> list[dict]:
    """Normalise *data* into a flat list of word dicts.

    Handles:
    - ``{"words": [...]}`` — direct word list.
    - ``{"pages": [{"results": [...]}, ...]}`` — run format (first page only).
    - A list of dicts accidentally wrapped in a dict.
    """
    # Direct word list.
    words = data.get("words")
    if words is not None and isinstance(words, list):
        return words

    # Run format: {"pages": [...]}
    pages = data.get("pages")
    if pages is not None and isinstance(pages, list) and pages:
        first = pages[0]
        # Try "results" key.
        results = first.get("results")
        if results is not None and isinstance(results, list):
            return results
        # Try "data" key with hierarchical blocks.
        inner = first.get("data", first)
        if isinstance(inner, dict):
            extracted: list[dict] = []
            for block in inner.get("blocks", []):
                for line in block.get("lines", []):
                    extracted.extend(line.get("words", []))
            if extracted:
                return extracted
            if "text" in inner:
                return [inner]

    return []


def compute_all_novel_metrics(
    ocr_data: dict,
    reference_data: dict,
    *,
    engine: Callable[[str, int], list[dict]] | None = None,
    pdf_path: str | None = None,
    resolutions: list[int] | None = None,
    _test_texts: dict[int, list[dict]] | None = None,
    _test_reference: list[dict] | None = None,
) -> dict:
    """Compute all novel metrics plus standard CER/WER for comparison.

    Args:
        ocr_data: OCR output data dict.  Supports ``{"words": [...]}``
            for direct word lists or ``{"pages": [{"results": [...]}]}``
            for the run format.
        reference_data: Ground truth data dict (same format as *ocr_data*).
        engine: Optional — OCR engine callable for NSI.
        pdf_path: Optional — PDF path for NSI.
        resolutions: Optional — DPI resolutions for NSI.
        _test_texts: Internal — pre-computed OCR texts per resolution for NSI.
        _test_reference: Internal — reference word list for NSI.

    Returns:
        Dict with ``"standard"`` (CER, WER) and ``"novel"``
        (imagination_rate, confidence_calibration_error,
        noise_sensitivity_index) sections.
    """
    ocr_words = _extract_words_from_dict(ocr_data)
    ref_words = _extract_words_from_dict(reference_data)

    ref_text = " ".join(w["text"] for w in ref_words)
    hyp_text = " ".join(w["text"] for w in ocr_words)

    # Standard metrics.
    cer = compute_cer(ref_text, hyp_text)
    wer = compute_wer(
        [w["text"] for w in ref_words],
        [w["text"] for w in ocr_words],
    )

    # Novel metrics.
    ir = compute_imagination_rate(ocr_words, ref_words)
    cce = compute_confidence_calibration_error(ocr_words, ref_words)

    # Noise sensitivity (optional).
    nsi: dict | None = None
    if _test_texts is not None and _test_reference is not None:
        nsi = compute_noise_sensitivity_index(
            _test_texts=_test_texts, _test_reference=_test_reference,
        )
    elif engine is not None and pdf_path is not None and resolutions is not None:
        nsi = compute_noise_sensitivity_index(engine, pdf_path, resolutions)

    return {
        "standard": {
            "cer": cer,
            "wer": wer,
        },
        "novel": {
            "imagination_rate": ir,
            "confidence_calibration_error": cce,
            "noise_sensitivity_index": nsi,
        },
    }


def validate_metrics_on_fixtures() -> dict:
    """Validation report comparing novel vs. standard metrics.

    Creates multiple synthetic OCR-vs-reference scenarios with known
    characteristics and verifies that:

    - Imagination Rate correctly identifies hallucinated words.
    - CCE correctly measures confidence calibration quality.
    - Novel metrics provide orthogonal signal not captured by CER/WER.

    Returns:
        Dict with ``"scenarios"`` (per-scenario metric breakdown),
        ``"verdicts"`` (individual pass/fail checks), and
        ``"all_pass"`` (overall boolean).
    """
    # ── Test scenario 1: perfect OCR ─────────────────────────────────────
    perfect_ocr = [
        {"text": "the", "confidence": 0.95},
        {"text": "quick", "confidence": 0.93},
        {"text": "brown", "confidence": 0.97},
        {"text": "fox", "confidence": 0.94},
    ]
    perfect_ref = [
        {"text": "the", "confidence": 1.0},
        {"text": "quick", "confidence": 1.0},
        {"text": "brown", "confidence": 1.0},
        {"text": "fox", "confidence": 1.0},
    ]

    # ── Test scenario 2: OCR with hallucinations ─────────────────────────
    hallu_ocr = [
        {"text": "the", "confidence": 0.95},
        {"text": "quick", "confidence": 0.93},
        {"text": "brown", "confidence": 0.97},
        {"text": "fox", "confidence": 0.94},
        {"text": "extra", "confidence": 0.80},
        {"text": "madeup", "confidence": 0.75},
        {"text": "nonexistent", "confidence": 0.70},
    ]
    hallu_ref = list(perfect_ref)

    # ── Test scenario 3: substitutions with overconfidence ───────────────
    # Uses words with fuzz.ratio < 60 for definite substitution classification.
    overconf_ocr = [
        {"text": "hello", "confidence": 0.95},
        {"text": "xxxxx", "confidence": 0.90},   # substitution (0% sim), high conf
        {"text": "world", "confidence": 0.97},
        {"text": "yyyyy", "confidence": 0.92},    # substitution (0% sim), high conf
    ]
    overconf_ref = [
        {"text": "hello", "confidence": 1.0},
        {"text": "right", "confidence": 1.0},
        {"text": "world", "confidence": 1.0},
        {"text": "wrong", "confidence": 1.0},
    ]

    # ── Test scenario 4: well-calibrated (low conf on errors) ────────────
    well_cal_ocr = [
        {"text": "hello", "confidence": 0.95},
        {"text": "xxxxx", "confidence": 0.30},   # substitution (0% sim), low conf
        {"text": "world", "confidence": 0.97},
        {"text": "yyyyy", "confidence": 0.25},    # substitution (0% sim), low conf
    ]
    well_cal_ref = [
        {"text": "hello", "confidence": 1.0},
        {"text": "right", "confidence": 1.0},
        {"text": "world", "confidence": 1.0},
        {"text": "wrong", "confidence": 1.0},
    ]

    scenarios = [
        ("perfect", perfect_ocr, perfect_ref),
        ("hallucinated", hallu_ocr, hallu_ref),
        ("overconfident", overconf_ocr, overconf_ref),
        ("well_calibrated", well_cal_ocr, well_cal_ref),
    ]

    results: dict[str, Any] = {}
    for name, ocr, ref in scenarios:
        ref_text = " ".join(w["text"] for w in ref)
        hyp_text = " ".join(w["text"] for w in ocr)
        cer = compute_cer(ref_text, hyp_text)
        wer = compute_wer([w["text"] for w in ref], [w["text"] for w in ocr])
        ir = compute_imagination_rate(ocr, ref)
        cce = compute_confidence_calibration_error(ocr, ref)

        results[name] = {
            "standard": {"cer": cer, "wer": wer},
            "novel": {
                "imagination_rate": ir,
                "confidence_calibration_error": cce,
            },
        }

    # ── Verdicts ─────────────────────────────────────────────────────────
    verdicts: list[dict] = []

    # Perfect scenario: IR ≈ 0, CCE ≈ 0.
    perfect = results["perfect"]
    verdicts.append({
        "check": "Perfect OCR → IR ≈ 0.0",
        "pass": perfect["novel"]["imagination_rate"] == 0.0,
        "value": perfect["novel"]["imagination_rate"],
    })
    verdicts.append({
        "check": "Perfect OCR → low CCE",
        "pass": perfect["novel"]["confidence_calibration_error"] < 0.01,
        "value": perfect["novel"]["confidence_calibration_error"],
    })

    # Hallucinated scenario: IR > 0.
    hal = results["hallucinated"]
    verdicts.append({
        "check": "Hallucinated → IR > 0.0",
        "pass": hal["novel"]["imagination_rate"] > 0.0,
        "value": hal["novel"]["imagination_rate"],
    })

    # Overconfident CCE > Well-calibrated CCE.
    oc = results["overconfident"]
    wc = results["well_calibrated"]
    verdicts.append({
        "check": "Overconfident CCE > Well-calibrated CCE",
        "pass": oc["novel"]["confidence_calibration_error"]
        > wc["novel"]["confidence_calibration_error"],
        "value": {
            "overconfident_cce": oc["novel"]["confidence_calibration_error"],
            "well_calibrated_cce": wc["novel"]["confidence_calibration_error"],
        },
    })

    # Both CER and IR increase with hallucinations (orthogonal signals).
    verdicts.append({
        "check": "Hallucinated CER > Perfect CER",
        "pass": results["hallucinated"]["standard"]["cer"]
        > results["perfect"]["standard"]["cer"],
        "value": {
            "perfect_cer": results["perfect"]["standard"]["cer"],
            "hallucinated_cer": results["hallucinated"]["standard"]["cer"],
        },
    })
    verdicts.append({
        "check": "Hallucinated IR > Perfect IR (0.0)",
        "pass": results["hallucinated"]["novel"]["imagination_rate"]
        > results["perfect"]["novel"]["imagination_rate"],
        "value": {
            "perfect_ir": results["perfect"]["novel"]["imagination_rate"],
            "hallucinated_ir": results["hallucinated"]["novel"]["imagination_rate"],
        },
    })

    all_pass = all(v["pass"] for v in verdicts)

    return {
        "scenarios": results,
        "verdicts": verdicts,
        "all_pass": all_pass,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _linregress_slope(x: list[float], y: list[float]) -> float:
    """Compute the slope of a simple linear regression ``y = m*x + b``.

    Uses the least-squares formula:

        m = Σ((x_i - x̄)(y_i - ȳ)) / Σ((x_i - x̄)²)

    Returns ``0.0`` when there are fewer than 2 data points or the
    denominator is zero.
    """
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=True))
    den = sum((xi - x_mean) ** 2 for xi in x)
    return num / den if den != 0 else 0.0

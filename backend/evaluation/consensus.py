"""Consensus entropy based auto-GT generation for OCRScore.

Automatically generates ground truth from multiple OCR engine outputs by
measuring agreement entropy (Consensus Entropy) and producing consensus text.

Uses pairwise Needleman-Wunsch alignment (via :mod:`backend.alignment.aligner`)
to align engine word sequences, then computes entropy of agreement and routes
the output based on configurable thresholds.

SIZE_OK — Full pipeline (alignment, entropy, voting, GT, validation) per
Phase 2 Task 17 spec. Every function is required by the task contract.
"""

from __future__ import annotations

import math
from collections import Counter

from backend.alignment.aligner import needleman_wunsch

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_GT_CONFIG: dict = {
    "ce_threshold_low": 0.2,
    "ce_threshold_high": 0.6,
}


# ── Word extraction helpers ──────────────────────────────────────────────────


def _extract_words_from_output(output: dict) -> list[dict]:
    """Extract a flat list of word dicts from an engine output dict.

    Handles:
    - ``{"words": [word_dict, ...]}`` — flat word list.
    - ``{"data": {"blocks": [{"lines": [{"words": [...]}]}]}}`` — hierarchical.
    - Bare ``{"text": "..."}`` — single-word fallback.

    Args:
        output: Engine output dict.

    Returns:
        List of word dicts, each with at least ``"text"``.
    """
    words: list[dict] | None = output.get("words")
    if words is not None and isinstance(words, list):
        return words

    data: dict = output.get("data", output)
    if isinstance(data, dict):
        result: list[dict] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                result.extend(line.get("words", []))
        if result:
            return result

    text: str = output.get("text", "")
    if text:
        return [{"text": text}]

    return []


def _get_engine_texts(engine_outputs: list[dict]) -> list[str]:
    """Extract full text strings from engine outputs for entropy computation.

    Args:
        engine_outputs: List of engine output dicts.

    Returns:
        List of text strings, one per engine.
    """
    texts: list[str] = []
    for output in engine_outputs:
        words = _extract_words_from_output(output)
        if words:
            texts.append(" ".join(w["text"] for w in words))
        else:
            texts.append(output.get("text", ""))
    return texts


# ── Multi-sequence word alignment ────────────────────────────────────────────


def _align_word_sequences(
    sequences: list[list[str]],
) -> list[list[str | None]]:
    """Align multiple word sequences against the first as reference.

    Uses pairwise Needleman-Wunsch to align each subsequent sequence
    against the first (reference), then projects all engine words onto
    the reference positions.

    Args:
        sequences: List of word lists, one per engine.

    Returns:
        List of aligned positions. Each position is a list of
        ``(str | None)`` values, one per engine, where ``None``
        represents a deletion (gap) at that position.
    """
    if not sequences:
        return []

    reference = sequences[0]
    n_engines = len(sequences)

    # Pairwise NW against reference (engine 0).
    alignments: list[list[tuple[int | None, int | None]]] = []
    for i in range(1, n_engines):
        path, _ = needleman_wunsch(reference, sequences[i])
        alignments.append(path)

    # Build aligned positions by projecting onto reference indices.
    aligned: list[list[str | None]] = []
    for ref_idx, ref_word in enumerate(reference):
        position: list[str | None] = [ref_word]

        for engine_idx, path in enumerate(alignments):
            engine_seq = sequences[engine_idx + 1]
            engine_word: str | None = None
            for ref_i, eng_i in path:
                if ref_i == ref_idx:
                    if eng_i is not None:
                        engine_word = engine_seq[eng_i]
                    break
            position.append(engine_word)

        aligned.append(position)

    return aligned


# ── Consensus Entropy ────────────────────────────────────────────────────────


def compute_consensus_entropy(engine_texts: list[str]) -> float:
    """Compute Consensus Entropy across multiple OCR engine texts.

    For each aligned word position, computes the probability distribution
    of text across engines, then:

        H = -Σ p(x) · log2(p(x))

    averaged across all aligned positions and normalised to [0.0, 1.0].

    Args:
        engine_texts: List of text strings, one per OCR engine.

    Returns:
        Normalised entropy in ``[0.0, 1.0]``:
        - ``0.0`` = perfect agreement (all engines produce identical text)
        - ``1.0`` = maximum disagreement

    Note:
        Returns ``0.0`` when fewer than 2 texts are provided, since
        entropy is not meaningful with a single engine.
    """
    n_engines = len(engine_texts)
    if n_engines < 2:
        return 0.0

    sequences = [text.split() for text in engine_texts]
    aligned = _align_word_sequences(sequences)

    if not aligned:
        return 0.0

    total_entropy = 0.0
    for position_texts in aligned:
        counter: Counter[str | None] = Counter(position_texts)
        position_entropy = 0.0
        for count in counter.values():
            p = count / n_engines
            if p > 0:
                position_entropy -= p * math.log2(p)
        total_entropy += position_entropy

    avg_entropy = total_entropy / len(aligned)
    max_entropy = math.log2(n_engines)

    if max_entropy == 0:
        return 0.0

    return min(avg_entropy / max_entropy, 1.0)


# ── Majority voting ──────────────────────────────────────────────────────────


def _majority_vote_words(
    words_at_position: list[dict | None],
) -> dict | None:
    """Equal-weight majority vote for words at an aligned position.

    When a tie occurs (two or more texts have the same max count), the
    function falls back to confidence-weighted voting among the tied texts.

    Args:
        words_at_position: List of word dicts (or ``None`` for gaps),
            one per engine.

    Returns:
        Winning word dict with added ``"vote_count"`` and ``"total_votes"``,
        or ``None`` if all positions are gaps.
    """
    counter: Counter[str] = Counter()
    word_by_text: dict[str, dict] = {}

    for word in words_at_position:
        if word is not None:
            text = word["text"]
            counter[text] += 1
            if text not in word_by_text:
                word_by_text[text] = word

    if not counter:
        return None

    max_count = max(counter.values())
    tied_texts = [t for t, c in counter.items() if c == max_count]

    if len(tied_texts) == 1:
        winner = dict(word_by_text[tied_texts[0]])
        winner["vote_count"] = max_count
        winner["total_votes"] = len(words_at_position)
        return winner

    return _confidence_weighted_vote(words_at_position, tied_texts)


def _confidence_weighted_vote(
    words_at_position: list[dict | None],
    candidates: list[str] | None = None,
) -> dict | None:
    """Confidence-weighted majority vote for words at an aligned position.

    Each engine's vote is weighted by word confidence. When *candidates*
    is provided, only those texts are considered. When ``None`` (default),
    all non-``None`` texts are considered.

    Args:
        words_at_position: List of word dicts (or ``None``), one per engine.
        candidates: Optional subset of texts to restrict voting to.

    Returns:
        Winning word dict with added ``"confidence"`` (average),
        ``"vote_count"``, and ``"total_votes"``, or ``None``.
    """
    confidence_sums: Counter[str] = Counter()
    confidence_n: Counter[str] = Counter()
    word_by_text: dict[str, dict] = {}

    for word in words_at_position:
        if word is not None:
            text = word["text"]
            conf = word.get("confidence", 1.0)
            confidence_sums[text] += conf
            confidence_n[text] += 1
            if text not in word_by_text:
                word_by_text[text] = word

    if not confidence_sums:
        return None

    if candidates is not None:
        for text in list(confidence_sums):
            if text not in candidates:
                del confidence_sums[text]

    if not confidence_sums:
        return None

    winner_text = max(confidence_sums, key=lambda t: confidence_sums[t])
    winner = dict(word_by_text[winner_text])
    avg_conf = confidence_sums[winner_text] / confidence_n[winner_text]
    winner["confidence"] = avg_conf
    winner["vote_count"] = confidence_n[winner_text]
    winner["total_votes"] = len(words_at_position)

    return winner


# ── Engine output alignment ──────────────────────────────────────────────────


def _align_engine_outputs(engine_outputs: list[dict]) -> dict:
    """Align multiple engine outputs at word level.

    Uses pairwise Needleman-Wunsch to align each engine's words against
    the first engine's words (reference).

    Args:
        engine_outputs: List of engine output dicts. Each should contain
            a ``"words"`` key (list of word dicts) or a ``"data"`` key
            with the canonical JSONB hierarchy.

    Returns:
        Dict with:
        - ``"aligned_words"`` — list of per-position lists, each containing
          a word dict or ``None`` per engine
        - ``"engines"`` — list of engine identifiers
        - ``"num_engines"`` — number of engines
        - ``"num_positions"`` — number of aligned positions
    """
    word_lists: list[list[dict]] = [
        _extract_words_from_output(o) for o in engine_outputs
    ]

    if not word_lists or not word_lists[0]:
        return {
            "aligned_words": [],
            "engines": [
                o.get("engine", f"engine_{i}")
                for i, o in enumerate(engine_outputs)
            ],
            "num_engines": len(engine_outputs),
            "num_positions": 0,
        }

    reference_words = word_lists[0]
    reference_texts = [w["text"] for w in reference_words]
    n_engines = len(word_lists)

    # Pairwise NW alignments against reference.
    alignments: list[list[tuple[int | None, int | None]]] = []
    engine_texts_list: list[list[str]] = []
    for i in range(1, n_engines):
        eng_texts = [w["text"] for w in word_lists[i]]
        engine_texts_list.append(eng_texts)
        path, _ = needleman_wunsch(reference_texts, eng_texts)
        alignments.append(path)

    # Build aligned positions.
    aligned_words: list[list[dict | None]] = []
    for ref_idx, ref_word in enumerate(reference_words):
        position: list[dict | None] = [ref_word]

        for engine_idx, path in enumerate(alignments):
            eng_word: dict | None = None
            for ref_i, eng_i in path:
                if ref_i == ref_idx:
                    if eng_i is not None:
                        eng_word = word_lists[engine_idx + 1][eng_i]
                    break
            position.append(eng_word)

        aligned_words.append(position)

    return {
        "aligned_words": aligned_words,
        "engines": [
            o.get("engine", f"engine_{i}")
            for i, o in enumerate(engine_outputs)
        ],
        "num_engines": n_engines,
        "num_positions": len(aligned_words),
    }


def _majority_vote(
    aligned_words: list[list[dict | None]],
) -> list[dict]:
    """Perform majority vote at each aligned position.

    Args:
        aligned_words: List of aligned positions from
            :func:`_align_engine_outputs`.

    Returns:
        List of winning word dicts, one per aligned position.
    """
    result: list[dict] = []
    for position in aligned_words:
        winner = _majority_vote_words(position)
        if winner is not None:
            result.append(winner)
    return result


# ── Confidence-weighted consensus ────────────────────────────────────────────


def compute_confidence_weighted_consensus(engine_outputs: list[dict]) -> dict:
    """Compute a confidence-weighted consensus across engine outputs.

    At each aligned position, each engine's vote is weighted by its
    per-word confidence. The function also detects positions where
    the confidence-weighted result disagrees with equal-weight voting
    (CE failsafe flag).

    Args:
        engine_outputs: List of engine output dicts, each containing
            words with ``"text"`` and optionally ``"confidence"``.

    Returns:
        Dict with:
        - ``"aligned_positions"`` — raw aligned word data
        - ``"consensus_words"`` — winning word at each position
        - ``"agreement_flags"`` — bool per position where weighted
          and unweighted consensus disagree
        - ``"num_positions"`` — total aligned positions
    """
    alignment = _align_engine_outputs(engine_outputs)
    aligned_words: list[list[dict | None]] = alignment["aligned_words"]

    consensus_words: list[dict] = []
    agreement_flags: list[bool] = []

    for position_words in aligned_words:
        equal_weight = _majority_vote_words(position_words)
        weighted = _confidence_weighted_vote(position_words)

        equal_text: str | None = (
            equal_weight.get("text") if equal_weight is not None else None
        )
        weighted_text: str | None = (
            weighted.get("text") if weighted is not None else None
        )
        flags = equal_text != weighted_text

        # Prefer the weighted result; fall back to equal weight.
        if weighted is not None:
            consensus_words.append(weighted)
        elif equal_weight is not None:
            consensus_words.append(equal_weight)

        agreement_flags.append(flags)

    return {
        "aligned_positions": aligned_words,
        "consensus_words": consensus_words,
        "agreement_flags": agreement_flags,
        "num_positions": len(aligned_words),
    }


# ── JSONB block builder ──────────────────────────────────────────────────────


def _build_blocks_from_words(words: list[dict]) -> list[dict]:
    """Build partial JSONB block structure from a flat list of word dicts.

    Produces a single text block with one line holding all words, using
    the same canonical schema as ``PageResultData``.

    Args:
        words: List of word dicts with at least ``"text"``.

    Returns:
        List of block dicts matching the PageResult JSONB schema.
    """
    if not words:
        return []

    return [
        {
            "type": "text",
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "confidence": 1.0,
            "order": 0,
            "lines": [
                {
                    "text": " ".join(w["text"] for w in words),
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "confidence": 1.0,
                    "order": 0,
                    "words": [
                        {
                            "text": w["text"],
                            "bbox": w.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                            "confidence": w.get("confidence", 1.0),
                            "order": i,
                            "chars": [],
                        }
                        for i, w in enumerate(words)
                    ],
                }
            ],
        }
    ]


# ── Ground Truth Builder ─────────────────────────────────────────────────────


def build_ground_truth(
    engine_outputs: list[dict],
    config: dict | None = None,
) -> dict:
    """Build ground truth from multiple OCR engine outputs.

    The function:
    1. Aligns engine outputs at word level (NW).
    2. Computes Consensus Entropy across engine texts.
    3. Routes based on entropy level:
       - **Low entropy** (``< 0.2``): Auto-accept consensus as GT.
       - **Medium entropy** (``0.2 - 0.6``): Auto-accept but flag for review.
       - **High entropy** (``> 0.6``): Reject — return empty GT for human review.
    4. Runs CE failsafe: flags positions where confidence-weighted
       consensus disagrees with equal-weight consensus.

    Args:
        engine_outputs: List of engine output dicts (minimum 2 for
            meaningful consensus).
        config: Optional config dict. Default keys:
            - ``ce_threshold_low`` (``float``, default ``0.2``)
            - ``ce_threshold_high`` (``float``, default ``0.6``)

    Returns:
        Dict with:
        - ``source`` — ``"auto_consensus"`` or ``None``
        - ``source_config`` — config used to build this GT
        - ``consensus_entropy`` — computed CE value
        - ``pages`` — list of page result dicts (empty when rejected)
        - ``needs_review`` — whether human review is recommended
        - ``warnings`` — list of warning messages
    """
    cfg = {**_DEFAULT_GT_CONFIG, **(config or {})}
    ce_low = cfg["ce_threshold_low"]
    ce_high = cfg["ce_threshold_high"]

    n_engines = len(engine_outputs)
    warnings: list[str] = []

    def _engine_ids() -> list[str]:
        return [
            o.get("engine", f"engine_{i}")
            for i, o in enumerate(engine_outputs)
        ]

    def _base_result() -> dict:
        return {
            "source_config": {
                "engines_used": _engine_ids(),
                "ce_threshold_low": ce_low,
                "ce_threshold_high": ce_high,
            },
        }

    # ── Edge: no engines ────────────────────────────────────────────────
    if n_engines == 0:
        return {
            **_base_result(),
            "source": None,
            "consensus_entropy": 0.0,
            "pages": [],
            "needs_review": False,
            "warnings": ["No engine outputs provided."],
        }

    # ── Edge: single engine ─────────────────────────────────────────────
    if n_engines == 1:
        words = _extract_words_from_output(engine_outputs[0])
        pages = [{"blocks": _build_blocks_from_words(words), "tables": []}]
        return {
            **_base_result(),
            "source": "auto_consensus",
            "consensus_entropy": 0.0,
            "pages": pages,
            "needs_review": True,
            "warnings": [
                "Single engine — output used as GT with low confidence."
            ],
        }

    # ── Multi-engine: build consensus ───────────────────────────────────
    engine_texts = _get_engine_texts(engine_outputs)
    ce = compute_consensus_entropy(engine_texts)

    weighted = compute_confidence_weighted_consensus(engine_outputs)
    consensus_words = weighted.get("consensus_words", [])
    agreement_flags: list[bool] = weighted.get("agreement_flags", [])

    # CE failsafe.
    if any(agreement_flags):
        warnings.append(
            "CE failsafe triggered: confidence-weighted consensus disagrees "
            "with equal-weight consensus at one or more positions."
        )

    # ── Route by CE level ───────────────────────────────────────────────
    if ce >= ce_high:
        return {
            **_base_result(),
            "source": None,
            "consensus_entropy": ce,
            "pages": [],
            "needs_review": True,
            "warnings": warnings + [
                f"High CE ({ce:.3f}) — unable to build consensus GT. "
                "Engine outputs returned for human review."
            ],
        }

    pages = [{"blocks": _build_blocks_from_words(consensus_words), "tables": []}]

    if ce < ce_low:
        return {
            **_base_result(),
            "source": "auto_consensus",
            "consensus_entropy": ce,
            "pages": pages,
            "needs_review": bool(any(agreement_flags)),
            "warnings": warnings,
        }

    # Medium entropy (ce_low <= ce < ce_high).
    return {
        **_base_result(),
        "source": "auto_consensus",
        "consensus_entropy": ce,
        "pages": pages,
        "needs_review": True,
        "warnings": warnings + [
            f"Moderate CE ({ce:.3f}) — output flagged for human review."
        ],
    }


# ── Validation ───────────────────────────────────────────────────────────────


def _validate_candidate_gt(gt_data: dict) -> bool:
    """Validate a candidate ground truth dict against consensus rules.

    Rules:
    - ``source`` must be ``"auto_consensus"`` or ``None``.
    - ``consensus_entropy`` must be in ``[0.0, 1.0]``.
    - ``pages`` must be a list where each item has ``"blocks"`` and
      ``"tables"`` keys.
    - ``needs_review`` must be a ``bool``.
    - ``warnings`` must be a ``list``.

    Args:
        gt_data: Ground truth dict to validate.

    Returns:
        ``True`` when all validation checks pass.
    """
    source = gt_data.get("source")
    if source is not None and source != "auto_consensus":
        return False

    ce = gt_data.get("consensus_entropy", -1.0)
    if not isinstance(ce, (int | float)):
        return False
    if ce < 0.0 or ce > 1.0:
        return False

    pages = gt_data.get("pages")
    if not isinstance(pages, list):
        return False
    for page in pages:
        if not isinstance(page, dict):
            return False
        if "blocks" not in page or "tables" not in page:
            return False

    if not isinstance(gt_data.get("needs_review"), bool):
        return False

    return isinstance(gt_data.get("warnings"), list)

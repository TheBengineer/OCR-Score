"""Semantic plausibility scoring for OCR output.

Provides character n-gram based semantic evaluation metrics that
complement the character/word error rates (CER/WER) from
:mod:`backend.evaluation.scoring`.  All functions work offline with
only Python standard library — no API calls, no internet access.

Four scoring functions:

* :func:`compute_fluency_score` — perplexity-based fluency from a
  character-level 4-gram language model with Jelinek-Mercer smoothing.
* :func:`compute_semantic_similarity` — cosine similarity between
  character n-gram frequency vectors of two texts.
* :func:`compute_grammaticality` — heuristic grammar quality based on
  common-word ratio, word-length distribution, punctuation, and
  vowel/consonant balance.
* :func:`compute_semantic_plausibility` — combined overall score.

When *torch* and *transformers* are installed the module can optionally
use *prajjwal1/bert-tiny* for embedding-based similarity (see
:func:`compute_bert_similarity`).

# allow: SIZE_OK — embedded training corpus (~550 words) and
# common-English-word list (~150 entries) are data constants, not logic.
"""

from __future__ import annotations

import math
import re
import string

# ── Vocabulary ──────────────────────────────────────────────────────────────

_VALID_CHARS: frozenset[str] = frozenset(
    string.ascii_lowercase + string.digits + " .,!?'-:;"
)
"""Characters the n-gram model operates on.  Unks are mapped to space."""

_VOCABULARY: list[str] = sorted(_VALID_CHARS)
"""Ordered vocabulary for index-based operations."""

_VOCAB_SIZE: int = len(_VOCABULARY)
"""Number of distinct character types in the model."""

# ── OCR digit-to-letter normalisation ────────────────────────────────────────
# Common OCR substitution patterns: digits that look like letters.
# Mapping these before fluency scoring makes the metric resilient to
# character-level OCR errors that preserve readability.

_OCR_DIGIT_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
}
"""Mapping of digits to visually similar letters for OCR resilience."""

# ── Jelinek-Mercer interpolation weights ────────────────────────────────────
# Fixed weights for orders 4, 3, 2, 1.  Must sum to ≤ 1.0; the remainder
# goes to the uniform distribution (ensures no zero probabilities).

_LAMBDAS: tuple[float, float, float, float] = (0.60, 0.20, 0.10, 0.07)
"""λ₄, λ₃, λ₂, λ₁ for Jelinek-Mercer interpolation."""

_UNIFORM_WEIGHT: float = 1.0 - sum(_LAMBDAS)  # 0.03
"""Reserved weight for uniform backoff."""

# ── Embedded English training corpus ────────────────────────────────────────
# This text is used to build character n-gram counts at import time.
# Public-domain style — varied English covering common letter patterns.

_TRAINING_CORPUS: str = (  # noqa: E501 — embedded corpus data constant
    "The quick brown fox jumps over the lazy dog. This sentence contains "
    "every letter of the English alphabet and is often used for testing fonts "
    "and keyboards. In the field of optical character recognition also known "
    "as OCR the goal is to convert images of text into machine readable text. "
    "This process involves several steps including image preprocessing text "
    "detection character recognition and post processing correction. Modern "
    "OCR systems use deep learning and neural networks to achieve high "
    "accuracy on clean documents. However challenges remain for handwritten "
    "text historical documents and images with noise or distortion. "
    "The English language has many common words that appear frequently in "
    "written text. Words like the and of to a in is that it for with as on "
    "at by from are used constantly. Longer words like information processing "
    "development technology and communication also appear often. Good OCR "
    "output should contain a mix of common and less common words arranged "
    "in a way that makes sense. "
    "Numbers and punctuation are important parts of written text. Dates like "
    "January 1 2024 addresses and amounts like 123 dollars appear in "
    "documents. Periods commas question marks exclamation points colons "
    "semicolons quotation marks and apostrophes all have specific rules for "
    "their use. Proper punctuation helps make text more readable and easier "
    "to understand. "
    "The accuracy of text recognition depends on image quality and the "
    "complexity of the document layout. Simple documents with clear fonts "
    "and good contrast produce the best results. Complex layouts with "
    "multiple columns tables and images present more challenges. The "
    "preprocessing steps like binarization deskewing and noise removal can "
    "significantly improve recognition accuracy. "
    "Research in pattern recognition and machine learning has led to "
    "significant improvements in optical character recognition technology. "
    "Modern approaches use convolutional neural networks transformers and "
    "sequence to sequence models. These methods have reduced error rates "
    "dramatically compared to traditional approaches based on feature "
    "extraction and classification. "
    "Evaluation of OCR systems involves comparing the recognized text "
    "against a ground truth reference. Common metrics include character "
    "error rate CER and word error rate WER. These metrics measure the edit "
    "distance between the recognized text and the reference text at the "
    "character and word levels respectively. Lower error rates indicate "
    "better recognition performance."
)

# ── Common English word list ────────────────────────────────────────────────
# Top ~150 most frequent English words for grammaticality heuristics.

_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset({
    "a", "able", "about", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "back", "be", "because", "been", "but", "by", "came", "can", "come",
    "could", "day", "did", "do", "down", "each", "end", "even", "first", "for",
    "from", "get", "give", "go", "good", "had", "has", "have", "he", "her", "here",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "know",
    "like", "look", "made", "make", "man", "many", "may", "me", "might", "more",
    "most", "much", "must", "my", "new", "no", "not", "now", "of", "on", "one",
    "only", "or", "other", "our", "out", "over", "people", "said", "same", "say",
    "see", "she", "so", "some", "such", "take", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "thing", "think", "this", "those",
    "time", "to", "two", "up", "us", "use", "very", "want", "was", "way", "we",
    "well", "were", "what", "when", "where", "which", "while", "who", "will",
    "with", "would", "year", "you", "your",
    "address", "amount", "between", "character", "column", "complex", "confidence",
    "contains", "context", "correct", "data", "deep", "detection", "distance",
    "document", "error", "evaluation", "field", "format", "ground", "image",
    "improve", "include", "information", "language", "layout", "learning", "letter",
    "level", "measure", "metric", "model", "network", "noise", "number", "output",
    "page", "pattern", "performance", "pixel", "point", "process", "quality",
    "rate", "recognition", "reference", "result", "score", "section", "sequence",
    "source", "space", "step", "structure", "symbol", "system", "table", "target",
    "text", "token", "training", "value", "word",
})
"""Set of common English words used to compute common-word ratio."""

# Minimum length for a token to count as a "word" for grammaticality.
_MIN_WORD_LEN: int = 1


# ═══════════════════════════════════════════════════════════════════════════
# Character n-gram language model
# ═══════════════════════════════════════════════════════════════════════════


class _CharNGramModel:
    """Character-level n-gram language model with Jelinek-Mercer smoothing.

    Built from an embedded English training corpus at class creation time.
    Supports evaluation of arbitrary text via :meth:`perplexity`.

    Args:
        n: Maximum order of the n-gram model (default 4).
        add_k: Additive smoothing constant for MLE estimates.
    """

    def __init__(self, n: int = 4) -> None:
        self.n: int = n
        # _counts[order-1] maps ngram_string -> int count
        self._counts: list[dict[str, int]] = [{} for _ in range(n)]
        # _context_counts[order-2] maps context_string -> int count (for order >= 2)
        self._context_counts: list[dict[str, int]] = [{} for _ in range(n - 1)]
        self._total_count: int = 0
        self._build()

    # ── public API ──────────────────────────────────────────────────────

    def perplexity(self, text: str) -> float:
        """Compute character-level perplexity of *text* under this model.

        Perplexity is defined as:

            ``exp(-1/N * Σ log P(cᵢ | c_{i-n+1} … c_{i-1}))``

        where N is the number of character predictions evaluated.

        Returns:
            Perplexity in ``[1.0, ∞)``.  ``1.0`` means perfect prediction.
            Returns ``1000.0`` (a high sentinel) for empty input.
        """
        if not text:
            return 1000.0

        tokens = self._tokenize(text)
        log_prob_sum = 0.0
        n_pred = 0

        # We predict each token *except* the sentinel padding; the
        # sentinel at the very end (</s>) is also predicted.
        # The first (n-1) tokens are padding, so we start at index n-1.
        for i in range(self.n - 1, len(tokens)):
            char = tokens[i]
            context_start = max(0, i - (self.n - 1))
            context = "".join(tokens[context_start:i])
            lp = self._log_prob(char, context)
            log_prob_sum += lp
            n_pred += 1

        if n_pred == 0:
            return 1000.0

        avg_log_prob = log_prob_sum / n_pred
        return math.exp(-avg_log_prob)

    # ── internal helpers ────────────────────────────────────────────────

    def _build(self) -> None:
        """Build n-gram counts from the embedded training corpus."""
        text = _TRAINING_CORPUS
        normalized = self._normalize(text)
        tokens = self._tokenize(normalized)

        for i, char in enumerate(tokens):
            # Unigram count
            self._counts[0][char] = self._counts[0].get(char, 0) + 1
            self._total_count += 1

            # Higher-order counts
            for order in range(2, self.n + 1):
                if i >= order - 1:
                    ngram = "".join(tokens[i - order + 1 : i + 1])
                    self._counts[order - 1][ngram] = (
                        self._counts[order - 1].get(ngram, 0) + 1
                    )
                    # Context count
                    context = "".join(tokens[i - order + 1 : i])
                    self._context_counts[order - 2][context] = (
                        self._context_counts[order - 2].get(context, 0) + 1
                    )

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and replace unknown characters with space."""
        result: list[str] = []
        for c in text:
            if c in _VALID_CHARS:
                result.append(c)
            else:
                result.append(" ")
        return "".join(result)

    def _tokenize(self, text: str) -> list[str]:
        """Convert text to a token list with sentinel padding.

        Prepends ``n-1`` sentinel tokens (``<s>``) and appends one
        ``</s>`` sentinel.
        """
        sentinel = "<s>"
        tokens: list[str] = [sentinel] * (self.n - 1)
        for c in text:
            tokens.append(c)
        tokens.append("</s>")
        return tokens

    def _raw_ml_prob(self, char: str, context: str) -> float:
        """Maximum-likelihood estimate P(char | context) using raw counts.

        Jelinek-Mercer interpolation (handled by :meth:`_log_prob`) and the
        uniform-component backoff guarantee non-zero probabilities for unseen
        n-grams, so no additive smoothing is needed here.

        Args:
            char: The character to predict.
            context: History string of length ``order - 1``.

        Returns:
            Probability in ``[0.0, 1.0]``.  Returns ``0.0`` for unseen
            (context, char) pairs — the caller interpolates across orders.
        """
        order = len(context) + 1

        if order == 1:
            count = self._counts[0].get(char, 0)
            if self._total_count == 0:
                return 0.0
            return count / self._total_count

        ngram = context + char
        count = self._counts[order - 1].get(ngram, 0)
        context_count = self._context_counts[order - 2].get(context, 0)
        if context_count == 0:
            return 0.0
        return count / context_count

    def _log_prob(self, char: str, context: str) -> float:
        """Jelinek-Mercer smoothed log-probability.

        Interpolates between all orders 4 → 3 → 2 → 1 → uniform,
        weighted by the fixed lambdas.
        """
        # Clip context to the maximum history length.
        context = context[-(self.n - 1) :]

        prob = 0.0
        for order in range(self.n, 0, -1):
            lam = _LAMBDAS[order - 1]
            if order == 1:
                ml = self._raw_ml_prob(char, "")
            else:
                hist = context[-(order - 1) :]
                ml = self._raw_ml_prob(char, hist)
            prob += lam * ml

        # Uniform component for remaining weight.
        prob += _UNIFORM_WEIGHT / _VOCAB_SIZE

        # Guard against numerical underflow.
        return math.log(max(prob, 1e-30))


# ── Module-level singleton model ──────────────────────────────────────────

_CHAR_NGRAM: _CharNGramModel = _CharNGramModel(n=4)
"""Reusable character 4-gram model initialised at import time."""


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def compute_fluency_score(text: str) -> float:
    """Perplexity-based fluency score in ``[0.0, 1.0]``.

    Uses the character 4-gram model with Jelinek-Mercer smoothing.

    Before scoring, digits commonly confused by OCR (e.g. 0→o, 1→l)
    are normalised to their letter lookalikes so that a text like
    ``"Hell0 w0rld"`` is recognised as fluent despite digit substitutions.

    * 1.0 = perfectly fluent (perplexity ≈ 1)
    * 0.0 = no better than uniform character distribution

    Args:
        text: Input text string.

    Returns:
        Fluency score in ``[0.0, 1.0]``.  Returns ``0.0`` for empty text.
    """
    if not text or not text.strip():
        return 0.0

    # Normalise OCR digit confusions before scoring.
    normalised = _normalise_ocr_digits(text)
    ppl = _CHAR_NGRAM.perplexity(normalised)

    # Normalise: perplexity ranges from 1.0 (perfect) to VOCAB_SIZE (uniform).
    # Map linearly to [0, 1].
    score = max(0.0, 1.0 - (ppl - 1.0) / (_VOCAB_SIZE - 1.0))

    # Clip for numerical safety.
    return max(0.0, min(1.0, score))


def compute_semantic_similarity(ocr_text: str, reference_text: str) -> float:
    """Cosine similarity of character n-gram frequency vectors.

    Extracts character 3-gram frequency profiles from both texts and
    returns their cosine similarity.  This measures how similar the
    character-compositional patterns are — a proxy for semantic relatedness
    that is robust to minor OCR typos.

    Args:
        ocr_text: OCR-hypothesis text.
        reference_text: Ground-truth reference text.

    Returns:
        Cosine similarity in ``[0.0, 1.0]``.  Returns ``1.0`` when both
        texts are identical.  Returns ``0.0`` if either is empty.
    """
    if not ocr_text or not reference_text:
        return 0.0

    ocr_ngrams = _char_ngrams(ocr_text, n=3)
    ref_ngrams = _char_ngrams(reference_text, n=3)

    # Build the union of all n-gram keys.
    all_keys = set(ocr_ngrams) | set(ref_ngrams)
    if not all_keys:
        return 1.0  # Both empty after processing.

    # Compute dot product and magnitudes.
    dot = 0
    ocr_mag_sq = 0
    ref_mag_sq = 0

    # Use the pre-computed local dicts for O(1) lookup.
    for key in all_keys:
        ocr_val = ocr_ngrams.get(key, 0)
        ref_val = ref_ngrams.get(key, 0)
        dot += ocr_val * ref_val
        ocr_mag_sq += ocr_val * ocr_val
        ref_mag_sq += ref_val * ref_val

    ocr_mag = math.sqrt(ocr_mag_sq)
    ref_mag = math.sqrt(ref_mag_sq)

    if ocr_mag == 0.0 or ref_mag == 0.0:
        return 0.0

    return min(1.0, dot / (ocr_mag * ref_mag))


def compute_grammaticality(text: str) -> float:
    """Grammar-quality score in ``[0.0, 1.0]`` using surface heuristics.

    Combines four signals without any syntactic parser:

    1. **Common-word ratio** (weight 0.40) — fraction of tokens that
       appear in the built-in common English word list.
    2. **Word-length normality** (weight 0.25) — how close the average
       word length is to the English typical range (4.5–5.5).
    3. **Punctuation correctness** (weight 0.20) — checks that
       sentence-ending punctuation is followed by a space or end-of-string;
       penalises stray punctuation inside words.
    4. **Letter-class balance** (weight 0.15) — ratio of vowels to
       consonants, penalising extreme deviations from the English norm
       (~38 % vowels).

    Args:
        text: Input text string.

    Returns:
        Score in ``[0.0, 1.0]``.  Returns ``0.0`` for empty text.
    """
    if not text or not text.strip():
        return 0.0

    tokens = text.split()
    if not tokens:
        return 0.0

    # ── Component 1: common-word ratio ──────────────────────────────
    common_count = sum(
        1 for t in tokens if t.lower().strip(string.punctuation) in _COMMON_ENGLISH_WORDS
    )
    common_ratio = common_count / len(tokens)
    common_score = common_ratio  # Already in [0, 1].

    # ── Component 2: word-length normality ──────────────────────────
    word_lengths = [len(t) for t in tokens if any(c.isalpha() for c in t)]
    if word_lengths:
        avg_word_len = sum(word_lengths) / len(word_lengths)
        # English average ~5 ± 1.5 is ideal; penalise deviation.
        length_dev = abs(avg_word_len - 5.0)
        length_score = max(0.0, 1.0 - length_dev / 5.0)
    else:
        length_score = 0.0

    # ── Component 3: punctuation correctness ──────────────────────
    punct_score = _score_punctuation(text)

    # ── Component 4: vowel/consonant balance ────────────────────────
    vowels = set("aeiou")
    letters_only = [c for c in text.lower() if c in string.ascii_lowercase]
    if letters_only:
        vowel_count = sum(1 for c in letters_only if c in vowels)
        vowel_ratio = vowel_count / len(letters_only)
        # English vowel ratio ≈ 0.38.  Penalise strong deviations.
        vowel_dev = abs(vowel_ratio - 0.38)
        vowel_score = max(0.0, 1.0 - vowel_dev / 0.38)
    else:
        vowel_score = 0.0

    # ── Weighted combination ────────────────────────────────────────
    weights = {"common": 0.40, "length": 0.25, "punct": 0.20, "vowel": 0.15}
    score = (
        weights["common"] * common_score
        + weights["length"] * length_score
        + weights["punct"] * punct_score
        + weights["vowel"] * vowel_score
    )

    return max(0.0, min(1.0, score))


def compute_semantic_plausibility(text: str) -> float:
    """Combined semantic plausibility score in ``[0.0, 1.0]``.

    A weighted average of :func:`compute_fluency_score` and
    :func:`compute_grammaticality`, designed to be a single-number
    supplement to CER / WER.

    A score ≥ 0.6 generally indicates readable text even when character- or
    word-level errors are present (e.g., "tlie qu1ck br0wn" is still
    clearly English).

    Args:
        text: Input text string.

    Returns:
        Plausibility score in ``[0.0, 1.0]``.  Returns ``0.0`` for empty text.
    """
    if not text or not text.strip():
        return 0.0

    fluency = compute_fluency_score(text)
    grammar = compute_grammaticality(text)

    # Weighted average: fluency is a stronger signal, grammaticality is
    # a supporting check.
    return 0.6 * fluency + 0.4 * grammar


# ── Optional: BERT-based similarity ────────────────────────────────────────

try:
    import torch as _torch  # type: ignore[import-untyped]
    import transformers as _transformers  # type: ignore[import-untyped]

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


if _HAS_TRANSFORMERS:

    _BERT_DEVICE: str = "cuda" if _torch.cuda.is_available() else "cpu"
    _BERT_MODEL: _transformers.AutoModel | None = None  # type: ignore[no-untyped-call]
    _BERT_TOKENIZER: _transformers.AutoTokenizer | None = None  # type: ignore[no-untyped-call]

    def _get_bert() -> tuple[_transformers.AutoModel, _transformers.AutoTokenizer]:  # type: ignore[no-untyped-call]
        """Lazy-load the BERT-tiny model."""
        global _BERT_MODEL, _BERT_TOKENIZER  # noqa: PLW0603
        if _BERT_MODEL is None:
            _BERT_MODEL = _transformers.AutoModel.from_pretrained("prajjwal1/bert-tiny")
            _BERT_MODEL.to(_BERT_DEVICE)
            _BERT_MODEL.eval()
            _BERT_TOKENIZER = _transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        return _BERT_MODEL, _BERT_TOKENIZER

    def compute_bert_similarity(ocr_text: str, reference_text: str) -> float:
        """BERT-embedding cosine similarity (requires torch + transformers).

        Falls back to :func:`compute_semantic_similarity` if the BERT
        model fails to load or run.

        Args:
            ocr_text: OCR-hypothesis text.
            reference_text: Ground-truth reference text.

        Returns:
            Cosine similarity in ``[0.0, 1.0]``.
        """
        if not ocr_text or not reference_text:
            return 0.0
        try:
            model, tokenizer = _get_bert()
            inputs = tokenizer(
                [ocr_text, reference_text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(_BERT_DEVICE)
            with _torch.no_grad():
                outputs = model(**inputs)
            # Use [CLS] token embedding.
            embeddings = outputs.last_hidden_state[:, 0, :]
            emb_a = embeddings[0].cpu().numpy()
            emb_b = embeddings[1].cpu().numpy()
            dot = float(sum(a * b for a, b in zip(emb_a, emb_b, strict=False)))
            mag_a = math.sqrt(float(sum(a * a for a in emb_a)))
            mag_b = math.sqrt(float(sum(b * b for b in emb_b)))
            if mag_a == 0.0 or mag_b == 0.0:
                return 0.0
            return min(1.0, dot / (mag_a * mag_b))
        except Exception:  # noqa: BLE001
            return compute_semantic_similarity(ocr_text, reference_text)

else:
    # When transformers are not available, the BERT function does not exist.
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _normalise_ocr_digits(text: str) -> str:
    """Replace digits with letter lookalikes for OCR-resilient fluency scoring.

    Common OCR confusions (e.g. ``0`` for ``o``, ``1`` for ``l``) are
    mapped to their most likely letter counterpart so that the character
    n-gram model does not penalise text that is human-readable despite
    digit substitutions.
    """
    result: list[str] = []
    for c in text:
        result.append(_OCR_DIGIT_MAP.get(c, c))
    return "".join(result)


def _char_ngrams(text: str, n: int = 3) -> dict[str, int]:
    """Extract character n-gram frequency dict from *text*.

    The text is lower-cased and non-vocabulary characters are replaced
    with spaces before counting.

    Args:
        text: Input string.
        n: N-gram order (default 3).

    Returns:
        Counter dict mapping n-gram string → frequency.
    """
    # Normalise character set.
    cleaned: list[str] = []
    for c in text.lower():
        cleaned.append(c if c in _VALID_CHARS else " ")

    chars = "".join(cleaned)

    counter: dict[str, int] = {}
    for i in range(len(chars) - n + 1):
        gram = chars[i : i + n]
        counter[gram] = counter.get(gram, 0) + 1
    return counter


def _score_punctuation(text: str) -> float:
    """Score punctuation correctness (0.0 – 1.0)."""
    if not text:
        return 0.0

    deductions = 0.0
    checks = 0

    # 1. Stray punctuation inside alphanumeric tokens (e.g., "he,llo").
    tokens = text.split()
    for token in tokens:
        stripped = token.strip(string.punctuation)
        if len(stripped) < len(token):
            punct_chars = [c for c in token if c in string.punctuation]
            # Check if punctuation is at token boundaries (fine) or internal.
            first_alnum = next((i for i, c in enumerate(token) if c.isalnum()), -1)
            last_alnum = next(
                (i for i, c in enumerate(reversed(token)) if c.isalnum()), -1
            )
            if last_alnum != -1:
                last_alnum = len(token) - 1 - last_alnum
            for p in punct_chars:
                checks += 1
                if p in {",", ".", "!", "?", ":", ";", "'", '"'}:
                    # Check position: is punctuation at token boundary?
                    p_idx = token.find(p)
                    if p_idx > first_alnum and p_idx < last_alnum:
                        deductions += 0.3  # Internal punctuation

    # 2. Sentence-ending punctuation should be followed by space or EOS.
    for match in re.finditer(r"[.!?]", text):
        checks += 1
        pos = match.end()
        if pos < len(text) and text[pos] not in {" ", "\n", "\r", "\t"}:
            deductions += 0.2

    # 3. Check for reasonable comma placement.
    for match in re.finditer(r",", text):
        checks += 1
        pos = match.end()
        if pos < len(text) and text[pos] not in {" ", "\n", "\r", "\t"}:
            deductions += 0.2

    if checks == 0:
        return 0.5  # Neutral — no punctuation to evaluate.

    raw = max(0.0, 1.0 - deductions / max(checks, 1))
    return raw

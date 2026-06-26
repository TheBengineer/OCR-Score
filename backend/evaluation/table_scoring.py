"""GriTS (Grid Table Similarity) scoring for OCR table evaluation.

Provides GriTS_Top (topology), GriTS_Con (content), and GriTS_Loc (location)
metrics following Smock et al. 2022, plus full table-structure evaluation.

Typical usage::

    from backend.evaluation.table_scoring import (
        compute_table_structure_metrics,
        grits_con,
        grits_loc,
        grits_top,
    )

    # Score a single table pair
    top = grits_top(pred_cells, gt_cells)
    con = grits_con(pred_cells, gt_cells)
    loc = grits_loc(pred_cells, gt_cells)

    # Score a full set of tables across a page
    metrics = compute_table_structure_metrics(pred_tables, gt_tables)
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

# ── Public API ────────────────────────────────────────────────────────────────


def grits_top(
    pred_cells: list[dict],
    gt_cells: list[dict],
    *,
    threshold: float = 0.5,
) -> float:
    """GriTS_Top: cell topology (row/col structure) similarity.

    Uses separable 2D-LCS on the cell matrices: computes row subsequence
    similarity and column subsequence similarity, then averages them.

    Args:
        pred_cells: Predicted table cells (NormalizedSchema.TableCell format).
        gt_cells: Ground-truth table cells.
        threshold: Row/column similarity threshold for LCS matching (default 0.5).

    Returns:
        Score in [0.0, 1.0].
    """
    if not pred_cells and not gt_cells:
        return 1.0

    _, _, row_pairs, col_pairs = _compute_aligned_cells(
        pred_cells, gt_cells, threshold,
    )
    matrix_a = _build_cell_matrix(pred_cells)
    matrix_b = _build_cell_matrix(gt_cells)

    n_rows_a, n_rows_b = len(matrix_a), len(matrix_b)
    row_sim = (
        2.0 * len(row_pairs) / (n_rows_a + n_rows_b)
        if (n_rows_a + n_rows_b) > 0
        else 1.0
    )

    cols_a = _transpose(matrix_a)
    cols_b = _transpose(matrix_b)
    n_cols_a, n_cols_b = len(cols_a), len(cols_b)
    col_sim = (
        2.0 * len(col_pairs) / (n_cols_a + n_cols_b)
        if (n_cols_a + n_cols_b) > 0
        else 1.0
    )

    return (row_sim + col_sim) / 2.0


def grits_con(
    pred_cells: list[dict],
    gt_cells: list[dict],
    *,
    threshold: float = 0.5,
) -> float:
    """GriTS_Con: cell text content similarity.

    After cell alignment from GriTS_Top, averages text similarity of matched
    cells using Levenshtein distance. Unmatched cells contribute 0.0.

    Args:
        pred_cells: Predicted table cells.
        gt_cells: Ground-truth table cells.
        threshold: Row/column similarity threshold for LCS matching.

    Returns:
        Score in [0.0, 1.0].
    """
    if not pred_cells and not gt_cells:
        return 1.0

    matrix_a, matrix_b, row_pairs, col_pairs = _compute_aligned_cells(
        pred_cells, gt_cells, threshold,
    )

    return _compute_cell_value_mean(
        matrix_a, matrix_b, row_pairs, col_pairs,
        lambda a, b: _cell_text_similarity(
            a.get("text", ""), b.get("text", ""),
        ),
    )


def grits_loc(
    pred_cells: list[dict],
    gt_cells: list[dict],
    *,
    threshold: float = 0.5,
) -> float:
    """GriTS_Loc: cell bounding box similarity.

    After cell alignment from GriTS_Top, averages IoU of matched cells'
    bounding boxes. Unmatched cells contribute 0.0.

    Args:
        pred_cells: Predicted table cells.
        gt_cells: Ground-truth table cells.
        threshold: Row/column similarity threshold for LCS matching.

    Returns:
        Score in [0.0, 1.0].
    """
    if not pred_cells and not gt_cells:
        return 1.0

    matrix_a, matrix_b, row_pairs, col_pairs = _compute_aligned_cells(
        pred_cells, gt_cells, threshold,
    )

    return _compute_cell_value_mean(
        matrix_a, matrix_b, row_pairs, col_pairs,
        lambda a, b: _iou(a["bbox"], b["bbox"]),
    )


def compute_table_structure_metrics(
    pred_tables: list[dict],
    gt_tables: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> dict:
    """Full set of table evaluation metrics across multiple tables.

    Matches predicted tables to ground-truth tables by bounding-box IoU,
    then computes GriTS scores for matched pairs and detection statistics
    for unmatched tables.

    Args:
        pred_tables: List of predicted table dicts (NormalizedSchema.Table
            format with ``bbox``, ``cells`` keys).
        gt_tables: List of ground-truth table dicts.
        iou_threshold: IoU threshold for table-level matching (default 0.5).

    Returns:
        Dict with keys: ``grits_top``, ``grits_con``, ``grits_loc``,
        ``structure_precision``, ``structure_recall``, ``cell_accuracy``,
        ``table_detection_precision``, ``table_detection_recall``.
    """
    if not pred_tables and not gt_tables:
        return {
            "grits_top": 1.0,
            "grits_con": 1.0,
            "grits_loc": 1.0,
            "structure_precision": 1.0,
            "structure_recall": 1.0,
            "cell_accuracy": 1.0,
            "table_detection_precision": 1.0,
            "table_detection_recall": 1.0,
        }

    # Match predicted tables to ground-truth tables via greedy IoU matching.
    pred_to_gt: dict[int, int] = {}
    gt_matched: set[int] = set()

    for i, pred in enumerate(pred_tables):
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_tables):
            if j in gt_matched:
                continue
            iou_val = _iou(pred["bbox"], gt["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou > iou_threshold and best_j >= 0:
            pred_to_gt[i] = best_j
            gt_matched.add(best_j)

    tp = len(pred_to_gt)
    fp = len(pred_tables) - tp
    fn = len(gt_tables) - tp

    table_precision = _safe_div(tp, tp + fp)
    table_recall = _safe_div(tp, tp + fn)

    # Compute GriTS for each matched table pair.
    grits_top_scores: list[float] = []
    grits_con_scores: list[float] = []
    grits_loc_scores: list[float] = []
    struct_precisions: list[float] = []
    struct_recalls: list[float] = []

    for pred_idx, gt_idx in pred_to_gt.items():
        pred_cells = pred_tables[pred_idx].get("cells", [])
        gt_cells = gt_tables[gt_idx].get("cells", [])

        if not pred_cells and not gt_cells:
            grits_top_scores.append(1.0)
            grits_con_scores.append(1.0)
            grits_loc_scores.append(1.0)
            struct_precisions.append(1.0)
            struct_recalls.append(1.0)
        else:
            sp, sr = _compute_structure_precision_recall(pred_cells, gt_cells)
            grits_top_scores.append(grits_top(pred_cells, gt_cells))
            grits_con_scores.append(grits_con(pred_cells, gt_cells))
            grits_loc_scores.append(grits_loc(pred_cells, gt_cells))
            struct_precisions.append(sp)
            struct_recalls.append(sr)

    avg_grits_top = _safe_mean(grits_top_scores)
    avg_grits_con = _safe_mean(grits_con_scores)
    avg_grits_loc = _safe_mean(grits_loc_scores)
    avg_struct_precision = _safe_mean(struct_precisions)
    avg_struct_recall = _safe_mean(struct_recalls)

    # Cell accuracy: F1 of structure precision/recall.
    p, r = avg_struct_precision, avg_struct_recall
    cell_accuracy = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        "grits_top": avg_grits_top,
        "grits_con": avg_grits_con,
        "grits_loc": avg_grits_loc,
        "structure_precision": avg_struct_precision,
        "structure_recall": avg_struct_recall,
        "cell_accuracy": cell_accuracy,
        "table_detection_precision": table_precision,
        "table_detection_recall": table_recall,
    }


# ── Cell matrix helpers ───────────────────────────────────────────────────────


def _build_cell_matrix(cells: list[dict]) -> list[list[dict | None]]:
    """Convert a flat cell list into a 2D grid.

    The matrix is sized by the maximum (row + row_span) and (col + col_span)
    across all cells.  Each cell is placed at its ``(row, col)`` start position;
    positions covered by ``row_span`` / ``col_span`` remain ``None``.

    Args:
        cells: List of cell dicts with ``row``, ``col``, ``row_span``,
            ``col_span`` keys.

    Returns:
        2D list where ``matrix[row][col]`` is the cell dict or ``None``.
    """
    if not cells:
        return []

    n_rows = max(c["row"] + c.get("row_span", 1) for c in cells)
    n_cols = max(c["col"] + c.get("col_span", 1) for c in cells)

    if n_rows == 0 or n_cols == 0:
        return []

    matrix: list[list[dict | None]] = [[None] * n_cols for _ in range(n_rows)]
    for cell in cells:
        r = cell["row"]
        c = cell["col"]
        matrix[r][c] = cell

    return matrix


def _transpose(matrix: list[list]) -> list[list]:
    """Transpose a 2D list (rows become columns)."""
    if not matrix or not matrix[0]:
        return []
    return [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]


# ── Cell matching and similarity ──────────────────────────────────────────────


def _cell_matrix_match(val_a: dict | None, val_b: dict | None) -> bool:
    """Two cell-matrix entries match structurally.

    Returns ``True`` when both are ``None`` (spanned positions) or both are
    actual cell dicts.
    """
    return (val_a is None) == (val_b is None)


def _row_similarity(row_a: list, row_b: list) -> float:
    """1D-LCS similarity between two row vectors from a cell matrix.

    Uses DP to compute the longest common subsequence where two entries
    match according to :func:`_cell_matrix_match`.

    Returns:
        ``2 * LCS_len / (len(row_a) + len(row_b))``.
    """
    n, m = len(row_a), len(row_b)

    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if _cell_matrix_match(row_a[i - 1], row_b[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[n][m]
    return 2.0 * lcs_len / (n + m)


def _compute_lcs_similarity_and_pairs(
    seq_a: list,
    seq_b: list,
    similarity_fn,
    *,
    threshold: float = 0.5,
) -> tuple[float, list[tuple[int, int]]]:
    """Compute LCS between two sequences using a pairwise similarity function.

    Two items ``seq_a[i]`` and ``seq_b[j]`` count as a match when
    ``similarity_fn(seq_a[i], seq_b[j]) > threshold``.

    Args:
        seq_a: First sequence.
        seq_b: Second sequence.
        similarity_fn: Callable ``(item_a, item_b) -> float``.
        threshold: Similarity threshold for a match.

    Returns:
        ``(similarity, matched_pairs)`` where similarity is ``2 * LCS_len /
        (len_a + len_b)`` and ``matched_pairs`` is a list of ``(i, j)``
        index pairs in LCS order.
    """
    n, m = len(seq_a), len(seq_b)

    if n == 0 and m == 0:
        return 1.0, []
    if n == 0 or m == 0:
        return 0.0, []

    # Precompute similarity matrix.
    sim = [[similarity_fn(seq_a[i], seq_b[j]) for j in range(m)] for i in range(n)]

    # DP for LCS.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sim[i - 1][j - 1] > threshold:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[n][m]
    similarity = 2.0 * lcs_len / (n + m)

    # Backtrace to recover matched index pairs.
    matched_pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if sim[i - 1][j - 1] > threshold and dp[i][j] == dp[i - 1][j - 1] + 1:
            matched_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    matched_pairs.reverse()
    return similarity, matched_pairs


# ── Alignment computation ─────────────────────────────────────────────────────


def _compute_aligned_cells(
    pred_cells: list[dict],
    gt_cells: list[dict],
    threshold: float = 0.5,
) -> tuple[list[list[dict | None]], list[list[dict | None]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Compute GriTS cell alignment between two cell lists.

    Builds cell matrices for both tables and runs separable 2D-LCS along
    rows and columns.

    Returns:
        ``(pred_matrix, gt_matrix, row_pairs, col_pairs)`` where
        ``row_pairs`` are ``(pred_row, gt_row)`` matched index pairs and
        ``col_pairs`` are ``(pred_col, gt_col)`` matched index pairs.
    """
    matrix_a = _build_cell_matrix(pred_cells)
    matrix_b = _build_cell_matrix(gt_cells)

    if not matrix_a or not matrix_b:
        return matrix_a, matrix_b, [], []

    _, row_pairs = _compute_lcs_similarity_and_pairs(
        matrix_a, matrix_b, _row_similarity, threshold=threshold,
    )

    cols_a = _transpose(matrix_a)
    cols_b = _transpose(matrix_b)

    if not cols_a or not cols_b:
        return matrix_a, matrix_b, row_pairs, []

    _, col_pairs = _compute_lcs_similarity_and_pairs(
        cols_a, cols_b, _row_similarity, threshold=threshold,
    )

    return matrix_a, matrix_b, row_pairs, col_pairs


# ── Structure precision / recall ──────────────────────────────────────────────


def _compute_structure_precision_recall(
    pred_cells: list[dict],
    gt_cells: list[dict],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Cell-level structure precision and recall for a table pair.

    A predicted cell is "correct" if its row and column are both in the
    aligned (LCS-matched) region.  Precision = correct / total predicted
    cells.  Recall = correct gt cells / total gt cells.

    Returns:
        ``(precision, recall)`` each in ``[0.0, 1.0]``.
    """
    matrix_a, matrix_b, row_pairs, col_pairs = _compute_aligned_cells(
        pred_cells, gt_cells, threshold,
    )

    matched_rows_a = {r_a for r_a, _ in row_pairs}
    matched_cols_a = {c_a for c_a, _ in col_pairs}

    predicted_correct = 0
    predicted_total = 0
    for r in range(len(matrix_a)):
        for c in range(len(matrix_a[0])):
            if matrix_a[r][c] is not None:
                predicted_total += 1
                if r in matched_rows_a and c in matched_cols_a:
                    predicted_correct += 1

    matched_rows_b = {r_b for _, r_b in row_pairs}
    matched_cols_b = {c_b for _, c_b in col_pairs}

    gt_correct = 0
    gt_total = 0
    for r in range(len(matrix_b)):
        for c in range(len(matrix_b[0])):
            if matrix_b[r][c] is not None:
                gt_total += 1
                if r in matched_rows_b and c in matched_cols_b:
                    gt_correct += 1

    # If both tables have no cells, the structure is trivially correct.
    if predicted_total == 0 and gt_total == 0:
        return 1.0, 1.0

    precision = _safe_div(predicted_correct, predicted_total)
    recall = _safe_div(gt_correct, gt_total)
    return precision, recall


# ── Content / location aggregation ────────────────────────────────────────────


def _compute_cell_value_mean(
    matrix_a: list[list[dict | None]],
    matrix_b: list[list[dict | None]],
    row_pairs: list[tuple[int, int]],
    col_pairs: list[tuple[int, int]],
    compare_fn,
) -> float:
    """Average a cell-level comparison function across aligned cells.

    For each position in the aligned grid (Cartesian product of matched row
    pairs and matched col pairs):
    - Both cells present: ``compare_fn(cell_a, cell_b)`` is added.
    - Both ``None`` (spanned on both sides): ``1.0`` is added.
    - One present, one ``None``: ``0.0`` is added (unmatched).

    Returns:
        Mean value across all aligned positions.
    """
    if not row_pairs or not col_pairs:
        return 0.0

    total = 0.0
    count = 0

    for r_a, r_b in row_pairs:
        for c_a, c_b in col_pairs:
            count += 1
            cell_a = matrix_a[r_a][c_a]
            cell_b = matrix_b[r_b][c_b]

            if cell_a is not None and cell_b is not None:
                total += compare_fn(cell_a, cell_b)
            elif cell_a is None and cell_b is None:
                total += 1.0
            # else: one is None, other is not → unmatched → 0.0

    return total / count if count > 0 else 1.0


# ── Primitive metric helpers ──────────────────────────────────────────────────


def _cell_text_similarity(text_a: str, text_b: str) -> float:
    """Levenshtein-based text similarity for two cell content strings.

    Returns ``1.0 - normalized_edit_distance`` where the distance is
    normalised by the longer string length.

    Args:
        text_a: First cell text.
        text_b: Second cell text.

    Returns:
        Similarity in ``[0.0, 1.0]``.
    """
    if text_a == text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    distance = Levenshtein.distance(text_a, text_b)
    max_len = max(len(text_a), len(text_b))
    return 1.0 - distance / max_len


def _iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Intersection over Union for two bounding boxes.

    Args:
        bbox_a: ``[x0, y0, x1, y1]`` coordinates.
        bbox_b: ``[x0, y0, x1, y1]`` coordinates.

    Returns:
        IoU in ``[0.0, 1.0]``.
    """
    x_left = max(bbox_a[0], bbox_b[0])
    y_top = max(bbox_a[1], bbox_b[1])
    x_right = min(bbox_a[2], bbox_b[2])
    y_bottom = min(bbox_a[3], bbox_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ── Internal arithmetic helpers ───────────────────────────────────────────────


def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divide, returning 0.0 when the denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def _safe_mean(values: list[float]) -> float:
    """Safely mean a list, returning 1.0 when the list is empty."""
    return sum(values) / len(values) if values else 1.0

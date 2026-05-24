# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (list[list[int]] | np.ndarray): Matched indices of shape (K, 2), where K is the number of
            matches.
        unmatched_a (np.ndarray): Unmatched indices from the first set, with shape (L,).
        unmatched_b (np.ndarray): Unmatched indices from the second set, with shape (M,).

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).

    Examples:
        Compute IoU distance between two sets of tracks
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks, where each track contains embedding features.
        detections (list[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks and M
            is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost


def color_histogram_distance(
    atracks: list, btracks: list, prev_img: np.ndarray | None, curr_img: np.ndarray | None
) -> np.ndarray:
    """Compute HSV histogram cost between previous-track ROIs and current detection ROIs."""
    cost_matrix = np.ones((len(atracks), len(btracks)), dtype=np.float32)
    if cost_matrix.size == 0 or prev_img is None or curr_img is None:
        return cost_matrix

    for i, atrack in enumerate(atracks):
        a_roi = _get_roi_from_bbox(getattr(atrack, "prev_xyxy", None), prev_img)
        if a_roi is None:
            continue
        for j, btrack in enumerate(btracks):
            b_roi = _get_roi_from_bbox(btrack.xyxy, curr_img)
            if b_roi is None:
                continue
            cost_matrix[i, j] = 1.0 - _color_histogram_similarity(a_roi, b_roi)

    return cost_matrix


def keep_top_n(cost_matrix: np.ndarray, n: int = 3) -> np.ndarray:
    """Keep the top-n smallest costs per detection column and mask the rest to 1.0."""
    num_rows, num_cols = cost_matrix.shape
    filtered = np.full((num_rows, num_cols), 1.0, dtype=np.float32)
    if num_rows == 0 or num_cols == 0:
        return filtered

    for col in range(num_cols):
        col_data = cost_matrix[:, col]
        finite_indices = np.isfinite(col_data)
        if not np.any(finite_indices):
            continue
        finite_rows = np.flatnonzero(finite_indices)
        top_n = np.argsort(col_data[finite_indices])[: min(n, len(finite_rows))]
        filtered[finite_rows[top_n], col] = col_data[finite_indices][top_n]

    return filtered


def fuse_iou_with_color(iou_cost_matrix: np.ndarray, color_cost_matrix: np.ndarray) -> np.ndarray:
    """Fuse IoU and color costs via multiplicative similarity."""
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix
    iou_sim = 1.0 - iou_cost_matrix
    color_sim = 1.0 - color_cost_matrix
    return 1.0 - (iou_sim * color_sim)


def _color_histogram_similarity(a_roi: np.ndarray | None, b_roi: np.ndarray | None) -> float:
    """Return HSV histogram similarity in [0, 1] using Bhattacharyya distance."""
    if a_roi is None or b_roi is None:
        return 0.0

    a_hsv = cv2.cvtColor(a_roi, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_roi, cv2.COLOR_BGR2HSV)
    a_hist = cv2.calcHist([a_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    b_hist = cv2.calcHist([b_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(a_hist, a_hist)
    cv2.normalize(b_hist, b_hist)
    return float(1.0 - cv2.compareHist(a_hist, b_hist, cv2.HISTCMP_BHATTACHARYYA))


def _get_roi_from_bbox(bbox: np.ndarray | list | tuple | None, img: np.ndarray | None) -> np.ndarray | None:
    """Clip bbox to the image boundary and return the ROI."""
    if bbox is None or img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, np.asarray(bbox).reshape(-1)[:4])
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

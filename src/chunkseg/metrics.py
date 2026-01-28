"""Metric computation for time-chunked segmentation evaluation.

Binary classification metrics are implemented with numpy (no torch dependency).
Segmentation-specific metrics use segeval and nltk.
"""

import decimal
import math
from collections import defaultdict

import numpy as np
import segeval
from nltk.metrics.segmentation import ghd


def _binary_confusion(pred: np.ndarray, ref: np.ndarray):
    """Compute TP, FP, FN, TN from binary 0/1 arrays."""
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    tp = int(np.sum(pred_b & ref_b))
    fp = int(np.sum(pred_b & ~ref_b))
    fn = int(np.sum(~pred_b & ref_b))
    tn = int(np.sum(~pred_b & ~ref_b))
    return tp, fp, fn, tn


def binary_precision(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, fp, _, _ = _binary_confusion(pred, ref)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def binary_recall(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, _, fn, _ = _binary_confusion(pred, ref)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def binary_accuracy(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, fp, fn, tn = _binary_confusion(pred, ref)
    total = tp + fp + fn + tn
    return (tp + tn) / total if total > 0 else 0.0


def binary_specificity(pred: np.ndarray, ref: np.ndarray) -> float:
    _, fp, _, tn = _binary_confusion(pred, ref)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


_masses = segeval.convert_nltk_to_masses


def _segeval_safe(metric_fn, pred_str: str, ref_str: str) -> float:
    """Call a segeval metric with fallback handling for edge cases."""
    pred_masses = _masses(pred_str)
    ref_masses = _masses(ref_str)
    try:
        return float(metric_fn(pred_masses, ref_masses))
    except (decimal.InvalidOperation, ZeroDivisionError):
        return 1.0 if pred_str == ref_str else 0.0
    except Exception:
        if pred_str == ref_str:
            return 1.0
        raise


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute all metrics for a single prediction/reference pair.

    Args:
        pred: Binary 0/1 prediction array.
        ref: Binary 0/1 reference array.

    Returns:
        Dict mapping metric names to float values.
    """
    assert len(pred) == len(ref), (
        f"Length mismatch: pred({len(pred)}) vs ref({len(ref)})"
    )

    pred_str = "".join(str(int(v)) for v in pred)
    ref_str = "".join(str(int(v)) for v in ref)

    metrics = {}

    metrics["precision"] = binary_precision(pred, ref)
    metrics["recall"] = binary_recall(pred, ref)
    metrics["accuracy"] = binary_accuracy(pred, ref)
    metrics["specificity"] = binary_specificity(pred, ref)
    metrics["pk"] = _segeval_safe(segeval.pk, pred_str, ref_str)
    metrics["window_diff"] = _segeval_safe(segeval.window_diff, pred_str, ref_str)
    metrics["boundary_similarity"] = _segeval_safe(
        segeval.boundary_similarity, pred_str, ref_str
    )

    metrics["ghd"] = ghd(pred_str, ref_str)
    metrics["num_segments"] = int(pred_str.count("1"))
    metrics["reference/num_segments"] = int(ref_str.count("1"))

    return metrics


def bootstrap_confidence_interval(
    data,
    statistic_func=np.mean,
    alpha: float = 0.05,
    num_iterations: int = 100,
):
    """Compute bootstrap confidence interval for a statistic.

    Returns:
        Tuple of (std, lower_bound, upper_bound, margin_of_error).
    """
    data = np.asarray(data)
    bootstrapped = np.empty(num_iterations)
    for i in range(num_iterations):
        resampled = np.random.choice(data, size=len(data), replace=True)
        bootstrapped[i] = statistic_func(resampled)

    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    std = float(np.std(bootstrapped))
    lower = float(np.percentile(bootstrapped, lower_percentile))
    upper = float(np.percentile(bootstrapped, upper_percentile))
    margin = (upper - lower) / 2

    return std, lower, upper, margin


def f1_score_with_std_dev(
    p: float, r: float, sigma_p: float, sigma_r: float
) -> tuple[float, float]:
    """Compute F1 from precision/recall with uncertainty propagation.

    Returns:
        Tuple of (f1, sigma_f1).
    """
    if (p + r) == 0:
        return 0.0, 0.0
    f1 = 2 * (p * r) / (p + r)
    df_dp = 2 * r / (p + r) - 2 * p * r / (p + r) ** 2
    df_dr = 2 * p / (p + r) - 2 * p * r / (p + r) ** 2
    sigma_f = math.sqrt((df_dp ** 2 * sigma_p ** 2) + (df_dr ** 2 * sigma_r ** 2))
    return f1, sigma_f


def aggregate_metrics(
    all_metrics: list[dict],
    num_iterations: int = 100,
) -> dict:
    """Aggregate per-sample metrics into means with bootstrap confidence intervals.

    Args:
        all_metrics: List of dicts (one per sample) from :func:`compute_metrics`.
        num_iterations: Bootstrap iterations.

    Returns:
        Dict mapping metric name to ``{mean, std, ci_lower, ci_upper}``.
        Also includes a derived ``f1`` entry computed from aggregated P/R.
    """
    if not all_metrics:
        return {}

    collected: dict[str, list[float]] = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            collected[k].append(v)

    result = {}
    stds: dict[str, float] = {}

    data_stats = {"num_segments", "reference/num_segments"}

    for name, values in collected.items():
        arr = np.array(values)
        mean = float(np.mean(arr))
        if name in data_stats:
            std = float(np.std(arr))
            result[name] = {
                "mean": mean,
                "std": std,
                "ci_lower": mean - std,
                "ci_upper": mean + std,
            }
            stds[name] = std
        else:
            std, ci_lower, ci_upper, _ = bootstrap_confidence_interval(
                arr, np.mean, num_iterations=num_iterations,
            )
            result[name] = {
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
            stds[name] = std

    # Derived F1 from aggregated precision/recall
    if "precision" in result and "recall" in result:
        p = result["precision"]["mean"]
        r = result["recall"]["mean"]
        sigma_p = stds["precision"]
        sigma_r = stds["recall"]
        f1, sigma_f = f1_score_with_std_dev(p, r, sigma_p, sigma_r)
        result["f1"] = {
            "mean": f1,
            "std": sigma_f,
            "ci_lower": f1 - sigma_f,
            "ci_upper": f1 + sigma_f,
        }

    return result

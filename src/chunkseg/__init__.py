"""chunkseg — Time-chunked segmentation evaluation."""

__version__ = "0.3.1"

from .evaluate import evaluate, evaluate_batch
from .display import print_results

__all__ = ["evaluate", "evaluate_batch", "print_results", "compute_title_scores"]


def __getattr__(name):
    if name == "compute_title_scores":
        from .titles import compute_title_scores
        return compute_title_scores
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

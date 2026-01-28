"""chunkseg — Time-chunked segmentation evaluation."""

__version__ = "0.1.0"

from .evaluate import evaluate, evaluate_batch
from .display import print_results

__all__ = ["evaluate", "evaluate_batch", "print_results"]

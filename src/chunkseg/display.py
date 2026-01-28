"""Result formatting and printing for segmentation evaluation."""

from __future__ import annotations


_DATA_STATS = {"num_segments", "reference/num_segments"}

_round = lambda x: round(x, 2)


def print_results(results: dict) -> None:
    """Print aggregated evaluation results to stdout.

    Args:
        results: Dict as returned by :func:`chunkseg.evaluate.evaluate_batch`
            (metric name -> ``{mean, std, ci_lower, ci_upper}``).
    """
    if not results:
        print("No results to display.")
        return

    for name, vals in results.items():
        mean = vals["mean"]
        std = vals["std"]

        if name in _DATA_STATS:
            print(f"{name}: {_round(mean)} ± {_round(std)}")
        else:
            print(f"{name}: {_round(mean * 100)} ± {_round(std * 100)}")

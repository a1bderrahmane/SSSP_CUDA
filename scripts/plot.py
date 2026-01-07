#!/usr/bin/env python3

"""Plot grouped bar chart of solver timings from test_results.csv."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "test_results.csv"


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"Results file not found: {path}")
    df = pd.read_csv(path)
    required = {
        "test_name",
        "cpu_time_ns",
        "gpu_time_ns",
        "hybrid_time_ns",
        "cpu_time_spread",
        "gpu_time_spread",
        "hybrid_time_spread",
        "cpu_status",
        "gpu_status",
        "hybrid_status",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns in CSV: {', '.join(sorted(missing))}")
    return df


def main():
    df = load_results(RESULTS_CSV)
    if df.empty:
        sys.exit("No data to plot.")

    tests = df["test_name"].tolist()
    # Convert nanoseconds to milliseconds for a more readable y-axis.
    ns_to_ms = 1e-6
    cpu_ms = df["cpu_time_ns"] * ns_to_ms
    gpu_ms = df["gpu_time_ns"] * ns_to_ms
    hybrid_ms = df["hybrid_time_ns"] * ns_to_ms

    cpu_times = cpu_ms.tolist()
    gpu_times = gpu_ms.tolist()
    hybrid_times = hybrid_ms.tolist()

    # Convert percentage spreads into absolute error (ms) for error bars.
    cpu_errs = (cpu_ms * (df["cpu_time_spread"] / 100.0)).fillna(np.nan).tolist()
    gpu_errs = (gpu_ms * (df["gpu_time_spread"] / 100.0)).fillna(np.nan).tolist()
    hybrid_errs = (hybrid_ms * (df["hybrid_time_spread"] / 100.0)).fillna(
        np.nan
    ).tolist()

    x = range(len(tests))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        [i - width for i in x],
        cpu_times,
        width,
        label="CPU",
        yerr=cpu_errs,
        capsize=4,
    )
    ax.bar(x, gpu_times, width, label="GPU", yerr=gpu_errs, capsize=4)
    ax.bar(
        [i + width for i in x],
        hybrid_times,
        width,
        label="HYBRID",
        yerr=hybrid_errs,
        capsize=4,
    )

    ax.set_ylabel("Time (ms)")
    ax.set_title("Solver performance per dataset")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tests, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    output_path = REPO_ROOT / "test_results.png"
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()

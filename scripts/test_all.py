#!/usr/bin/env python3

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
SOLUTIONS_DIR = DATASETS_DIR / "solutions"
EXECUTABLE = REPO_ROOT / "exec" / "sssp_solver"
RESULTS_CSV = REPO_ROOT / "test_results.csv"
SOURCE_NODE = 0
SEED_VALUE = 0
PERF_CMD_BASE = [
    "perf",
    "stat",
    "-r",
    "10",
    "--no-big-num",
    "-e",
    "duration_time",
    "-x",
    ",",
]


def normalize_value(value: str, unit: str) -> float:
    try:
        v = float(value)
    except ValueError:
        return float("nan")
    unit = unit.lower()
    if unit.startswith("nsec"):
        return v / 1e9
    if unit.startswith("usec"):
        return v / 1e6
    if unit.startswith("msec"):
        return v / 1e3
    return v


def parse_perf(stderr: str) -> dict:
    metrics = {}
    for line in stderr.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        value, _, event, spread, *_rest = parts
        try:
            val = float(value.strip())
        except ValueError:
            val = float("nan")
        try:
            sp = float(str(spread).strip().rstrip("%"))
        except ValueError:
            sp = float("nan")
        metrics[event.strip()] = {"value": val, "spread": sp}
    return metrics


def parse_distances(path: Path) -> dict:
    distances = {}
    if not path.exists():
        return distances
    pattern_vertex = re.compile(r"vertex\s+(\d+)")
    pattern_value = re.compile(r":\s*([-\d]+)")
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("node,"):
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                distances[int(parts[0])] = int(parts[1])
                continue
            m_node = pattern_vertex.search(stripped)
            m_val = pattern_value.search(stripped)
            if m_node and m_val:
                distances[int(m_node.group(1))] = int(m_val.group(1))
    return distances


def compare_distances(solution: Path, output: Path) -> tuple[bool, str]:
    sol = parse_distances(solution)
    out = parse_distances(output)
    if sol == out:
        return True, ""
    missing = [n for n in sol if n not in out]
    mismatched = [
        f"{n}: expected {sol[n]}, got {out.get(n, 'NA')}"
        for n in sol
        if n in out and sol[n] != out[n]
    ]
    msg_parts = []
    if missing:
        msg_parts.append(f"missing nodes: {missing}")
    if mismatched:
        msg_parts.append("mismatched: " + "; ".join(mismatched[:10]))
        if len(mismatched) > 10:
            msg_parts.append(f"... {len(mismatched) - 10} more mismatches")
    return False, " | ".join(msg_parts) if msg_parts else "distance sets differ"


def ensure_paths():
    if not shutil.which("perf"):
        sys.exit("perf not found in PATH. Please install perf to collect timing.")
    if not EXECUTABLE.exists():
        sys.exit(f"Executable not found: {EXECUTABLE} (build with 'make').")
    if not DATASETS_DIR.is_dir():
        sys.exit(f"Datasets directory not found: {DATASETS_DIR}")
    if not SOLUTIONS_DIR.is_dir():
        sys.exit(f"Solutions directory not found: {SOLUTIONS_DIR}")


def main():
    ensure_paths()
    results = []
    solvers = ["CPU", "GPU", "HYBRID"]
    dataset_files = sorted(DATASETS_DIR.glob("*.txt"))
    if not dataset_files:
        sys.exit(f"No datasets (*.txt) found in {DATASETS_DIR}")

    logs_dir = REPO_ROOT / "tmp_test_logs"
    logs_dir.mkdir(exist_ok=True)

    for graph_file in dataset_files:
        dataset_name = graph_file.name
        solution_file = SOLUTIONS_DIR / dataset_name
        if not solution_file.is_file():
            print(f"[SKIP] {dataset_name}: missing solution file {solution_file}")
            results.append(
                {
                    "test_name": dataset_name,
                    "cpu_time_ns": float("nan"),
                    "cpu_time_spread": float("nan"),
                    "cpu_status": "SKIP",
                    "gpu_time_ns": float("nan"),
                    "gpu_time_spread": float("nan"),
                    "gpu_status": "SKIP",
                    "hybrid_time_ns": float("nan"),
                    "hybrid_time_spread": float("nan"),
                    "hybrid_status": "SKIP",
                    "status": "SKIP",
                }
            )
            continue

        print(f"[RUN ] {dataset_name}")
        row = {
            "test_name": dataset_name,
            "cpu_time_ns": float("nan"),
            "cpu_time_spread": float("nan"),
            "cpu_status": "FAIL",
            "gpu_time_ns": float("nan"),
            "gpu_time_spread": float("nan"),
            "gpu_status": "FAIL",
            "hybrid_time_ns": float("nan"),
            "hybrid_time_spread": float("nan"),
            "hybrid_status": "FAIL",
        }

        for solver in solvers:
            output_file = logs_dir / f"{dataset_name}.{solver.lower()}.out"
            log_file = logs_dir / f"{dataset_name}.{solver.lower()}.log"
            perf_output = logs_dir / f"{dataset_name}.{solver.lower()}.perf"

            cmd = PERF_CMD_BASE + [
                "--",
                str(EXECUTABLE),
                "-i",
                str(graph_file),
                "-v",
                solver,
                "-n",
                str(SOURCE_NODE),
                "--seed",
                str(SEED_VALUE),
                "-o",
                str(output_file),
                "-l",
                str(log_file),
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            perf_output.write_text(proc.stderr or "", encoding="utf-8")
            metrics = parse_perf(proc.stderr or "")
            duration = metrics.get("duration_time", {})
            time_ns = duration.get("value", float("nan"))
            spread = duration.get("spread", float("nan"))

            status_key = f"{solver.lower()}_status"
            time_key = f"{solver.lower()}_time_ns"
            spread_key = f"{solver.lower()}_time_spread"
            row[time_key] = time_ns
            row[spread_key] = spread

            if proc.returncode != 0:
                print(f"       {solver} solver failed (see {log_file})")
                row[status_key] = "FAIL"
                continue

            ok, detail = compare_distances(solution_file, output_file)
            row[status_key] = "PASS" if ok else "FAIL"
            if row[status_key] == "FAIL":
                print(f"[FAIL] {dataset_name} ({solver}) {detail}")
            else:
                print(f"[PASS] {dataset_name} ({solver})")

        all_statuses = [row["cpu_status"], row["gpu_status"], row["hybrid_status"]]
        if all(s == "PASS" for s in all_statuses):
            row["status"] = "PASS"
        elif all(s == "SKIP" for s in all_statuses):
            row["status"] = "SKIP"
        else:
            row["status"] = "FAIL" if any(s == "FAIL" for s in all_statuses) else "PASS"

        results.append(row)

    df = pd.DataFrame(
        results,
        columns=[
            "test_name",
            "cpu_time_ns",
            "cpu_time_spread",
            "cpu_status",
            "gpu_time_ns",
            "gpu_time_spread",
            "gpu_status",
            "hybrid_time_ns",
            "hybrid_time_spread",
            "hybrid_status",
            "status",
        ],
    )
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

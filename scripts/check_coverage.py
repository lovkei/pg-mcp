#!/usr/bin/env python3
"""Per-module coverage gate for pg-mcp.

Runs pytest with coverage and enforces stricter thresholds on security- and
resilience-critical modules than the global 80% floor.

Usage:
    uv run python scripts/check_coverage.py [-- tests args]

Exit codes:
    0: all module gates pass
    1: one or more gates fail, or pytest itself failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Module prefix -> minimum coverage percentage (branch coverage enabled).
GATES: dict[str, float] = {
    "src/pg_mcp/services/sql_validator.py": 95.0,
    "src/pg_mcp/services/": 85.0,
    "src/pg_mcp/resilience/": 90.0,
    "src/pg_mcp/config/": 80.0,
    "src/pg_mcp/models/": 80.0,
}

DEFAULT_OVERALL = 80.0


def parse_total(line: str) -> tuple[float, float] | None:
    """Parse a coverage report row into (cover percent, missing).

    Args:
        line: One line from ``coverage report`` output.

    Returns:
        Tuple of (percent, missing statements) or None if the line does not
        carry a numeric total.
    """
    parts = line.split()
    for i, part in enumerate(parts):
        if part.endswith("%"):
            try:
                pct = float(part.rstrip("%"))
            except ValueError:
                return None
            missing = 0.0
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                missing = float(parts[i + 1])
            return pct, missing
    return None


def main() -> int:
    """Run coverage and enforce per-module gates.

    Returns:
        Process exit code: 0 when all gates pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest (after --)",
    )
    args = parser.parse_args()

    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [
        "uv",
        "run",
        "pytest",
        "-q",
        "--cov=src/pg_mcp",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        "--cov-branch",
        *extra,
    ]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(":: pytest failed; coverage gate not evaluated", file=sys.stderr)
        return 1

    report = subprocess.run(
        ["uv", "run", "coverage", "report", "--include=src/pg_mcp/*"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if report.returncode != 0:
        print(report.stdout, file=sys.stderr)
        return 1

    lines = report.stdout.splitlines()
    print(report.stdout)

    # Overall gate: last "TOTAL" line
    overall_ok = True
    for line in lines:
        if line.startswith("TOTAL"):
            parsed = parse_total(line)
            if parsed is None:
                continue
            pct, _ = parsed
            if pct < DEFAULT_OVERALL:
                print(
                    f"FAIL overall: {pct:.1f}% < {DEFAULT_OVERALL}%",
                    file=sys.stderr,
                )
                overall_ok = False
            else:
                print(f"OK overall: {pct:.1f}% >= {DEFAULT_OVERALL}%")
            break

    # Per-module gates
    module_ok = True
    current_module = ""
    for line in lines:
        path = line.split()[0] if line.split() else ""
        matched = ""
        for gate in GATES:
            if path.startswith(gate) and len(gate) > len(matched):
                matched = gate
        if matched:
            current_module = matched
            parsed = parse_total(line)
            if parsed is None:
                continue
            pct, _ = parsed
            threshold = GATES[matched]
            if pct < threshold:
                print(
                    f"FAIL {path}: {pct:.1f}% < {threshold} (gate: {matched})",
                    file=sys.stderr,
                )
                module_ok = False
            else:
                print(f"OK {path}: {pct:.1f}% >= {threshold}")

    if current_module == "":
        print("WARN: no module matched the coverage gates", file=sys.stderr)

    return 0 if (overall_ok and module_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

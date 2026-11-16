#!/usr/bin/env python3
"""Reproduce numerical results without figures or overwriting archived evidence.

Default: recompute all numerical summaries from saved cohort records and compare
them with the archived CSVs. No models are retrained. Use --verify for independent
checks of saved predictions, or --smoke for a fresh seed-42 HGB training run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
INPUT_DIRS = ("cohorts", "completion", "propensity_audit", "denominator_audit",
              "convergence_sensitivity", "lbe_optimizer_audit")


def link_or_copy(source: str, destination: str) -> str:
    """Only used for immutable cohort/prediction inputs in a separate work tree."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def run(work: Path, script: str, *arguments: str) -> None:
    print(f"Running {script} {' '.join(arguments)}", flush=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    subprocess.run([sys.executable, str(work / "code" / script), *arguments],
                   cwd=work, env=env, check=True)


def prepare(work: Path, *, cached: bool) -> None:
    if work.exists():
        raise SystemExit(f"Output already exists: {work}. Choose another --output directory.")
    if work == ROOT or ROOT.is_relative_to(work) or any(
        work.is_relative_to(ROOT / name) for name in ("results", "code", "original")
    ):
        raise SystemExit("Choose a separate output folder outside the archived results/code/original trees.")
    work.mkdir(parents=True)
    shutil.copytree(ROOT / "code", work / "code", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "original", work / "original")
    (work / "results" / "tables").mkdir(parents=True)
    if cached:
        for name in INPUT_DIRS:
            src = ROOT / "results" / name
            if not src.is_dir():
                raise SystemExit(f"Missing input directory: {src}")
            # The diagnostic script regenerates denominator records; these must
            # be independent copies, never hardlinks to archived evidence.
            copy_function = shutil.copy2 if name == "denominator_audit" else link_or_copy
            shutil.copytree(src, work / "results" / name, copy_function=copy_function)
        # These files are copied, not hardlinked: verification writes new reports.
        for src in (ROOT / "results").glob("*.json"):
            shutil.copy2(src, work / "results" / src.name)
        raw = ROOT / "results/tables/prospective_sampling_simulation_raw.csv"
        shutil.copy2(raw, work / "results/tables" / raw.name)


def compare_tables(work: Path) -> dict:
    import numpy as np
    import pandas as pd
    records = []
    failures = []
    for produced in sorted((work / "results/tables").glob("*.csv")):
        archived = ROOT / "results/tables" / produced.name
        if not archived.exists():
            records.append({"file": produced.name, "status": "new output"})
            continue
        a = pd.read_csv(archived)
        b = pd.read_csv(produced)
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, check_dtype=False,
                                          rtol=1e-10, atol=1e-12)
            numeric = a.select_dtypes(include="number").columns
            differences = [float(np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy())))
                           for c in numeric if len(a) and a[c].notna().any()]
            records.append({"file": produced.name, "status": "PASS", "rows": len(a),
                            "max_abs_numeric_difference": max(differences, default=0.0)})
        except AssertionError as exc:
            records.append({"file": produced.name, "status": "FAIL", "detail": str(exc)})
            failures.append(produced.name)
    result = {"status": "FAIL" if failures else "PASS", "tables": records,
              "comparison": "same columns and row order; numeric rtol=1e-10, atol=1e-12",
              "models_retrained": False, "plotting_code_used": False}
    (work / "comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("Table comparison failed: " + ", ".join(failures))
    return result


def smoke(work: Path) -> None:
    import numpy as np
    run(work, "run_r2.py", "--suite", "baseline", "--max-seeds", "1", "--workers", "1")
    archived = np.load(ROOT / "results/cohorts/baseline/42.npz")
    produced = np.load(work / "results/cohorts/baseline/42.npz")
    for name in ("ids", "y", "s", "models"):
        if not np.array_equal(archived[name], produced[name]):
            raise SystemExit(f"Smoke check failed: {name}")
    error = float(np.max(np.abs(archived["scores"] - produced["scores"])))
    result = {"status": "PASS" if error <= 1e-12 else "FAIL", "seed": 42,
              "scope": "Fresh baseline original HGB procedures and low-spending ranking only",
              "max_abs_score_difference": error,
              "exact_scores": bool(np.array_equal(archived["scores"], produced["scores"]))}
    (work / "smoke_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit("Smoke scores differ; use the recorded Python and dependency versions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--verify", action="store_true", help="Also verify saved predictions and numerical gradients")
    parser.add_argument("--smoke", action="store_true", help="Run one fresh seed-42 baseline; no aggregation")
    args = parser.parse_args()
    if args.smoke and args.verify:
        raise SystemExit("Use --smoke and --verify in separate invocations.")
    work = args.output.resolve()
    prepare(work, cached=not args.smoke)
    if args.smoke:
        smoke(work)
    else:
        run(work, "summarize_r2.py")
        run(work, "diagnostics_complete_r2.py")
        run(work, "diagnostics_complete_r2.py", "--fitted-only")
        run(work, "summarize_completion.py")
        run(work, "aggregate_optimizer_audit.py")
        run(work, "summarize_numerical_sensitivity.py")
        run(work, "make_numerical_tables.py")
        run(work, "export_figure_data.py", "--compare-to",
            str(ROOT / "results/figure_source_data"))
        report = compare_tables(work)
        print(f"Compared {sum(r['status'] == 'PASS' for r in report['tables'])} archived CSV tables.", flush=True)
        if args.verify:
            run(work, "verify_r2.py")
            run(work, "verify_completion.py")
            run(work, "verify_additional_audits.py")
    print(f"Completed. New outputs are in {work}", flush=True)


if __name__ == "__main__":
    main()

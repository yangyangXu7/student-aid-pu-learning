#!/usr/bin/env python3
"""Reproduce numerical results without figures or overwriting archived evidence.

Default: recompute all numerical summaries from saved cohort records and compare
them with the archived CSVs. No models are retrained. Use --verify for numerical
and retained-prediction checks (the verified scope is recorded), or --smoke for a
fresh seed-42 HGB training run. Nonbaseline prediction caches omitted from a
compact distribution do not affect the default numerical aggregation workflow.
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



def verify_baseline_completion(work: Path) -> dict:
    """Recompute every retained baseline method metric and LR consistency check."""
    import numpy as np
    sys.path.insert(0, str(work / "code"))
    from run_r2 import calc_metrics
    checked_scores = 0
    max_error = 0.0
    for seed in range(42, 142):
        with np.load(work / f"results/completion/baseline/{seed}.npz") as added, np.load(
            work / f"results/cohorts/baseline/{seed}.npz"
        ) as base, np.load(work / f"results/cohorts/logistic_learner/{seed}.npz") as logistic:
            for key in ("ids", "y", "s"):
                if not np.array_equal(added[key], base[key]):
                    raise SystemExit(f"Baseline prediction alignment failed: seed {seed}, {key}")
                if not np.array_equal(added[key], logistic[key]):
                    raise SystemExit(f"Logistic prediction alignment failed: seed {seed}, {key}")
            if not np.isfinite(added["scores"]).all() or not np.isfinite(added["propensities"]).all():
                raise SystemExit(f"Nonfinite baseline predictions: seed {seed}")
            record = json.loads((work / f"results/completion/baseline/{seed}.json").read_text())
            rows = {row["model"]: row for row in record["rows"]}
            names = added["models"].tolist()
            for name, scores in zip(names, added["scores"]):
                metrics, _ = calc_metrics(added["y"], added["s"], scores, added["ids"])
                for metric, value in metrics.items():
                    expected = rows[name][metric]
                    if value is None:
                        if expected is not None:
                            raise SystemExit(f"Metric mismatch: {seed}, {name}, {metric}")
                    else:
                        error = abs(float(value) - float(expected))
                        max_error = max(max_error, error)
                        if error > 1e-12:
                            raise SystemExit(f"Metric mismatch: {seed}, {name}, {metric}: {error}")
                checked_scores += 1
            for name in ("Naive", "CostSensitive", "ElkanNoto", "Spy", "TrueLabel"):
                original = logistic["scores"][logistic["models"].tolist().index(name)]
                fitted = added["scores"][names.index(name + "_LR")]
                if not np.array_equal(original, fitted):
                    raise SystemExit(f"Baseline logistic consistency failed: {seed}, {name}")
    report = {"status": "PASS", "baseline_cohorts": 100,
              "completion_score_vectors": checked_scores,
              "max_recomputed_metric_error": max_error,
              "logistic_consistency": "all five methods, all 100 baseline cohorts"}
    (work / "baseline_prediction_verification.json").write_text(json.dumps(report, indent=2))
    return report


def verify_available_predictions(work: Path) -> None:
    """Report exactly which saved predictions were independently rechecked."""
    missing = []
    for family in ("cohorts", "completion"):
        for record in (work / "results" / family).rglob("*.json"):
            if not record.with_suffix(".npz").is_file():
                missing.append(record.relative_to(work).as_posix())
    # The compact distribution retains the original verifier's three audit seeds
    # in every scenario, as well as all baseline and logistic-learner predictions.
    run(work, "verify_r2.py")
    if missing:
        run(work, "verify_completion.py", "--numerical-only")
        baseline = verify_baseline_completion(work)
    else:
        run(work, "verify_completion.py")
        baseline = verify_baseline_completion(work)
    run(work, "verify_additional_audits.py")
    report = {"status": "PASS", "models_retrained": False,
              "full_prediction_cache_available": not missing,
              "omitted_nonbaseline_prediction_cache_files": len(missing),
              "original_metric_checks": "audit seeds 42, 91, 141 wherever present in each scenario",
              "completion_baseline": baseline,
              "numerical_checks": "logistic and joint-likelihood gradients, soft-label equivalence, posterior identity",
              "additional_audits": "all included denominator, convergence, and optimizer prediction arrays",
              "limitation": ("Nonbaseline completion prediction arrays were not independently rechecked; regenerate omitted caches using the full retraining commands for the full prediction audit." if missing else None)}
    (work / "verification_scope.json").write_text(json.dumps(report, indent=2))
    print("Verification scope: " + ("retained baseline/audit predictions and numerical checks; full nonbaseline prediction audit requires regenerated caches." if missing else "complete prediction cache available."), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--verify", action="store_true", help="Verify numerical calculations and retained prediction arrays; record the exact scope")
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
            verify_available_predictions(work)
    print(f"Completed. New outputs are in {work}", flush=True)


if __name__ == "__main__":
    main()

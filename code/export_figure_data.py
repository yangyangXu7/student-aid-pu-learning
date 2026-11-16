#!/usr/bin/env python3
"""Recreate the numerical source data for the current manuscript's Figs 1-8.

No plotting library is imported and no image is produced. Summary CSVs and saved
seed-42 predictions are the only inputs. Historical figure numbering is ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = ["beta_0.0", "baseline", "logistic", "threshold", "heterogeneous",
              "reverse", "nonmonotonic", "unobserved"]
STRESS = ["poor_rate_0.05", "poor_rate_0.25", "frugal_rate_0.0", "frugal_rate_0.25",
          "overlap_0.5", "overlap_1.0", "noise_scale_1.5", "noise_scale_2.0",
          "false_positive_rate_0.01", "false_positive_rate_0.03", "alternative_factors"]
HGB = ["Naive", "CostSensitive", "ElkanNoto", "Spy", "TrueLabel", "LowSpending"]
FIG1_COLUMNS = ["Naive", "CostSensitive", "ElkanNoto", "Spy", "TrueLabel",
                "Naive_LR", "Spy_LR", "TrueLabel_LR", "SAR_EM", "R1_IndexEM", "LBE_LF"]
FIG1_AP_ORDER = ["Naive", "CostSensitive", "ElkanNoto", "Spy", "TrueLabel",
                 "Naive_LR", "Spy_LR", "SAR_EM", "R1_IndexEM", "LBE_LF", "TrueLabel_LR"]
FIG6 = ["Raw_div_latent_q_all", "Raw_div_latent_q_all_clipped",
        "Raw_div_conditional_eX", "Raw_div_eX_output_cap_1.0",
        "Cal_div_latent_q_all", "Cal_div_conditional_eX",
        "Known_g_div_latent_q_all", "Known_g_div_conditional_eX"]


def one(frame: pd.DataFrame, **criteria: object) -> dict:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in criteria.items():
        mask &= frame[key].to_numpy() == value
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(rows)}")
    return rows.iloc[0].to_dict()


def build(results: Path, tables: Path) -> dict[str, pd.DataFrame]:
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(tables / name)

    outputs = {}
    with np.load(results / "cohorts/baseline/42.npz") as base, np.load(
        results / "completion/baseline/42.npz"
    ) as added:
        for key in ("ids", "y", "s"):
            if not np.array_equal(base[key], added[key]):
                raise ValueError(f"Seed-42 prediction alignment failed: {key}")
        scores = {name: row for name, row in zip(base["models"].tolist(), base["scores"])}
        scores.update({name: row for name, row in zip(added["models"].tolist(), added["scores"])})
        unlabeled = base["s"] == 0
        y = base["y"][unlabeled]
        pred = pd.DataFrame({"id": base["ids"][unlabeled], "y": y,
                             **{name: scores[name][unlabeled] for name in FIG1_COLUMNS}})
    outputs["Fig1_seed42_predictions_U.csv"] = pred
    outputs["Fig1_AP_validation.csv"] = pd.DataFrame([
        {"model": name, "AP_U": average_precision_score(y, pred[name]),
         "n_U": len(y), "n_hidden": int(y.sum())} for name in FIG1_AP_ORDER
    ])

    # Figs 2, 3 and 7 retain the original cohort-bootstrap stream. The completed
    # benchmark uses a different deterministic stream and must not substitute it.
    capacity = read("capacity_summary.csv")
    outputs["Fig2_capacity_summary.csv"] = capacity[
        (capacity.scenario == "baseline") & capacity.model.isin(HGB)
    ].reset_index(drop=True)
    original_pairs = read("paired_method_differences.csv")
    outputs["Fig3_paired_differences.csv"] = original_pairs[
        original_pairs.scenario.isin(MECHANISMS)
        & (original_pairs.method_a == "Spy") & (original_pairs.method_b == "Naive")
        & original_pairs.metric.isin(["AP_full", "HR_U_10"])
    ].reset_index(drop=True)

    pairs = read("complete_paired_differences.csv")
    outputs["Fig4_plotted_values.csv"] = pd.DataFrame([
        one(pairs, scenario=scenario, method_a=method, method_b="Naive_LR", metric=metric)
        for metric in ["AP_U", "HR_U_10"]
        for method in ["Spy_LR", "SAR_EM", "R1_IndexEM", "LBE_LF"]
        for scenario in MECHANISMS
    ], columns=pairs.columns)
    outputs["Fig5_paired_stress_values.csv"] = pd.DataFrame([
        one(pairs, scenario=scenario, method_a="Spy", method_b="Naive", metric="AP_U")
        for scenario in STRESS
    ], columns=pairs.columns)
    denominator = read("denominator_audit_summary.csv")
    outputs["Fig6_plotted_values.csv"] = pd.DataFrame([
        one(denominator, model=model, metric="AP_full") for model in FIG6
    ], columns=denominator.columns)

    # Fig 7 filters the beta scenarios from summary_metrics; Fig 8 filters two
    # outcomes from the complete prospective sampling summary, both copied below.
    for name in ["summary_metrics.csv", "complete_summary_metrics.csv",
                 "complete_paired_differences.csv", "denominator_audit_summary.csv",
                 "em_convergence_summary.csv", "prospective_sampling_simulation_summary.csv",
                 "spy_grid_summary.csv"]:
        outputs[name] = read(name)
    summary = outputs["complete_summary_metrics.csv"]
    outputs["stress_absolute_values.csv"] = pd.DataFrame([
        one(summary, scenario=scenario, model=model, metric=metric)
        for metric in ["AP_U", "HR_U_10"]
        for scenario in STRESS
        for model in ["Spy", "SAR_EM", "R1_IndexEM", "LBE_LF", "TrueLabel_LR"]
    ], columns=summary.columns)
    return outputs


def compare(outputs: dict[str, pd.DataFrame], reference: Path) -> list[dict]:
    records = []
    for name, actual in outputs.items():
        path = reference / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing archived figure source data: {path}")
        expected = pd.read_csv(path)
        # Preserve schema, text values and row order. Numeric differences only
        # allow decimal serialization/aggregation round-off, not changed values.
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-12)
        records.append({"file": name, "rows": len(actual), "status": "PASS"})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tables-dir", type=Path,
                        help="Defaults to <results-dir>/tables")
    parser.add_argument("--output", type=Path,
                        help="Defaults to <results-dir>/figure_source_data")
    parser.add_argument("--compare-to", type=Path,
                        help="Compare all 15 CSVs with an existing archived source-data directory")
    args = parser.parse_args()
    results = args.results_dir.resolve()
    tables = (args.tables_dir or results / "tables").resolve()
    output = (args.output or results / "figure_source_data").resolve()
    if args.compare_to and output == args.compare_to.resolve():
        raise SystemExit("Use separate output and comparison directories.")
    data = build(results, tables)
    records = compare(data, args.compare_to.resolve()) if args.compare_to else []
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        frame.to_csv(output / name, index=False)
    report = {"status": "PASS" if records else "EXPORTED",
              "csv_files": len(data), "figures": "Current manuscript Figs 1-8",
              "plotting_code_used": False, "models_retrained": False,
              "comparison": "Exact schema/text/row order; numeric atol=1e-12, rtol=0",
              "files": records}
    (output / "numerical_export_report.json").write_text(json.dumps(report, indent=2))
    print(f"Exported {len(data)} numerical source CSVs for Figs 1-8 to {output}")


if __name__ == "__main__":
    main()

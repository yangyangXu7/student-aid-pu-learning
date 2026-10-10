# Student-Aid PU Learning

Identifying economically disadvantaged students from campus behavioural data using **positive and unlabeled (PU) learning**.

## Overview

Need-based financial aid is identified from campus behavioural records, but the administrative aid roster only records students who applied and were confirmed. Students who never applied are therefore *unlabeled*, not negative — a setting that standard supervised classifiers handle incorrectly. This repository implements a PU-learning treatment of the problem: positive examples (confirmed aid recipients) plus a large unlabeled set, with methods that explicitly account for instance-dependent labeling.

## Repository layout

| File | Purpose |
|---|---|
| `pu_methods.py` | PU estimators: Naive (U as N), Cost-sensitive, Spy two-step, Elkan–Noto weighted, and an Oracle ceiling built on true labels |
| `features_real.py` | Feature builder for the real campus records (card / dorm / library / score / subsidy) |
| `run_experiments.py` | End-to-end PU experiment and figures (synthetic mode by default; real mode optional) |
| `multi_seed.py` | Robustness aggregation of the full comparison over multiple random seeds |
| `sensitivity.py` | Labeling-mechanism sweep (β) and feature-group ablation |
| `requirements.txt` | Python dependencies |

## Installation

```
pip install -r requirements.txt
```

Dependencies: `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `matplotlib>=3.7`.

## Usage

### 1. Build features from real campus records

```
python features_real.py --peek data/raw/train    # inspect the column layout first
python features_real.py --build data/raw/train   # writes data/features_real.csv
```

`features_real.py` documents the expected raw files (`card_*.txt`, `dorm_*.txt`, `library_*.txt`, `score_*.txt`, `subsidy_*.txt`) and a `CONFIG` dict to adjust column indices if your mirror differs.

### 2. Run the PU experiment

```
python run_experiments.py              # synthetic mode (default)
python run_experiments.py --mode real  # requires data/features_real.csv
```

Outputs (metrics, evaluation arrays, figures) are written to `results/`.

### 3. Robustness and sensitivity analyses

```
python multi_seed.py    # aggregate metrics over multiple seeds
python sensitivity.py   # labeling-mechanism sweep + feature-group ablation
```

## Data

The raw campus records are **not redistributed** (privacy and third-party access). `features_real.py` describes the expected layout so you can rebuild the feature table from your own mirror of the data.

## Notes

The experiment scripts import a synthetic-data generator and a shared figure style module that are not part of this upload (`synth_data.py` for `run_experiments.py` / `multi_seed.py`; `plot_style.py` for `sensitivity.py`). Add them to run the scripts out of the box.

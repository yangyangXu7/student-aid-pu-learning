# Included datasets

The complete baseline datasets are included as 100 NumPy archives in
`results/cohorts/baseline/`: `42_data.npz` through `141_data.npz`.
Each file is one complete 12,000-student generated cohort.

| Array | Shape | Meaning |
| --- | --- | --- |
| `X` | 12000 by 14 | Observed feature matrix |
| `y` | 12000 | True hardship labels |
| `s` | 12000 | Observed aid-record labels |
| `train_ids` | 8400 | Training row indices |
| `test_ids` | 3600 | Test row indices |
| `features` | 14 | Feature names in column order |

```python
from pathlib import Path
import numpy as np

path = Path('results/cohorts/baseline/42_data.npz')
with np.load(path, allow_pickle=False) as data:
    X = data['X']
    y = data['y']
    s = data['s']
    feature_names = data['features'].tolist()
    train_ids = data['train_ids']
    test_ids = data['test_ids']
print(X.shape, y.shape, s.shape)
```

The 14 columns are monthly_total_spend, avg_meal_price, meals_per_day,
breakfast_rate, weekend_on_campus, canteen_ratio, snack_drink_ratio,
low_price_window_ratio, supermarket_spend, library_visits, dorm_night_out,
score_rank_pct, noise_a, and noise_b.

These are generated observations; no external individual student records are
included. The generator and scenario definitions are in `code/generator_r2.py`
and `protocol/scenario_configurations.csv`. The 100 baseline feature archives
remain at their original paths so the numerical reproduction scripts work
without path changes. Other scenario feature matrices are regenerated from the
included definitions when refitting.

All baseline predictions, all per-cohort JSON numerical records, all numerical
table files, and final figure-source data are included. Most nonbaseline
per-student prediction NPZ caches are omitted to keep this single ZIP small.
The retained nonbaseline caches are original-method predictions for seeds 42,
91, and 141 wherever those cases exist, all 100 `logistic_learner` predictions,
and all additional diagnostic/audit arrays. Numerical aggregation uses the
retained result records and requires no additional download. The full retraining
commands in `README.md` regenerate the omitted prediction caches.

Run `python check_datasets.py` from the project root to verify all 100 datasets,
their splits, and labels against both sets of baseline prediction files.

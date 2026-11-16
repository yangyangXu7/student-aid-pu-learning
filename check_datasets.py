"""Check the complete baseline datasets included with this project."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    for seed in range(42, 142):
        path = ROOT / 'results' / 'cohorts' / 'baseline' / f'{seed}_data.npz'
        if not path.is_file():
            raise SystemExit(f'Missing dataset: {path.relative_to(ROOT)}. Copy the complete project, including results/.')
        with np.load(path, allow_pickle=False) as data:
            X, y, s = data['X'], data['y'], data['s']
            train, test = data['train_ids'], data['test_ids']
            assert X.shape == (12000, 14), path
            assert y.shape == s.shape == (12000,), path
            assert data['features'].shape == (14,), path
            assert train.shape == (8400,) and test.shape == (3600,), path
            assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(12000)), path
            assert np.isfinite(X).all(), path
            assert np.isin(y, [0, 1]).all() and np.isin(s, [0, 1]).all(), path
            assert (s <= y).all(), path
            for folder in ('cohorts', 'completion'):
                prediction = ROOT / 'results' / folder / 'baseline' / f'{seed}.npz'
                with np.load(prediction, allow_pickle=False) as pred:
                    assert np.array_equal(pred['ids'], test), prediction
                    assert np.array_equal(pred['y'], y[test]), prediction
                    assert np.array_equal(pred['s'], s[test]), prediction
    print('PASS: 100 complete baseline datasets; 12,000 rows and 14 features each.')
    print('PASS: all training/test splits and labels match the saved baseline predictions.')

if __name__ == '__main__':
    main()

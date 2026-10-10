# -*- coding: utf-8 -*-
"""PU learning methods, all built on the same base classifier for a fair
comparison. `s` is the OBSERVED label: 1 = labeled positive (received aid),
0 = unlabeled (did NOT receive aid; mixture of true negatives and hidden
positives).

Implementation notes worth keeping for the paper:
* Elkan & Noto (2008) has two usages. The popular "rescale by s(x)/c" does
  NOT change the ranking (monotone transform of the naive scores), so it is
  useless for Top-K screening. We implement their *weighted retraining*
  variant (their Sec. 3), where every unlabeled example enters the training
  set twice -- once as positive with weight w(x)=P(y=1|x,s=0) and once as
  negative with weight 1-w(x). This DOES change the ranking.
* Spy technique (Liu et al., 2002): two-step strategy. Plant a fraction of
  known positives ("spies") into U, train labeled-vs-unlabeled, then treat
  unlabeled examples scoring below the spies' low quantile as reliable
  negatives (RN); retrain P vs RN.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def _hgb(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, random_state=seed
    )


def fit_naive(X, s, seed=0):
    """Traditional supervised baseline: unlabeled treated as negative."""
    m = _hgb(seed)
    m.fit(X, s)
    return (lambda Z: m.predict_proba(Z)[:, 1]), {}


def fit_cost_sensitive(X, s, seed=0):
    """Same as naive but with balanced sample weights."""
    w = np.where(s == 1, (s == 0).sum() / max((s == 1).sum(), 1), 1.0)
    m = _hgb(seed)
    m.fit(X, s, sample_weight=w)
    return (lambda Z: m.predict_proba(Z)[:, 1]), {}


def fit_spy(X, s, spy_frac=0.15, quantile=5, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(s == 1)[0]
    n_spy = max(int(len(pos) * spy_frac), 1)
    spies = rng.choice(pos, n_spy, replace=False)

    s_tmp = s.copy()
    s_tmp[spies] = 0
    m1 = _hgb(seed)
    m1.fit(X, s_tmp)
    sc = m1.predict_proba(X)[:, 1]

    thr = np.percentile(sc[spies], quantile)
    rn = (s == 0) & (sc < thr)
    if rn.sum() < 50:  # degenerate fallback
        thr = np.percentile(sc[s == 0], 30)
        rn = (s == 0) & (sc < thr)

    mask = (s == 1) | rn
    m2 = _hgb(seed)
    m2.fit(X[mask], s[mask])
    info = {"n_spies": int(n_spy), "n_reliable_neg": int(rn.sum()),
            "spy_threshold": float(thr)}
    return (lambda Z: m2.predict_proba(Z)[:, 1]), info


def fit_elkan_noto(X, s, c_holdout=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(s == 1)[0]
    hold = rng.choice(pos, max(int(len(pos) * c_holdout), 1), replace=False)
    tr = np.ones(len(s), bool)
    tr[hold] = False

    g = _hgb(seed)
    g.fit(X[tr], s[tr])
    c = float(np.clip(g.predict_proba(X[hold])[:, 1].mean(), 1e-3, 1 - 1e-3))

    gu = np.clip(g.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    w = np.clip((1 - c) / c * gu / (1 - gu), 0.0, 1.0)  # P(y=1 | x, s=0)

    U = s == 0
    X2 = np.vstack([X[s == 1], X[U], X[U]])
    y2 = np.concatenate([
        np.ones((s == 1).sum()), np.ones(U.sum()), np.zeros(U.sum())
    ])
    sw = np.concatenate([np.ones((s == 1).sum()), w[U], 1 - w[U]])

    f = _hgb(seed)
    f.fit(X2, y2, sample_weight=sw)
    return (lambda Z: f.predict_proba(Z)[:, 1]), {"c_hat": round(c, 4)}


def fit_oracle(X, y_true, seed=0):
    """Upper-bound reference: trained on the (normally unobservable) truth."""
    m = _hgb(seed)
    m.fit(X, y_true)
    return (lambda Z: m.predict_proba(Z)[:, 1]), {}

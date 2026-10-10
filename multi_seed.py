# -*- coding: utf-8 -*-
"""Robustness over simulation replicates: rerun the full comparison for ten
random seeds and aggregate mean and standard deviation of every metric.
Figures continue to display the seed 42 replicate; the manuscript table
reports the aggregate."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import pu_methods as pu
from synth_data import generate, FEATURES

SEEDS = list(range(42, 52))
KS = [0.05, 0.10, 0.15]


def run_one(sd):
    df = generate(seed=sd)
    X = df[FEATURES].to_numpy()
    y = df["y_true"].to_numpy()
    s = df["labeled"].to_numpy()
    strata = [f"{a}{b}" for a, b in zip(y, s)]
    tr, te = train_test_split(np.arange(len(df)), test_size=0.30,
                              random_state=7, stratify=strata)
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    str_, ste = s[tr], s[te]
    hid = ((yte == 1) & (ste == 0)).astype(int)

    en_f, en_info = pu.fit_elkan_noto(Xtr, str_)
    models = {
        "Naive": pu.fit_naive(Xtr, str_)[0],
        "CostSensitive": pu.fit_cost_sensitive(Xtr, str_)[0],
        "Spy": pu.fit_spy(Xtr, str_)[0],
        "ElkanNoto": en_f,
        "Oracle": pu.fit_oracle(Xtr, ytr)[0],
    }
    out = []
    for name, f in models.items():
        sc = f(Xte)
        row = {"seed": sd, "model": name,
               "AP": average_precision_score(yte, sc)}
        order = np.argsort(-sc)
        for kf in KS:
            k = int(len(yte) * kf)
            sel = order[:k]
            tag = int(kf * 100)
            row[f"P@{tag}"] = yte[sel].mean()
            row[f"R@{tag}"] = yte[sel].sum() / yte.sum()
            row[f"HR@{tag}"] = hid[sel].sum() / max(hid.sum(), 1)
        out.append(row)
    meta = {"seed": sd, "c_true": float(str_.sum() / ytr.sum()),
            "c_hat": en_info["c_hat"],
            "flag_frac": None}
    # fraction of hidden positives inside the Spy top 10% among unlabeled
    sc = models["Spy"](Xte)
    top = np.zeros(len(yte), bool)
    top[np.argsort(-sc)[: int(len(yte) * 0.10)]] = True
    g2 = (ste == 0) & top
    meta["flag_frac"] = float(hid[g2].mean()) if g2.sum() else 0.0
    return out, meta


if __name__ == "__main__":
    rows, metas = [], []
    for sd in SEEDS:
        r, m = run_one(sd)
        rows += r
        metas.append(m)
        print(f"seed {sd} done  c_true={m['c_true']:.3f} "
              f"c_hat={m['c_hat']:.3f} flag_frac={m['flag_frac']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv("results/multi_seed_raw.csv", index=False)
    agg = df.drop(columns="seed").groupby("model").agg(["mean", "std"])
    agg.round(4).to_csv("results/multi_seed_agg.csv")
    md = pd.DataFrame(metas)
    md.to_csv("results/multi_seed_meta.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", agg.round(3).to_string())
    print("\n c_true mean %.3f sd %.3f | c_hat mean %.3f sd %.3f | "
          "flag_frac mean %.3f sd %.3f" % (
              md.c_true.mean(), md.c_true.std(), md.c_hat.mean(),
              md.c_hat.std(), md.flag_frac.mean(), md.flag_frac.std()))

# -*- coding: utf-8 -*-
"""Sensitivity analyses.

1. Mechanism sweep: vary the slope beta of the labeling propensity
   e = clip(a - beta * typicality) while holding the mean application rate
   near 0.55 by setting a = 0.55 + 0.55 * beta. beta = 0 recovers the SCAR
   setting; larger beta strengthens the instance dependent mechanism.
2. Feature group ablation for the Spy procedure: drop one behavioral
   feature group at a time and measure the cost.

Five replicates per configuration, seeds 42 through 46.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import plot_style as ps
import pu_methods as pu
from synth_data import generate, FEATURES

SEEDS = list(range(42, 47))
BETAS = [0.0, 0.3, 0.62, 0.9]

GROUPS = {
    "consumption level": ["monthly_total_spend", "avg_meal_price",
                          "supermarket_spend"],
    "consumption structure": ["canteen_ratio", "snack_drink_ratio",
                              "low_price_window_ratio"],
    "behavioral regularity": ["meals_per_day", "breakfast_rate",
                              "weekend_on_campus"],
    "non card records": ["library_visits", "dorm_night_out",
                         "score_rank_pct"],
}


def split(df, feats):
    X = df[feats].to_numpy()
    y = df["y_true"].to_numpy()
    s = df["labeled"].to_numpy()
    strata = [f"{a}{b}" for a, b in zip(y, s)]
    tr, te = train_test_split(np.arange(len(df)), test_size=0.30,
                              random_state=7, stratify=strata)
    return (X[tr], X[te], y[tr], y[te], s[tr], s[te],
            ((y[te] == 1) & (s[te] == 0)).astype(int))


def hr_at(sc, hid, kf=0.10):
    sel = np.argsort(-sc)[: int(len(sc) * kf)]
    return hid[sel].sum() / max(hid.sum(), 1)


def mechanism_sweep():
    rows = []
    for beta in BETAS:
        a = 0.55 + 0.55 * beta
        for sd in SEEDS:
            df = generate(seed=sd, label_intercept=a, label_slope=beta,
                          label_lo=0.10, label_hi=0.92)
            Xtr, Xte, ytr, yte, str_, ste, hid = split(df, FEATURES)
            en_f, en_info = pu.fit_elkan_noto(Xtr, str_)
            models = {
                "Naive": pu.fit_naive(Xtr, str_)[0],
                "ElkanNoto": en_f,
                "Spy": pu.fit_spy(Xtr, str_)[0],
                "Oracle": pu.fit_oracle(Xtr, ytr)[0],
            }
            c_true = float(str_.sum() / ytr.sum())
            for name, f in models.items():
                sc = f(Xte)
                rows.append({"beta": beta, "seed": sd, "model": name,
                             "AP": average_precision_score(yte, sc),
                             "HR10": hr_at(sc, hid),
                             "c_true": c_true,
                             "c_hat": en_info["c_hat"]})
            print(f"beta={beta} seed={sd} done")
    df = pd.DataFrame(rows)
    df.to_csv("results/sweep_raw.csv", index=False)
    agg = df.groupby(["beta", "model"]).agg(
        AP_m=("AP", "mean"), AP_s=("AP", "std"),
        HR_m=("HR10", "mean"), HR_s=("HR10", "std"),
        ct=("c_true", "mean"), ch=("c_hat", "mean")).round(4)
    agg.to_csv("results/sweep_agg.csv")
    print(agg.to_string())
    return agg


def fig5(agg):
    ps.apply()
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    styles = {"Naive": (ps.MODEL_COLORS["Naive supervised (U=N)"], "-", "o"),
              "ElkanNoto": (ps.MODEL_COLORS["Elkan-Noto weighted (PU)"], "-", "s"),
              "Spy": (ps.MODEL_COLORS["Spy two-step (PU)"], "-", "D"),
              "Oracle": ("k", (0, (4, 2)), "^")}
    labels = {"Naive": "Naive (U as N)", "ElkanNoto": "Elkan\u2013Noto weighted",
              "Spy": "Spy two-step", "Oracle": "Oracle (ceiling)"}
    for m, (c, lsty, mk) in styles.items():
        sub = agg.xs(m, level="model")
        ax.errorbar(sub.index, sub["HR_m"], yerr=sub["HR_s"], color=c,
                    ls=lsty, marker=mk, ms=4.5, lw=1.6, capsize=2.5,
                    label=labels[m])
    ax.set_xlabel(r"Labeling mechanism slope $\beta$"
                  "\n0 = SCAR, larger = stronger instance dependence")
    ax.set_ylabel("Recall among hidden positives, top 10% list")
    ax.set_xticks(BETAS)
    ax.legend(loc="best")
    ps.save(fig, "results", "fig5_mechanism_sweep")
    plt.close(fig)
    print("[fig5] written")


def ablation():
    rows = []
    for sd in SEEDS:
        df = generate(seed=sd)
        for drop, cols in [("full model", [])] + list(GROUPS.items()):
            feats = [f for f in FEATURES if f not in cols]
            Xtr, Xte, ytr, yte, str_, ste, hid = split(df, feats)
            f = pu.fit_spy(Xtr, str_)[0]
            sc = f(Xte)
            rows.append({"seed": sd, "dropped": drop,
                         "AP": average_precision_score(yte, sc),
                         "HR10": hr_at(sc, hid)})
        print(f"ablation seed={sd} done")
    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_raw.csv", index=False)
    agg = df.groupby("dropped").agg(
        AP_m=("AP", "mean"), AP_s=("AP", "std"),
        HR_m=("HR10", "mean"), HR_s=("HR10", "std")).round(4)
    agg.to_csv("results/ablation_agg.csv")
    print(agg.to_string())


if __name__ == "__main__":
    agg = mechanism_sweep()
    fig5(agg)
    ablation()

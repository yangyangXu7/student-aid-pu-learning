# -*- coding: utf-8 -*-
"""End-to-end PU experiment. Default runs on calibrated SYNTHETIC data
(--mode synthetic). After you download the real DataCastle files, run
features_real.py first, then --mode real.

Evaluation uses ground-truth `y_true`, which exists only in synthetic mode.
In real mode, the competition's subsidy labels serve as (imperfect) truth --
exactly the caveat the paper must discuss.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

import pu_methods as pu
from synth_data import generate, FEATURES

META = {"y_true", "labeled", "typicality", "stu_id"}
KS = [0.05, 0.10, 0.15]


def topk(y, scores, hidden=None):
    order = np.argsort(-scores)
    out = {}
    for kf in KS:
        k = int(len(y) * kf)
        sel = order[:k]
        tag = f"{int(kf * 100)}%"
        out[f"P@{tag}"] = float(y[sel].mean())
        out[f"R@{tag}"] = float(y[sel].sum() / max(y.sum(), 1))
        if hidden is not None:
            out[f"HiddenR@{tag}"] = float(hidden[sel].sum() / max(hidden.sum(), 1))
    return out


def main(mode, outdir):
    os.makedirs(outdir, exist_ok=True)
    if mode == "synthetic":
        df = generate()
        feats = FEATURES
    else:
        df = pd.read_csv("data/features_real.csv")
        feats = [c for c in df.columns if c not in META]
        df["y_true"] = df["labeled"]  # best available truth in real mode

    X = df[feats].to_numpy()
    y = df["y_true"].to_numpy()
    s = df["labeled"].to_numpy()

    strata = [f"{a}{b}" for a, b in zip(y, s)]
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=7, stratify=strata)
    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    str_, ste = s[idx_tr], s[idx_te]
    hidden_te = ((yte == 1) & (ste == 0)).astype(int)

    fitters = {
        "Naive supervised (U=N)": lambda: pu.fit_naive(Xtr, str_),
        "Cost-sensitive": lambda: pu.fit_cost_sensitive(Xtr, str_),
        "Spy two-step (PU)": lambda: pu.fit_spy(Xtr, str_),
        "Elkan-Noto weighted (PU)": lambda: pu.fit_elkan_noto(Xtr, str_),
        "Oracle (true labels)": lambda: pu.fit_oracle(Xtr, ytr),
    }

    rows, scores, infos = [], {}, {}
    for name, make in fitters.items():
        f, info = make()
        sc = f(Xte)
        scores[name] = sc
        infos[name] = info
        row = {"model": name,
               "PR-AUC": float(average_precision_score(yte, sc))}
        row.update(topk(yte, sc, hidden=hidden_te))
        rows.append(row)
        print(f"[done] {name}  {info}")

    met = pd.DataFrame(rows).round(4)
    met.to_csv(f"{outdir}/metrics.csv", index=False)
    print("\n", met.to_string(index=False))

    # export arrays so figures can be re-styled without retraining
    np.savez(f"{outdir}/eval_arrays.npz",
             yte=yte, ste=ste, hidden=hidden_te, Xte=Xte,
             scores=np.vstack([scores[n] for n in fitters]))
    with open(f"{outdir}/eval_meta.json", "w") as fh:
        json.dump({"models": list(fitters), "feats": list(feats),
                   "mode": mode}, fh, ensure_ascii=False)

    colors = {"Naive supervised (U=N)": "#888888",
              "Cost-sensitive": "#b8860b",
              "Spy two-step (PU)": "#d62728",
              "Elkan-Noto weighted (PU)": "#1f77b4",
              "Oracle (true labels)": "#2ca02c"}

    # ---- Fig 1: PR curves --------------------------------------------------
    plt.figure(figsize=(6.2, 4.6))
    for name, sc in scores.items():
        p, r, _ = precision_recall_curve(yte, sc)
        ls = "--" if "Oracle" in name else "-"
        plt.plot(r, p, ls, color=colors[name], lw=1.8,
                 label=f"{name} (AP={average_precision_score(yte, sc):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (ground-truth labels, synthetic)")
    plt.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig1_pr_curves.png", dpi=200)
    plt.close()

    # ---- Fig 2: recall of HIDDEN positives at K ---------------------------
    names = [n for n in fitters if "Oracle" not in n]
    width = 0.25
    xpos = np.arange(len(names))
    plt.figure(figsize=(7.0, 4.2))
    for j, kf in enumerate(KS):
        vals = [met.loc[met.model == n, f"HiddenR@{int(kf*100)}%"].iloc[0]
                for n in names]
        plt.bar(xpos + (j - 1) * width, vals, width,
                label=f"top {int(kf*100)}% list")
    plt.xticks(xpos, [n.replace(" (", "\n(") for n in names], fontsize=8)
    plt.ylabel("Recall among HIDDEN positives")
    plt.title("How many unreported poor students does each model recover?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig2_hidden_recall.png", dpi=200)
    plt.close()

    # ---- Fig 3: behavioural profiles of three groups ----------------------
    pu_sc = scores["Spy two-step (PU)"]
    order = np.argsort(-pu_sc)
    top10 = np.zeros(len(yte), bool)
    top10[order[:int(len(yte) * 0.10)]] = True
    g1 = ste == 1                      # labeled positives in test
    g2 = (ste == 0) & top10            # unlabeled flagged by PU model
    g3 = (ste == 0) & ~top10           # remaining unlabeled
    frac_hidden_g2 = float(hidden_te[g2].mean()) if g2.sum() else 0.0

    show = ["avg_meal_price", "meals_per_day", "breakfast_rate",
            "weekend_on_campus", "low_price_window_ratio",
            "snack_drink_ratio", "monthly_total_spend", "library_visits"]
    show = [f for f in show if f in feats]
    Xte_df = pd.DataFrame(Xte, columns=feats)
    mu, sd = Xte_df[show].mean(), Xte_df[show].std().replace(0, 1)
    Z = (Xte_df[show] - mu) / sd
    prof = pd.DataFrame({
        "Aid recipients (labeled P)": Z[g1].mean(),
        f"PU-flagged unlabeled (hidden-pos rate {frac_hidden_g2:.0%})": Z[g2].mean(),
        "Other unlabeled": Z[g3].mean(),
    })
    ax = prof.plot(kind="bar", figsize=(8.2, 4.4),
                   color=["#1f77b4", "#d62728", "#bbbbbb"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("z-scored feature mean")
    ax.set_title("PU-flagged students behave like aid recipients")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig3_profiles.png", dpi=200)
    plt.close()

    # ---- Fig 4: naive vs PU score scatter ---------------------------------
    nv = scores["Naive supervised (U=N)"]
    plt.figure(figsize=(5.6, 5.2))
    neg = (yte == 0)
    plt.scatter(nv[neg], pu_sc[neg], s=5, c="#cccccc", alpha=0.4,
                label="true negatives")
    plt.scatter(nv[ste == 1], pu_sc[ste == 1], s=9, c="#1f77b4", alpha=0.6,
                label="labeled positives")
    hp = hidden_te == 1
    plt.scatter(nv[hp], pu_sc[hp], s=12, c="#d62728", alpha=0.75,
                label="hidden positives")
    lim = [0, 1]
    plt.plot(lim, lim, "k--", lw=0.7)
    plt.xlabel("Naive supervised score")
    plt.ylabel("Spy two-step (PU) score")
    plt.title("Hidden positives sit above the diagonal:\nthe PU model rescues them")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig4_score_scatter.png", dpi=200)
    plt.close()

    summary = {
        "mode": mode,
        "n_total": int(len(df)),
        "n_positive_true": int(y.sum()),
        "n_labeled": int(s.sum()),
        "n_hidden_positives": int(((y == 1) & (s == 0)).sum()),
        "test_size": int(len(yte)),
        "hidden_fraction_in_pu_top10": frac_hidden_g2,
        "method_info": infos,
    }
    with open(f"{outdir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\nsummary:", json.dumps(summary, indent=2, ensure_ascii=False))

    # restyle figures to publication quality (overwrites the drafts above)
    try:
        import make_figures
        make_figures.make_all(outdir)
    except Exception as e:  # pragma: no cover
        print("figure restyle skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    main(args.mode, args.outdir)

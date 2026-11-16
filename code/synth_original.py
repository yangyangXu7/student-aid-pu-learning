# -*- coding: utf-8 -*-
"""Synthetic data generator, calibrated to the DataCastle subsidy dataset structure.

Ground-truth poverty (`y_true`) is KNOWN here, which is only possible in
simulation. Purpose: (1) validate the full PU pipeline end-to-end,
(2) demonstrate PU behaviour under a *non-SCAR* labeling mechanism
    (the most "typical" poor students are the least likely to apply).

Numbers produced from this generator MUST NOT be reported as main results.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "monthly_total_spend",     # 月均总消费
    "avg_meal_price",          # 单餐均价
    "meals_per_day",           # 日均食堂就餐次数
    "breakfast_rate",          # 早餐率
    "weekend_on_campus",       # 周末留校率(以周末有无消费近似)
    "canteen_ratio",           # 食堂消费金额占比
    "snack_drink_ratio",       # 超市零食饮料占比
    "low_price_window_ratio",  # 低价窗口就餐占比
    "supermarket_spend",       # 月均超市消费
    "library_visits",          # 月均图书馆进馆次数
    "dorm_night_out",          # 月均晚归/夜出次数
    "score_rank_pct",          # 成绩排名百分位(越小越好)
    "noise_a", "noise_b",      # 纯噪声特征(哨兵)
]


def generate(n=12000, poor_rate=0.13, frugal_rate=0.10, seed=42):
    rng = np.random.default_rng(seed)
    poor = rng.random(n) < poor_rate
    # typicality in [0,1]: how classic the poverty behaviour pattern is
    typ = rng.beta(2.2, 1.8, n)
    frugal = (~poor) & (rng.random(n) < frugal_rate)  # 节俭的非困难生(混淆项)

    def blend(p_typ, p_atyp, rich, frug, sd):
        base = np.where(
            poor, typ * p_typ + (1 - typ) * p_atyp,
            np.where(frugal, frug, rich),
        )
        return base + rng.normal(0, sd, n)

    avg_meal_price = np.clip(blend(5.2, 8.5, 10.5, 7.5, 1.2), 2.0, None)
    meals_per_day = np.clip(blend(2.7, 2.0, 1.6, 2.3, 0.35), 0.3, 3.5)
    breakfast_rate = np.clip(blend(0.78, 0.45, 0.28, 0.50, 0.10), 0, 1)
    weekend_on_campus = np.clip(blend(0.85, 0.60, 0.42, 0.55, 0.12), 0, 1)
    canteen_ratio = np.clip(blend(0.88, 0.70, 0.55, 0.72, 0.08), 0, 1)
    snack_drink_ratio = np.clip(blend(0.05, 0.15, 0.30, 0.15, 0.06), 0, 1)
    low_price_window_ratio = np.clip(blend(0.55, 0.30, 0.12, 0.30, 0.10), 0, 1)
    supermarket_spend = np.clip(blend(60, 150, 380, 160, 60), 0, None)
    library_visits = np.clip(blend(14, 10, 8, 10, 4), 0, None)
    dorm_night_out = np.clip(blend(1.5, 3.0, 5.5, 3.0, 1.2), 0, None)
    score_rank_pct = np.clip(blend(0.44, 0.50, 0.52, 0.50, 0.18), 0, 1)
    monthly_total_spend = (
        meals_per_day * 30 * avg_meal_price + supermarket_spend
        + rng.normal(0, 40, n)
    )

    df = pd.DataFrame({
        "monthly_total_spend": monthly_total_spend,
        "avg_meal_price": avg_meal_price,
        "meals_per_day": meals_per_day,
        "breakfast_rate": breakfast_rate,
        "weekend_on_campus": weekend_on_campus,
        "canteen_ratio": canteen_ratio,
        "snack_drink_ratio": snack_drink_ratio,
        "low_price_window_ratio": low_price_window_ratio,
        "supermarket_spend": supermarket_spend,
        "library_visits": library_visits,
        "dorm_night_out": dorm_night_out,
        "score_rank_pct": score_rank_pct,
        "noise_a": rng.normal(0, 1, n),
        "noise_b": rng.normal(0, 1, n),
    })
    df["y_true"] = poor.astype(int)

    # --- non-SCAR labeling mechanism -------------------------------------
    # P(apply | poor, typicality): the MORE typical the pattern, the LESS
    # likely the student applies (self-esteem / "others need it more").
    p_apply = np.clip(0.88 - 0.62 * typ, 0.12, 0.88)
    labeled = poor & (rng.random(n) < p_apply)
    df["labeled"] = labeled.astype(int)
    df["typicality"] = np.where(poor, typ, np.nan)
    return df


if __name__ == "__main__":
    d = generate()
    print(d[["y_true", "labeled"]].sum())
    hid = ((d.y_true == 1) & (d.labeled == 0)).sum()
    print("hidden positives:", hid, "/", d.y_true.sum())
    print("mean typicality  labeled:", d.loc[d.labeled == 1, "typicality"].mean().round(3),
          " hidden:", d.loc[(d.y_true == 1) & (d.labeled == 0), "typicality"].mean().round(3))

# -*- coding: utf-8 -*-
"""Feature builder for the REAL DataCastle files.

STATUS: EXPERIMENTAL -- written against the column layout used by the
first-place solution (lzddzh/DataMiningCompetitionFirstPrize), but I could
not run it against the real files inside this sandbox (data lives behind a
Baidu-pan login). BEFORE first use:

    python features_real.py --peek data/raw/train

to print the first lines of every file, then fix CONFIG below if your
mirror's column order differs.

Expected files under data/raw/train/ (and .../test/ with *_test names):
    card_train.txt      stu_id, deal_type, place, category, datetime, amount, balance
                        category in {食堂, 超市, 开水, 洗衣房, 淋浴, 图书馆,
                                     校医院, 教务处, 文印中心, 校车, 其他}
    dorm_train.txt      stu_id, datetime, direction(0/1)
    library_train.txt   stu_id, gate, datetime
    borrow_train.txt    stu_id, date, book[, code]
    score_train.txt     stu_id, college_id, rank
    subsidy_train.txt   stu_id, subsidy_amount (0/1000/1500/2000)
"""
import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

CONFIG = {
    "card":    {"id": 0, "category": 3, "datetime": 4, "amount": 5},
    "dorm":    {"id": 0, "datetime": 1, "direction": 2},
    "library": {"id": 0, "datetime": 2},
    "score":   {"id": 0, "college": 1, "rank": 2},
    "subsidy": {"id": 0, "amount": 1},
    "datetime_formats": ["%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"],
    "encoding": "utf-8",
}

CANTEEN, SUPERMARKET = "食堂", "超市"


def _parse_dt(s):
    s = s.strip().strip('"')
    for fmt in CONFIG["datetime_formats"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _rows(path, n_cols_min):
    with open(path, encoding=CONFIG["encoding"], errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= n_cols_min:
                yield [c.strip().strip('"') for c in row]


def build_features(raw_dir):
    c = CONFIG
    # ---------- card ----------
    agg = defaultdict(lambda: {
        "amt_total": 0.0, "amt_canteen": 0.0, "amt_market": 0.0,
        "n_canteen": 0, "canteen_prices": [], "days": set(),
        "weekend_days": set(), "breakfast_days": set(), "span": [None, None],
    })
    card_path = _find(raw_dir, "card")
    for row in _rows(card_path, 6):
        try:
            sid = int(row[c["card"]["id"]])
            amt = float(row[c["card"]["amount"]])
        except ValueError:
            continue
        dt = _parse_dt(row[c["card"]["datetime"]])
        if dt is None or amt <= 0:
            continue
        cat = row[c["card"]["category"]]
        a = agg[sid]
        a["amt_total"] += amt
        d = dt.date()
        a["days"].add(d)
        if a["span"][0] is None or dt < a["span"][0]:
            a["span"][0] = dt
        if a["span"][1] is None or dt > a["span"][1]:
            a["span"][1] = dt
        if d.weekday() >= 5:
            a["weekend_days"].add(d)
        if cat == CANTEEN:
            a["amt_canteen"] += amt
            a["n_canteen"] += 1
            a["canteen_prices"].append(amt)
            if dt.hour < 10:
                a["breakfast_days"].add(d)
        elif cat == SUPERMARKET:
            a["amt_market"] += amt

    all_prices = np.concatenate(
        [np.asarray(v["canteen_prices"]) for v in agg.values()
         if v["canteen_prices"]]) if agg else np.array([1.0])
    low_thr = np.percentile(all_prices, 25)

    rows = []
    for sid, a in agg.items():
        span_days = max((a["span"][1] - a["span"][0]).days, 1) if a["span"][0] else 1
        months = max(span_days / 30.0, 1.0)
        n_days = max(len(a["days"]), 1)
        prices = np.asarray(a["canteen_prices"]) if a["canteen_prices"] else np.array([0.0])
        n_weekends = max(int(span_days / 7 * 2), 1)
        rows.append({
            "stu_id": sid,
            "monthly_total_spend": a["amt_total"] / months,
            "avg_meal_price": float(prices.mean()),
            "meals_per_day": a["n_canteen"] / n_days,
            "breakfast_rate": len(a["breakfast_days"]) / n_days,
            "weekend_on_campus": min(len(a["weekend_days"]) / n_weekends, 1.0),
            "canteen_ratio": a["amt_canteen"] / max(a["amt_total"], 1e-9),
            "snack_drink_ratio": a["amt_market"] / max(a["amt_total"], 1e-9),
            "low_price_window_ratio": float((prices <= low_thr).mean()),
            "supermarket_spend": a["amt_market"] / months,
        })
    df = pd.DataFrame(rows).set_index("stu_id")

    # ---------- library ----------
    lib = defaultdict(int)
    for row in _rows(_find(raw_dir, "library"), 2):
        try:
            lib[int(row[c["library"]["id"]])] += 1
        except ValueError:
            continue
    df["library_visits"] = pd.Series(lib) / 9.0  # ~one academic year

    # ---------- dorm (late-night entries) ----------
    night = defaultdict(int)
    for row in _rows(_find(raw_dir, "dorm"), 2):
        try:
            sid = int(row[c["dorm"]["id"]])
        except ValueError:
            continue
        dt = _parse_dt(row[c["dorm"]["datetime"]])
        if dt and (dt.hour >= 23 or dt.hour < 6):
            night[sid] += 1
    df["dorm_night_out"] = pd.Series(night) / 9.0

    # ---------- score ----------
    ranks, college_size = {}, defaultdict(int)
    for row in _rows(_find(raw_dir, "score"), 3):
        try:
            sid = int(row[c["score"]["id"]])
            col = row[c["score"]["college"]]
            rk = float(row[c["score"]["rank"]])
        except ValueError:
            continue
        ranks[sid] = (col, rk)
        college_size[col] = max(college_size[col], rk)
    df["score_rank_pct"] = pd.Series(
        {sid: rk / max(college_size[col], 1) for sid, (col, rk) in ranks.items()})

    # ---------- subsidy label ----------
    sub = {}
    for row in _rows(_find(raw_dir, "subsidy"), 2):
        try:
            sub[int(row[c["subsidy"]["id"]])] = float(row[c["subsidy"]["amount"]])
        except ValueError:
            continue
    df["labeled"] = (pd.Series(sub) > 0).astype(int)

    df = df.fillna(0.0)
    df = df[df["labeled"].notna()] if "labeled" in df else df
    return df.reset_index()


def _find(raw_dir, key):
    for f in os.listdir(raw_dir):
        if f.startswith(key):
            return os.path.join(raw_dir, f)
    raise FileNotFoundError(f"no file starting with '{key}' in {raw_dir}")


def peek(raw_dir, n=2):
    for f in sorted(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, f)
        print(f"\n===== {f} =====")
        with open(p, encoding=CONFIG["encoding"], errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                print("  ", line.rstrip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", metavar="DIR")
    ap.add_argument("--build", metavar="DIR")
    args = ap.parse_args()
    if args.peek:
        peek(args.peek)
    elif args.build:
        out = build_features(args.build)
        out.to_csv("data/features_real.csv", index=False)
        print(out.describe().T)

from __future__ import annotations
import random
from typing import Dict, List, Sequence
import numpy as np
 
 
# Test set construction
 
def _sample_random_bundles(m: int, n: int, rng: random.Random) -> List[tuple]:

    bundles = set()
    attempts = 0
    max_attempts = 20 * n
    while len(bundles) < n and attempts < max_attempts:
        sz = rng.randint(1, m)
        bundles.add(tuple(sorted(rng.sample(range(m), sz))))
        attempts += 1
    return list(bundles)
 
 
def _estimate_avg_item_value(bidder, n_probe: int = 200,
                             rng: random.Random | None = None) -> np.ndarray:
    
    rng = rng or random.Random(0)
    m = bidder.m
    sums = np.zeros(m)
    counts = np.zeros(m)
 
    for _ in range(n_probe):
        sz = rng.randint(1, m)
        bundle = tuple(sorted(rng.sample(range(m), sz)))
        v_with = bidder.value(bundle)
        for j in bundle:
            b_minus = tuple(x for x in bundle if x != j)
            v_without = bidder.value(b_minus) if b_minus else 0.0
            sums[j] += (v_with - v_without)
            counts[j] += 1
 
    avg = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    floor = max(float(np.mean(np.abs(avg))) * 0.05, 1e-3)
    avg = np.maximum(avg, floor)
    return avg
 
 
def _true_utility_max_bundle(bidder, prices: Sequence[float]) -> tuple:

    candidates = bidder._demand_candidates()
    best_bundle = ()
    best_util = 0.0
    for b in candidates:
        v = bidder.value(b)
        u = v - sum(prices[j] for j in b)
        if u > best_util:
            best_util = u
            best_bundle = b
    return tuple(sorted(best_bundle))
 
 
def _sample_price_driven_bundles(bidder, n: int, rng: random.Random,
                                 price_multiplier: float = 3.0,
                                 avg_item_value: np.ndarray | None = None
                                 ) -> List[tuple]:

    m = bidder.m
    if avg_item_value is None:
        avg_item_value = _estimate_avg_item_value(bidder, rng=rng)
 
    bundles: List[tuple] = []
    for _ in range(n):
        prices = [rng.uniform(0.0, price_multiplier * avg_item_value[j])
                  for j in range(m)]
        bundles.append(_true_utility_max_bundle(bidder, prices))
    return bundles
 
# Metrics section
 
# Standard R^2 = 1 - SS_res / SS_tot
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   
    y_mean = y_true.mean()
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot
 
#  Centered R^2: subtract means from both sides, then standard R^2
def _r2_centered(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    ss_tot = float(np.sum(yt ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot
 
# Kendall tau-b rank correlation (uses scipy if available, otherwise an O(n^2) implementation)
def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    n = len(y_true)
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(y_true, y_pred)
        return float(tau) if tau == tau else float("nan")
    except ImportError:
        pass
 
    concordant = 0
    discordant = 0
    tie_t = 0
    tie_p = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            if dt == 0 and dp == 0:
                tie_t += 1
                tie_p += 1
            elif dt == 0:
                tie_t += 1
            elif dp == 0:
                tie_p += 1
            elif (dt > 0) == (dp > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    denom = ((total + tie_t) * (total + tie_p)) ** 0.5
    if denom < 1e-12:
        return float("nan")
    return (concordant - discordant) / denom
 
# MAE / mean true value on the set
def _scaled_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mt = float(y_true.mean())
    if abs(mt) < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / mt)
 

# Run the 4 metrcis 
def _compute_metrics(bidder, bundles: List[tuple]) -> Dict[str, float]:

    if not bundles:
        return {"n": 0, "R2": float("nan"), "R2c": float("nan"),
                "KT": float("nan"), "sMAE": float("nan"),
                "mean_true": float("nan"), "mean_pred": float("nan")}
 
    y_true = np.array([bidder.value(b) for b in bundles], dtype=np.float64)
    y_pred = np.array([bidder.predict_value(b) for b in bundles],
                      dtype=np.float64)
 
    return {
        "n": len(bundles),
        "R2":   _r2(y_true, y_pred),
        "R2c":  _r2_centered(y_true, y_pred),
        "KT":   _kendall_tau(y_true, y_pred),
        "sMAE": _scaled_mae(y_true, y_pred),
        "mean_true": float(y_true.mean()),
        "mean_pred": float(y_pred.mean()),
    }
 
 
 
def evaluate_bidder(bidder,
                    n_test: int = 200,
                    seed: int = 42,
                    price_multiplier: float = 3.0,
                    avg_item_value: np.ndarray | None = None
                    ) -> Dict[str, Dict[str, float]]:

    rng_r = random.Random(seed)
    rng_p = random.Random(seed + 1)
 
    T_r = _sample_random_bundles(bidder.m, n_test, rng_r)
    T_p = _sample_price_driven_bundles(
        bidder, n_test, rng_p,
        price_multiplier=price_multiplier,
        avg_item_value=avg_item_value,
    )
 
    return {
        "T_r": _compute_metrics(bidder, T_r),
        "T_p": _compute_metrics(bidder, T_p),
    }
 
 
def snapshot_quality(bidders, label: str, n_test: int = 200,
                     seed: int = 42, print_table: bool = True
                     ) -> List[Dict[str, Dict[str, float]]]:
    
    rows: List[Dict[str, Dict[str, float]]] = []
 
    for i, b in enumerate(bidders):
        m = evaluate_bidder(b, n_test=n_test, seed=seed + i)
        rows.append(m)
 
    if print_table:
        print(f"\n=== ML quality @ {label} "
              f"(n_test={n_test} per set) ===")
        header = (f"  {'bid':<4}{'|R_DQ|':<8}{'|R_VQ|':<8}"
                  f"{'R2_r':>8}{'R2c_r':>8}{'KT_r':>8}{'sMAE_r':>9}"
                  f"  | "
                  f"{'R2_p':>8}{'R2c_p':>8}{'KT_p':>8}{'sMAE_p':>9}")
        print(header)
        print("  " + "-" * (len(header) - 2))
 
        for i, (b, m) in enumerate(zip(bidders, rows)):
            tr = m["T_r"]
            tp = m["T_p"]
            print(
                f"  {i:<4}{len(b.R_DQ):<8}{len(b.R_VQ):<8}"
                f"{tr['R2']:>8.2f}{tr['R2c']:>8.2f}"
                f"{tr['KT']:>8.2f}{tr['sMAE']:>9.3f}"
                f"  | "
                f"{tp['R2']:>8.2f}{tp['R2c']:>8.2f}"
                f"{tp['KT']:>8.2f}{tp['sMAE']:>9.3f}"
            )
 
        def _agg(set_key, metric_key):
            vals = [m[set_key][metric_key] for m in rows
                    if not np.isnan(m[set_key][metric_key])]
            return float(np.mean(vals)) if vals else float("nan")
 
        print("  " + "-" * (len(header) - 2))
        print(
            f"  {'avg':<4}{'':<8}{'':<8}"
            f"{_agg('T_r','R2'):>8.2f}{_agg('T_r','R2c'):>8.2f}"
            f"{_agg('T_r','KT'):>8.2f}{_agg('T_r','sMAE'):>9.3f}"
            f"  | "
            f"{_agg('T_p','R2'):>8.2f}{_agg('T_p','R2c'):>8.2f}"
            f"{_agg('T_p','KT'):>8.2f}{_agg('T_p','sMAE'):>9.3f}"
        )
 
    return rows
 

# Single run: python main.py --model linear --valuation additive
# Multi-run: python main.py --multi --model linear --valuation pairwise


import argparse
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config
from bidder import Bidder
from auction import run_mlhca, run_cca_only
from utils import format_bundle

from valuation_generators import (
    create_additive_valuations,
    create_synergy_valuations,
    create_complementarity_valuations,
    create_mixed_valuations,
)

# Plot styling 
_LINE_COLOR    = "#2563EB"   
_FILL_COLOR    = "#2563EB"   
_ML_DQ_COLOR   = "#16A34A"   
_BRIDGE_COLOR  = "#DC2626"   
_ML_VQ_COLOR   = "#7C3AED"   


def make_bidders(seed: int = config.SEED_BASE):
    valuation_type = config.VALUATION_TYPE

    if valuation_type == "sats":
        from sats_valuation import create_sats_lsvm_valuations  
        val_fns, _, _ = create_sats_lsvm_valuations(seed=seed)


    elif valuation_type == "additive":
        val_fns = create_additive_valuations(
            config.N_BIDDERS,
            config.N_ITEMS,
            seed,
        )

    elif valuation_type == "pairwise":
        val_fns = create_synergy_valuations(
            config.N_BIDDERS,
            config.N_ITEMS,
            seed,
        )

    elif valuation_type == "complementarity":
        val_fns = create_complementarity_valuations(
            config.N_BIDDERS,
            config.N_ITEMS,
            seed,
        )

    elif valuation_type == "mixed":
        val_fns = create_mixed_valuations(
            config.N_BIDDERS,
            config.N_ITEMS,
            seed,
        )

    else:
        raise ValueError(f"Unknown VALUATION_TYPE: {valuation_type}")

    return [
        Bidder(i, val_fns[i], config.N_ITEMS)
        for i in range(config.N_BIDDERS)
    ]


def save_multirun_results(results, valuation, model, use_bridge):
    os.makedirs("results", exist_ok=True)

    bridge_label = "bridge" if use_bridge else "nobridge"
    save_path = f"results/results_{valuation}_{model}_{bridge_label}.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "run", "efficiency", "true_welfare", "opt_welfare",
            "total_queries", "n_dq", "n_vq",
            "dq_end_round", "bridge_round",
            "ml_dq_start_round", "ml_vq_start_round",
        ])

        for i, result in enumerate(results, start=1):
            writer.writerow([
                i,
                result.efficiency,
                result.true_welfare,
                result.opt_welfare,
                result.total_queries,
                result.n_dq,
                result.n_vq,
                result.dq_end_round,
                result.bridge_round,
                result.ml_dq_start_round,
                result.ml_vq_start_round,
            ])

    print(f"Saved results → {save_path}")


def save_multirun_records(results, valuation, model, use_bridge):
    os.makedirs("results/records", exist_ok=True)

    bridge_label = "bridge" if use_bridge else "nobridge"
    save_path = f"results/records/records_{valuation}_{model}_{bridge_label}.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "run", "round_num", "total_queries", "efficiency",
        ])

        for run_idx, result in enumerate(results, start=1):
            for rec in result.records:
                writer.writerow([
                    run_idx,
                    rec.round_num,
                    rec.total_queries,
                    rec.efficiency,
                ])

    print(f"Saved records → {save_path}")


def save_multirun_quality(results, valuation, model, use_bridge):

    os.makedirs("results/quality", exist_ok=True)

    bridge_label = "bridge" if use_bridge else "nobridge"
    save_path = (
        f"results/quality/quality_{valuation}_{model}_{bridge_label}.csv"
    )

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "run", "phase", "bidder",
            "R2_r", "R2c_r", "KT_r", "sMAE_r",
            "R2_p", "R2c_p", "KT_p", "sMAE_p",
            "mean_true_r", "mean_pred_r",
            "mean_true_p", "mean_pred_p",
        ])

        for run_idx, result in enumerate(results, start=1):
            for snap in result.quality_snapshots:
                for bidder_idx, m in enumerate(snap["rows"]):
                    tr, tp = m["T_r"], m["T_p"]
                    writer.writerow([
                        run_idx,
                        snap["label"],
                        bidder_idx,
                        tr["R2"], tr["R2c"], tr["KT"], tr["sMAE"],
                        tp["R2"], tp["R2c"], tp["KT"], tp["sMAE"],
                        tr["mean_true"], tr["mean_pred"],
                        tp["mean_true"], tp["mean_pred"],
                    ])

    print(f"Saved quality metrics → {save_path}")


def print_allocation(allocation, bidders, label: str):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    total = 0.0
    for i, bundle in enumerate(allocation):
        v = bidders[i].value(bundle)
        total += v
        print(f"  Bidder {i + 1}: {format_bundle(bundle)}  |  value = {v:.4f}")

    print(f"  Total welfare : {total:.4f}")


def print_summary(result):
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"  Optimal welfare   : {result.opt_welfare:.4f}")
    print(f"  Final welfare     : {result.true_welfare:.4f}")
    print(f"  Efficiency        : {result.efficiency:.2f}%")
    print(f"  Total queries     : {result.total_queries}")
    print(f"  Demand queries    : {result.n_dq}")
    print(f"  Value queries     : {result.n_vq}")
    print(f"  DQ end round      : {result.dq_end_round}")
    print(f"  Bridge round      : {result.bridge_round}")
    print(f"  ML DQ start round : {result.ml_dq_start_round}")
    print(f"  ML VQ start round : {result.ml_vq_start_round}")

    # Per-phase metric trajectory (averaged over bidders)
    if result.quality_snapshots:
        print(f"\n  Model quality trajectory (avg over bidders):")
        print(f"    {'phase':<15}"
              f"{'R2_r':>8}{'R2c_r':>8}{'KT_r':>8}{'sMAE_r':>9}"
              f"  |"
              f"{'R2_p':>8}{'R2c_p':>8}{'KT_p':>8}{'sMAE_p':>9}")

        for snap in result.quality_snapshots:
            rows = snap["rows"]

            def avg(key_set, key_metric):
                vals = [r[key_set][key_metric] for r in rows
                        if not np.isnan(r[key_set][key_metric])]
                return np.mean(vals) if vals else float("nan")

            print(f"    {snap['label']:<15}"
                  f"{avg('T_r','R2'):>8.2f}{avg('T_r','R2c'):>8.2f}"
                  f"{avg('T_r','KT'):>8.2f}{avg('T_r','sMAE'):>9.3f}"
                  f"  |"
                  f"{avg('T_p','R2'):>8.2f}{avg('T_p','R2c'):>8.2f}"
                  f"{avg('T_p','KT'):>8.2f}{avg('T_p','sMAE'):>9.3f}")


def _interpolate_runs(all_results, grid_q):
    mat = np.zeros((len(all_results), len(grid_q)))

    for i, res in enumerate(all_results):
        qs = [r.total_queries for r in res.records]
        ef = [r.efficiency for r in res.records]

        if len(qs) < 2:
            continue

        pairs = sorted(zip(qs, ef), key=lambda t: t[0])

        uniq_q, uniq_e = [], []
        last_q = None

        for q, e in pairs:
            if last_q is None or q != last_q:
                uniq_q.append(q)
                uniq_e.append(e)
                last_q = q
            else:
                uniq_e[-1] = e

        mat[i] = np.interp(grid_q, uniq_q, uniq_e)

    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=0) / np.sqrt(len(all_results))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _avg_query_at_round(results, round_fn):
    qs = []
    for res in results:
        target = round_fn(res)
        if target is None:
            continue
        for rec in res.records:
            if rec.round_num == target:
                qs.append(rec.total_queries)
                break
    return np.mean(qs) if qs else None


def plot_single(result, save_path="plots/efficiency_single.png"):
    """Two-panel single-run plot. Left: efficiency vs cumulative queries.
    Right: efficiency vs auction round with phase markers."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    recs = result.records
    rounds = [r.round_num for r in recs]
    bids = [r.total_queries for r in recs]
    effs = [r.efficiency for r in recs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Efficiency over auction progress", fontsize=13)

    ax1.plot(bids, effs, lw=2, marker="o", markersize=3, color=_LINE_COLOR)
    ax1.set_xlabel("Cumulative queries (DQ + VQ)")
    ax1.set_ylabel("Efficiency (%)")
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("Efficiency vs elicited bids")

    ax2.plot(rounds, effs, lw=2, marker="o", markersize=3, color=_LINE_COLOR)

    if result.ml_dq_start_round is not None:
        ax2.axvline(
            result.ml_dq_start_round,
            lw=1.4, linestyle="--", color=_ML_DQ_COLOR,
            label=f"Start ML-DQ r={result.ml_dq_start_round}",
        )

    if result.bridge_round is not None:
        ax2.axvline(
            result.bridge_round,
            lw=1.4, linestyle="--", color=_BRIDGE_COLOR,
            label=f"Bridge r={result.bridge_round}",
        )

    if result.ml_vq_start_round is not None:
        ax2.axvline(
            result.ml_vq_start_round,
            lw=1.4, linestyle=":", color=_ML_VQ_COLOR,
            label=f"Start ML-VQ r={result.ml_vq_start_round}",
        )

    ax2.set_xlabel("Auction round")
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Efficiency vs auction round")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved → {save_path}")


def plot_multirun(results, save_path="plots/efficiency_multirun.png", max_q=165):
    """Multi-run efficiency curve in the project's styled format."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_max_q = [r.records[-1].total_queries for r in results if r.records]

    if not all_max_q:
        print("No records to plot.")
        return

    max_q = max(all_max_q)
    grid_q = np.linspace(0, max_q, 300)

    mean, lo, hi = _interpolate_runs(results, grid_q)

    avg_ml_dq_q = _avg_query_at_round(results, lambda r: r.ml_dq_start_round)
    avg_br_q = _avg_query_at_round(results, lambda r: r.bridge_round)
    avg_vq_q = _avg_query_at_round(results, lambda r: r.ml_vq_start_round)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.fill_between(grid_q, lo, hi, color=_FILL_COLOR, alpha=0.18)
    ax.plot(
        grid_q, mean,
        color=_LINE_COLOR, linewidth=2.5,
        label="Efficiency curve",
    )

    if avg_ml_dq_q is not None:
        ax.axvline(
            avg_ml_dq_q,
            color=_ML_DQ_COLOR, linestyle="--", linewidth=1.5,
            label="Start of ML DQ Rounds",
        )

    if avg_br_q is not None:
        ax.axvline(
            avg_br_q,
            color=_BRIDGE_COLOR, linestyle="--", linewidth=1.5,
            label="Bridge Bid Starts",
        )

    if avg_vq_q is not None:
        ax.axvline(
            avg_vq_q,
            color=_ML_VQ_COLOR, linestyle=":", linewidth=1.8,
            label="Start of ML VQ Rounds",
        )

    ax.set_xlabel("Number of Elicited Bids", fontsize=14)
    ax.set_ylabel("Efficiency (%)", fontsize=14)
    ax.set_xlim(0, max_q)
    ax.set_ylim(0, 102)

    ax.set_xticks(np.arange(0, max_q + 1, 15))
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved → {save_path}")

def _run_one(bidders, args, use_bridge, show_progress=False):
    if args.cca:
        return run_cca_only(bidders, show_progress=show_progress)
    return run_mlhca(bidders, show_progress=show_progress, use_bridge=use_bridge)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED_BASE,
        help="Random seed for single run",
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run NUM_RUNS instances and plot average efficiency",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_TYPE,
        choices=["linear", "pairwise", "mlp", "mvnn", "subset"],
        help="ML model type",
    )

    parser.add_argument(
        "--valuation",
        type=str,
        default=config.VALUATION_TYPE,
        choices=["sats", "additive", "pairwise", "complementarity", "mixed"],
        help="Valuation/data type",
    )

    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Disable bridge bid phase",
    )

    parser.add_argument(
        "--cca",
        action="store_true",
        help="Run CCA-only baseline (no ML phases)",
    )
    parser.add_argument(
        "--cca-rounds",
        type=int,
        default=None,
        help="Override CCA_ROUNDS (only used with --cca)",
    )

    args = parser.parse_args()



    config.MODEL_TYPE = args.model
    config.VALUATION_TYPE = args.valuation
    
    if args.cca_rounds is not None:
        config.CCA_ROUNDS = args.cca_rounds

    use_bridge = not args.no_bridge

    bridge_label = "bridge" if use_bridge else "nobridge"

    mechanism = "cca" if args.cca else f"{args.model}_{bridge_label}"
    label = f"{config.VALUATION_TYPE}_{mechanism}"

    if args.multi:
        print(
            f"Auction Efficiency over Elicited Bids | "
            f"{config.NUM_RUNS} seeds | "
            f"{config.N_BIDDERS} bidders, {config.N_ITEMS} items | "
            f"valuation={config.VALUATION_TYPE}, model={args.model}, "
            f"bridge={use_bridge}\n"
        )

        results = []

        for run_idx in range(config.NUM_RUNS):
            seed = config.SEED_BASE + run_idx

            print(f"  Run {run_idx + 1}/{config.NUM_RUNS} | seed={seed}", flush=True)

            bidders = make_bidders(seed=seed)

            result = _run_one(bidders, args, use_bridge)


            results.append(result)

            print(
                f"    efficiency={result.efficiency:.1f}% | "
                f"queries={result.total_queries}"
            )

        effs = [r.efficiency for r in results]
        queries = [r.total_queries for r in results]

        print("\n" + "=" * 60)
        print("MULTI-RUN SUMMARY")
        print("=" * 60)
        print(f"  Mean efficiency : {np.mean(effs):.2f}%")
        print(f"  Std efficiency  : {np.std(effs):.2f}%")
        print(f"  Mean queries    : {np.mean(queries):.2f}")

        plot_multirun(
            results,
            save_path=f"plots/efficiency_multirun_{label}.png",
        )

        save_multirun_results(
            results,
            valuation=config.VALUATION_TYPE,
            model=args.model,
            use_bridge=use_bridge,
        )

        save_multirun_records(
            results,
            valuation=config.VALUATION_TYPE,
            model=args.model,
            use_bridge=use_bridge,
        )

        save_multirun_quality(
            results,
            valuation=config.VALUATION_TYPE,
            model=args.model,
            use_bridge=use_bridge,
        )

    else:
        print(
            f"Auction Efficiency over Elicited Bids | "
            f"seed={args.seed} | "
            f"{config.N_BIDDERS} bidders, {config.N_ITEMS} items | "
            f"valuation={config.VALUATION_TYPE}, model={args.model}, "
            f"bridge={use_bridge}"
        )

        bidders = make_bidders(seed=args.seed)

        result = _run_one(bidders, args, use_bridge, show_progress=True)


        print_allocation(result.allocation, bidders, "FINAL ALLOCATION")
        print_summary(result)

        plot_single(
            result,
            save_path=f"plots/efficiency_single_{label}.png",
        )


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import config
from bidder import Bidder
from wdp import wdp_inferred, wdp_ml, wdp_true, wdp_true_exact
from model_evaluation import snapshot_quality


@dataclass
class EfficiencyRecord:
    round_num: int
    total_queries: int
    efficiency: float


@dataclass
class AuctionResult:
    allocation: tuple
    true_welfare: float
    opt_welfare: float
    efficiency: float
    total_queries: int
    n_dq: int
    n_vq: int
    records: List[EfficiencyRecord] = field(default_factory=list)
    dq_end_round: int = 0
    bridge_round: int = 0
    ml_dq_start_round: Optional[int] = None
    ml_vq_start_round: Optional[int] = None
    quality_snapshots: list = field(default_factory=list)


def _count_demand(demand_bundles, m):
    counts = [0] * m
    for bundle in demand_bundles:
        for j in bundle:
            counts[j] += 1
    return counts


def _no_excess_demand(demand_bundles, m):
    counts = _count_demand(demand_bundles, m)
    return all(counts[j] <= config.SUPPLY[j] for j in range(m))


def _market_clears(demand_bundles, m):
    counts = _count_demand(demand_bundles, m)
    return all(counts[j] == config.SUPPLY[j] for j in range(m))


def _update_prices(prices, demand_bundles, m):
    new_prices = prices.copy()
    counts = _count_demand(demand_bundles, m)
    for j in range(m):
        if counts[j] > config.SUPPLY[j]:
            new_prices[j] += config.PRICE_STEP
    return new_prices


def _total_queries(bidders):
    return sum(b.n_queries for b in bidders)


def _true_welfare(bidders, allocation):
    return sum(bidders[i].value(allocation[i]) for i in range(len(bidders)))


def _record(records, bidders, opt, round_num):
    alloc, _ = wdp_inferred(bidders)
    welfare = _true_welfare(bidders, alloc)
    eff = 100.0 * welfare / opt if opt > 0 else 0.0

    if eff > 100.0 + 1e-6:
        print(
            f"[WARNING] Efficiency > 100% at round {round_num}: "
            f"welfare={welfare:.4f}, benchmark={opt:.4f}. "
            f"Benchmark may be approximate."
        )

    records.append(EfficiencyRecord(
        round_num=round_num,
        total_queries=_total_queries(bidders),
        efficiency=eff,
    ))
    return eff


def _maybe_snapshot(bidders, label, quality_snapshots):
    """Run a quality snapshot if enabled in config; append to the list."""
    if not getattr(config, "QUALITY_SNAPSHOTS", True):
        return
    n_test = getattr(config, "QUALITY_N_TEST", 200)
    rows = snapshot_quality(bidders, label, n_test=n_test)
    quality_snapshots.append({"label": label, "rows": rows})


# NEXTPRICE 
def nextprice(bidders, m, prices_init, n_steps=30, lr=0.3, mu=0.5):
    prices = list(prices_init)

    for _ in range(n_steps):
        demand = []
        for bidder in bidders:
            best_bundle = ()
            best_util = 0.0
            for bundle in bidder._demand_candidates():
                u = bidder.predict_value(bundle) - sum(prices[j] for j in bundle)
                if u > best_util:
                    best_util = u
                    best_bundle = bundle
            demand.append(best_bundle)

        excess = [0.0] * m
        for bundle in demand:
            for j in bundle:
                excess[j] -= 1
        for j in range(m):
            excess[j] += config.SUPPLY[j]

        new_prices = list(prices)
        for j in range(m):
            step = lr * (1 + mu) if excess[j] < 0 else lr
            new_prices[j] = max(0.0, prices[j] - step * excess[j])
        prices = new_prices

    return prices


# Phase 1: CCA 
def run_cca_phase(bidders, m, opt, records, show_progress=False):
    prices = [0.0] * m
    market_clears_alloc = None

    for r in range(1, config.CCA_ROUNDS + 1):
        demand = [b.demand_query(prices, use_model=False) for b in bidders]
        _record(records, bidders, opt, r)
        print(f"[CCA] round {r} recorded")

        if show_progress:
            current_eff = records[-1].efficiency
            max_price = max(prices)
            print(f"Round {r}: max price = {max_price:.2f}, "
                  f"efficiency = {current_eff:.1f}%")

        #if _market_clears(demand, m):
         #   market_clears_alloc = tuple(demand)
          #  if show_progress:
           #     print(f" Market cleared at CCA round {r}")
           # return market_clears_alloc, r, prices

        prices = _update_prices(prices, demand, m)

    return None, config.CCA_ROUNDS, prices


# Phase 2: ML-powered DQs 
def run_ml_dq_phase(bidders, m, opt, records, start_round,
                    prices_init, show_progress=False):
    prices = list(prices_init)
    market_clears_alloc = None

    for r in range(1, config.ML_DQ_ROUNDS + 1):
        cur_round = start_round + r
        print(f"[ML-DQ] round {r}: training...", flush=True)

        for i, bidder in enumerate(bidders):
            bidder.train_model(epochs=config.TRAIN_EPOCHS, lr=config.TRAIN_LR)

        print(f"[ML-DQ] round {r}: nextprice...", flush=True)
        prices = nextprice(
            bidders, m, prices,
            n_steps=config.NEXTPRICE_STEPS,
            lr=config.NEXTPRICE_LR,
            mu=config.NEXTPRICE_MU,
        )

        demand = [b.demand_query(prices, use_model=True) for b in bidders]
        _record(records, bidders, opt, cur_round)

        if show_progress:
            current_eff = records[-1].efficiency
            max_price = max(prices)
            print(f"Round {cur_round}: max price = {max_price:.2f}, "
                  f"efficiency = {current_eff:.1f}%")

        if _market_clears(demand, m):
            market_clears_alloc = tuple(demand)
            if show_progress:
                print(f" Market cleared at ML-DQ round {cur_round}")
            return market_clears_alloc, cur_round

    return None, start_round + config.ML_DQ_ROUNDS


# Phase 3: Bridge bid 
def run_bridge_phase(bidders, opt, records, dq_end_round,
                     market_clears_alloc, show_progress=False):
    """Each bidder reports her TRUE value for the bundle she would
    receive according to the WDP after the last DQ phase."""
    if market_clears_alloc is not None:
        tentative = market_clears_alloc
    else:
        tentative, _ = wdp_inferred(bidders)

    bridge_round = dq_end_round + 1
    for i, bidder in enumerate(bidders):
        bundle = tentative[i]

        if not bundle:
            for b, _ in reversed(bidder.R_DQ):
                if b:
                    bundle = b
                    break
        print(f"  bridge VQ bidder {i}: bundle={bundle}, "
            f"value={bidder.value(bundle):.2f}")
        bidder.value_query(bundle)

    _record(records, bidders, opt, bridge_round)

    if show_progress:
        current_eff = records[-1].efficiency
        print(f"Bridge round {bridge_round}: efficiency = {current_eff:.1f}%")

    return bridge_round


def run_ml_vq_phase(bidders, opt, records, round_offset, show_progress=False):
    for r in range(1, config.ML_VQ_ROUNDS + 1):
        print(f"[ML-VQ] round {r}: training...", flush=True)

        for i, bidder in enumerate(bidders):
            bidder.train_model(epochs=config.TRAIN_EPOCHS, lr=config.TRAIN_LR)

        print(f"[ML-VQ] round {r}: solving wdp_ml...", flush=True)
        alloc, pred_w = wdp_ml(bidders)

        new_info = False
        for i, bidder in enumerate(bidders):
            before = len(bidder.R_VQ)
            bidder.value_query(alloc[i])
            if len(bidder.R_VQ) > before:
                new_info = True

        cur_round = round_offset + r

        if new_info:
            _record(records, bidders, opt, cur_round)
            print(f"[ML-VQ] round {r}: done, pred_w={pred_w:.2f}, "
                  f"eff={records[-1].efficiency:.1f}%", flush=True)
        else:
            print(f"[ML-VQ] round {r}: done, no new VQ info", flush=True)


# Running the mechanism
def run_mlhca(bidders, show_progress=False, use_bridge=True):
    m = config.N_ITEMS
    records = []
    quality_snapshots = []

    print("[1] Starting benchmark WDP...")

    use_exact = getattr(config, "USE_EXACT_OPT", False)
    if use_exact and config.N_ITEMS > 12:
        print(
            f"[1] WARNING: USE_EXACT_OPT=True is infeasible at N_ITEMS={config.N_ITEMS} "
            f"(2^{config.N_ITEMS} bundles per bidder). Falling back to approximate wdp_true.",
            flush=True,
        )
        use_exact = False

    if use_exact:
        _, opt = wdp_true_exact(bidders)
        print(f"[2] Finished exact wdp_true: opt={opt:.4f}", flush=True)
    else:
        _, opt = wdp_true(bidders)
        print(f"[2] Finished approximate wdp_true: opt={opt:.4f}", flush=True)

    # Origin point for plotting
    records.append(EfficiencyRecord(
        round_num=0,
        total_queries=0,
        efficiency=0.0,
    ))

    # Phase 1: CCA demand queries 
    market_clearing_demand, dq_end, last_prices = run_cca_phase(
        bidders, m, opt, records, show_progress
    )

    _maybe_snapshot(bidders, "after_CCA", quality_snapshots)

    ml_dq_start_round = None

    # Phase 2: ML-powered demand queries 
    if market_clearing_demand is None:
        ml_dq_start_round = dq_end + 1

        market_clearing_demand, dq_end = run_ml_dq_phase(
            bidders, m, opt, records,
            start_round=dq_end,
            prices_init=last_prices,
            show_progress=show_progress,
        )

        _maybe_snapshot(bidders, "after_ML_DQ", quality_snapshots)

    if market_clearing_demand is not None:
        final_alloc = market_clearing_demand
        final_welfare = _true_welfare(bidders, final_alloc)
        efficiency = 100.0 * final_welfare / opt if opt > 0 else 0.0

        if show_progress:
            print(f"Early stop: market-clearing demand found at round {dq_end}")

        return AuctionResult(
            allocation=final_alloc,
            true_welfare=final_welfare,
            opt_welfare=opt,
            efficiency=efficiency,
            total_queries=_total_queries(bidders),
            n_dq=sum(b.n_dq for b in bidders),
            n_vq=sum(b.n_vq for b in bidders),
            records=records,
            dq_end_round=dq_end,
            bridge_round=dq_end,
            ml_dq_start_round=ml_dq_start_round,
            ml_vq_start_round=None,
            quality_snapshots=quality_snapshots,
        )

    # Phase 3: Bridge bid
    if use_bridge:
        bridge_round = run_bridge_phase(
            bidders, opt, records, dq_end,
            market_clearing_demand, show_progress,
        )

        _maybe_snapshot(bidders, "after_bridge", quality_snapshots)
    else:
        bridge_round = dq_end

    # Phase 4: ML value queries
    ml_vq_start_round = bridge_round + 1
    run_ml_vq_phase(bidders, opt, records, bridge_round, show_progress)

   
    _maybe_snapshot(bidders, "final", quality_snapshots)

    # Final allocation using inferred welfare
    final_alloc, _ = wdp_inferred(bidders)
    final_welfare = _true_welfare(bidders, final_alloc)
    efficiency = 100.0 * final_welfare / opt if opt > 0 else 0.0

    return AuctionResult(
        allocation=final_alloc,
        true_welfare=final_welfare,
        opt_welfare=opt,
        efficiency=efficiency,
        total_queries=_total_queries(bidders),
        n_dq=sum(b.n_dq for b in bidders),
        n_vq=sum(b.n_vq for b in bidders),
        records=records,
        dq_end_round=dq_end,
        bridge_round=bridge_round,
        ml_dq_start_round=ml_dq_start_round,
        ml_vq_start_round=ml_vq_start_round,
        quality_snapshots=quality_snapshots,
    )

def run_cca_only(bidders, show_progress=False):
    m = config.N_ITEMS
    records = []
    quality_snapshots = []

    use_exact = getattr(config, "USE_EXACT_OPT", False)
    if use_exact and config.N_ITEMS > 12:
        use_exact = False

    if use_exact:
        _, opt = wdp_true_exact(bidders)
    else:
        _, opt = wdp_true(bidders)

    print(f"[CCA-only] opt={opt:.4f}", flush=True)

    records.append(EfficiencyRecord(round_num=0, total_queries=0, efficiency=0.0))

    _, dq_end, _ = run_cca_phase(bidders, m, opt, records, show_progress)

    final_alloc, _ = wdp_inferred(bidders)
    final_welfare = _true_welfare(bidders, final_alloc)
    efficiency = 100.0 * final_welfare / opt if opt > 0 else 0.0

    return AuctionResult(
        allocation=final_alloc,
        true_welfare=final_welfare,
        opt_welfare=opt,
        efficiency=efficiency,
        total_queries=_total_queries(bidders),
        n_dq=sum(b.n_dq for b in bidders),
        n_vq=sum(b.n_vq for b in bidders),
        records=records,
        dq_end_round=dq_end,
        bridge_round=dq_end,
        ml_dq_start_round=None,
        ml_vq_start_round=None,
        quality_snapshots=quality_snapshots,
    )
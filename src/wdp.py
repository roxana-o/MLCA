import itertools
from ortools.linear_solver import pywraplp
from utils import generate_all_bundles
import config

def _solve_ilp(bidders, candidate_fn, value_fn, label="WDP"):
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if solver is None:
        raise RuntimeError("SCIP solver is not available.")

    alloc_var = {}                                       
    bidder_vars = {i: [] for i in range(len(bidders))}
    item_vars = {j: [] for j in range(config.N_ITEMS)}

    for i, bidder in enumerate(bidders):
        for bundle in candidate_fn(bidder):
            bundle = tuple(sorted(bundle))
            value = value_fn(bidder, bundle)

            if value <= 1e-9:
                continue

            var = solver.BoolVar(f"x_{i}_{len(bidder_vars[i])}")
            alloc_var[i, bundle] = (var, value)
            bidder_vars[i].append(var)
            for item in bundle:
                item_vars[item].append(var)

    if not alloc_var:
        return tuple(() for _ in bidders), 0.0

    solver.Maximize(
        solver.Sum(value * var for var, value in alloc_var.values())
    )

    for i in range(len(bidders)):
        solver.Add(solver.Sum(bidder_vars[i]) <= 1)

    for item in range(config.N_ITEMS):
        solver.Add(solver.Sum(item_vars[item]) <= config.SUPPLY[item])

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return tuple(() for _ in bidders), 0.0

    allocation = [() for _ in bidders]
    welfare = 0.0

    for (i, bundle), (var, value) in alloc_var.items():
        if var.solution_value() > 0.5:
            allocation[i] = bundle
            welfare += value

    return tuple(allocation), welfare

# Bundles seen in queries — used by wdp_inferred (final allocation).
def _elicited_candidates(bidder):
    seen = {()}
    for bundle, _ in bidder.R_DQ:
        seen.add(tuple(sorted(bundle)))
    for bundle, _ in bidder.R_VQ:
        seen.add(tuple(sorted(bundle)))
    return list(seen)

#  Generate many bundles, keep top_k by true value used for the approximate wdp_true benchmark.
def _top_bundle_candidates(bidder, max_size=9, top_k=3000):
    cands = {()}
    for size in range(1, max_size + 1):
        for bundle in itertools.combinations(range(bidder.m), size):
            cands.add(bundle)
    for bundle, _ in bidder.R_DQ:
        cands.add(tuple(sorted(bundle)))
    for bundle, _ in bidder.R_VQ:
        cands.add(tuple(sorted(bundle)))

    scored = sorted(
        ((bidder.value(bundle), bundle) for bundle in cands),
        reverse=True,
    )
    return [bundle for _, bundle in scored[:top_k]]


# Original ML candidate set kept for reference
def _ml_candidates(bidder, max_size=6):
    cands = {()}
    for size in range(1, max_size + 1):
        for bundle in itertools.combinations(range(bidder.m), size):
            cands.add(bundle)
    for bundle, _ in bidder.R_DQ:
        cands.add(tuple(sorted(bundle)))
    for bundle, _ in bidder.R_VQ:
        cands.add(tuple(sorted(bundle)))
    return list(cands)


# ML candidate bundles for wdp_ml, excluding bundles already in R_VQ.
def _ml_candidates_excluding_vq(bidder, max_size=6):
    queried = {tuple(sorted(b)) for b, _ in bidder.R_VQ}

    cands = set()
    if () not in queried:
        cands.add(())

    for size in range(1, max_size + 1):
        for bundle in itertools.combinations(range(bidder.m), size):
            if bundle not in queried:
                cands.add(bundle)

    for bundle, _ in bidder.R_DQ:
        b = tuple(sorted(bundle))
        if b not in queried:
            cands.add(b)

    # Fallback if every bundle has been queried (unlikely)
    if not cands:
        cands = set(_ml_candidates(bidder, max_size=max_size))

    return list(cands)

# Approximate optimal-welfare benchmark using top_k candidates
def wdp_true(bidders, bundles=None):
    return _solve_ilp(
        bidders,
        candidate_fn=lambda b: _top_bundle_candidates(b, max_size=6, top_k=3000),
        value_fn=lambda b, bundle: b.value(bundle),
        label="WDP_true",
    )

# Final allocation using only inferred values from elicited reports
def wdp_inferred(bidders, bundles=None):
    return _solve_ilp(
        bidders,
        candidate_fn=_elicited_candidates,
        value_fn=lambda b, bundle: b.inferred_value(bundle),
        label="WDP_inferred",
    )

# ML-guided allocation for selecting the next VQ bundle.
def wdp_ml(bidders, bundles=None):
    max_size = 4 if config.N_ITEMS >= 15 else 6
    return _solve_ilp(
        bidders,
        candidate_fn=lambda b: _ml_candidates_excluding_vq(b, max_size=max_size),
        value_fn=lambda b, bundle: b.predict_value(bundle),
        label="WDP_ml",
    )

# Exact optimal-welfare benchmark (only feasible for small m)
def wdp_true_exact(bidders):
    return _solve_ilp(
        bidders,
        candidate_fn=lambda b: generate_all_bundles(b.m),
        value_fn=lambda b, bundle: b.value(bundle),
        label="WDP_true_exact",
    )

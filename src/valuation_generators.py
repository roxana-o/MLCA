import numpy as np
import itertools


# Creates additive valuation functions for each bidder: 
# each bidder gets random values for items and the value of a bundle is simply the sum of the values of the items it contains. 

def create_additive_valuations(n_bidders, m, seed=0):
    rng = np.random.default_rng(seed)

    valuations = []

    for i in range(n_bidders):
        # Assign values to items for this bidder
        item_values = rng.uniform(1, 10, size=m)

        # Define how this bidder values bundles
        def value_of_bundle(bundle, item_values=item_values):
            return float(sum(item_values[j] for j in bundle))

        valuations.append(value_of_bundle)

    return valuations


# Creates valuation functions with pairwise synergies:
# the value of a bundle is the sum of individual item values plus additional bonuses for certain pairs of items that are more valuable together. 

def create_synergy_valuations(n_bidders, m, seed=0):

    rng = np.random.default_rng(seed)
    valuations = []

    for i in range(n_bidders):
        item_values = rng.uniform(1, 8, size=m)

        synergy = rng.uniform(0, 5, size=(m, m))

        synergy = (synergy + synergy.T) / 2

        def value(bundle, item_values=item_values, synergy=synergy):

            total = sum(item_values[j] for j in bundle)

            for a, b in itertools.combinations(bundle, 2):
                total += synergy[a, b]

            return float(total)

        valuations.append(value)

    return valuations

# Creates valuation functions with complementarities: 
# each bidder has a value for each individual item, but they also have a special set of items that they value most. A bonus is received only if all items in the target set are included in the bundle.

def create_complementarity_valuations(n_bidders, m, seed=0):
    
    rng = np.random.default_rng(seed)

    weights = rng.uniform(1, 5, size=(n_bidders, m)).tolist()

    target_sets = []
    bonuses = []

    for i in range(n_bidders):
        target = rng.choice(m, size=4, replace=False)
        target_sets.append(frozenset(int(x) for x in target))
        bonuses.append(float(rng.uniform(20, 50)))

    valuations = []

    for i in range(n_bidders):
        w_i = weights[i]
        target_i = target_sets[i]
        bonus_i = bonuses[i]

        def val_fn(bundle, w_i=w_i, target_i=target_i, bonus_i=bonus_i):
            value = 0.0
            for j in bundle:
                value += w_i[j]
            if target_i.issubset(bundle):
                value += bonus_i
            return value

        valuations.append(val_fn)

    return valuations

# Creates valuation functions with strict complementarities (required items) and substitutes (groups where extra items add little or no value):
# each bidder has a required set of items, a bundle has value only if all required items are present, otherwise value = 0.
# required sets are drawn from a small contested core of items, guaranteeing overlap across bidders.

def create_mixed_valuations(n_bidders, m, seed=0,
                                 required_size=4,
                                 contested_core_size=10):

    rng = np.random.default_rng(seed)

    contested_core_size = min(contested_core_size, m)
    contested_core = rng.choice(m, size=contested_core_size, replace=False)
    contested_core = sorted(int(x) for x in contested_core)

    required_sets = []
    substitute_sets = []
    base_values = []
    extra_values = []

    for i in range(n_bidders):
        required = set(
            int(x) for x in rng.choice(contested_core,
                                       size=required_size,
                                       replace=False)
        )
        required_sets.append(required)

        remaining = list(set(range(m)) - required)
        sub_size = min(3, len(remaining))
        substitute = set(
            int(x) for x in rng.choice(remaining, size=sub_size, replace=False)
        )
        substitute_sets.append(substitute)

        base_values.append(float(rng.uniform(30, 60)))
        extra_values.append(rng.uniform(0.5, 3, size=m))

    valuations = []
    for i in range(n_bidders):
        req_i = required_sets[i]
        sub_i = substitute_sets[i]
        base_i = base_values[i]
        extras_i = extra_values[i]

        def value(bundle, req_i=req_i, sub_i=sub_i,
                  base_i=base_i, extras_i=extras_i):
            bundle_set = set(bundle)
            if not req_i.issubset(bundle_set):
                return 0.0

            total = base_i
            sub_items = bundle_set.intersection(sub_i)
            if sub_items:
                total += max(extras_i[j] for j in sub_items)

            other_items = bundle_set - req_i - sub_i
            total += sum(extras_i[j] for j in other_items)
            return float(total)

        valuations.append(value)
    return valuations
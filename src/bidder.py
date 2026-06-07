import itertools
import random
import torch
from ml_model import create_model, train_value_model
from utils import bundle_to_vector
import config


class Bidder:
    def __init__(self, bidder_id: int, valuation_fn, m: int):
        self.id      = bidder_id
        self.m       = m
        self._val_fn = valuation_fn
        self._value_cache = {}

        self.R_DQ: list = []   
        self.R_VQ: list = []  

        self.model = create_model(m)
    
        self._rng = random.Random(1000 + bidder_id)

    # True value
    def value(self, bundle) -> float:
        key = tuple(sorted(bundle))
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached
        v = float(self._val_fn(key))
        self._value_cache[key] = v
        return v

    # Demand query 
    def demand_query(self, prices: list, use_model: bool = False) -> tuple:
        
        best_bundle  = ()
        best_utility = 0.0   

        for bundle in self._demand_candidates():
            if use_model:
                v = self.predict_value(bundle)
            else:
                v = self.value(bundle)
            u = v - sum(prices[j] for j in bundle)
            if u > best_utility:
                best_utility = u
                best_bundle  = bundle

        self.R_DQ.append((tuple(best_bundle), list(prices)))
        return tuple(best_bundle)

    #  Candidate bundles for the demand-query argmax.

    def _demand_candidates(self):

        seen = set()

        seen.add(())

        for j in range(self.m):
            seen.add((j,))

        # Pairs
        for j in range(self.m):
            for k in range(j + 1, self.m):
                seen.add((j, k))

        # Triples
        for combo in itertools.combinations(range(self.m), 3):
            seen.add(combo)

        # Previously demanded bundles
        for b, _ in self.R_DQ:
            seen.add(tuple(sorted(b)))

        # Previously VQ-queried bundles
        for b, _ in self.R_VQ:
            seen.add(tuple(sorted(b)))

        # Subsets of previously demanded bundles up to size 5
        for b, _ in self.R_DQ:
            b = tuple(sorted(b))
            if len(b) <= 8:
                for sz in range(4, min(len(b) + 1, 6)):
                    for sub in itertools.combinations(b, sz):
                        seen.add(sub)

        # Model suggested high value bundles of size 4-6
        if self.R_DQ or self.R_VQ:
            n_samples = getattr(config, "MODEL_CANDIDATE_SAMPLES", 80)
            n_keep    = getattr(config, "MODEL_CANDIDATE_KEEP", 25)
            sizes     = getattr(config, "MODEL_CANDIDATE_SIZES", (4, 5, 6))

            sampled = set()
            for _ in range(n_samples):
                sz = self._rng.choice(sizes)
                sz = min(sz, self.m)
                bundle = tuple(sorted(self._rng.sample(range(self.m), sz)))
                sampled.add(bundle)

            scored = sorted(
                sampled,
                key=lambda b: -self.predict_value(b),
            )
            for b in scored[:n_keep]:
                seen.add(b)

        return list(seen)

    # Value query 
    def value_query(self, bundle) -> float:
        """Ask oracle for v(bundle); cache in R_VQ."""
        bundle = tuple(sorted(bundle))
        for b, v in self.R_VQ:
            if b == bundle:
                return v
        v = self.value(bundle)
        self.R_VQ.append((bundle, v))
        return v

    # Inferred value 
    def inferred_value(self, bundle) -> float:
        bundle = tuple(sorted(bundle))

        # exact VQ answer
        for b, v in self.R_VQ:
            if b == bundle:
                return v

        # max price dot-product over all DQ rounds
        if not self.R_DQ:
            return 0.0

        return max(
            sum(prices[j] for j in bundle)
            for _, prices in self.R_DQ
        )

    # ML prediction
    def predict_value(self, bundle) -> float:
        bundle = tuple(sorted(bundle))
        x = bundle_to_vector(bundle, self.m).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(x).item()
            scale = getattr(self.model, '_value_scale', 1.0)
            return max(0.0, raw * scale)

    # Train model
    def train_model(self, epochs: int = None, lr: float = None):
        epochs = epochs or config.TRAIN_EPOCHS
        lr     = lr     or config.TRAIN_LR
        self.model = train_value_model(self.model, self, epochs=epochs, lr=lr)

    # Counts
    @property
    def n_dq(self): return len(self.R_DQ)

    @property
    def n_vq(self): return len(self.R_VQ)

    @property
    def n_queries(self): return self.n_dq + self.n_vq

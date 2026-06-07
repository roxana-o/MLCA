import itertools
import torch

def generate_all_bundles(m: int):
    bundles = []
    for r in range(m + 1):
        for combo in itertools.combinations(range(m), r):
            bundles.append(tuple(combo))
    return bundles

def bundle_to_vector(bundle, m: int) -> torch.Tensor:
    x = torch.zeros(m, dtype=torch.float32)
    for j in bundle:
        x[j] = 1.0
    return x


def format_bundle(bundle) -> str:
    if not bundle:
        return "{}"
    return "{" + ", ".join(str(j + 1) for j in bundle) + "}"


def format_prices(prices) -> str:
    return ", ".join(f"p{j+1}={p:.2f}" for j, p in enumerate(prices))

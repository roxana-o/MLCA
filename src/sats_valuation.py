import os
import jnius_config
from jnius import autoclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
jar_path = ROOT / "lib" / "sats-0.8.1.jar"

if not os.path.exists(jar_path):
    raise FileNotFoundError(f"SATS jar not found: {jar_path}")

jnius_config.set_classpath(jar_path)


# Load Java classes 
LocalSynergyValueModel = autoclass(
    "org.spectrumauctions.sats.core.model.lsvm.LocalSynergyValueModel"
)
RNGSupplier = autoclass(
    "org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier"
)
Bundle = autoclass(
    "org.marketdesignresearch.mechlib.core.Bundle"
)
BundleEntry = autoclass(
    "org.marketdesignresearch.mechlib.core.BundleEntry"
)
HashSet = autoclass("java.util.HashSet")


# Valuation wrapper 

class SATSLSVMValuation:
    def __init__(self, bidder, licenses):
        self.bidder   = bidder
        self.licenses = licenses
        self._cache   = {}

    def __call__(self, bundle):
        key = tuple(sorted(bundle))
        if key not in self._cache:
            sats_bundle      = self._to_bundle(key)
            value            = self.bidder.calculateValue(sats_bundle)
            self._cache[key] = float(value.doubleValue())
        return self._cache[key]

    def value(self, bundle):
        return self(bundle)

    def _to_bundle(self, bundle):
        entries = HashSet()
        for j in bundle:
            lic   = self.licenses[j]
            entry = BundleEntry(lic, 1)
            entries.add(entry)
        return Bundle(entries)


# SATS MIP optimal welfare 
def get_sats_optimal_welfare(world, java_bidders):

    try:
        LSVMStandardMIP = autoclass(
            "org.spectrumauctions.sats.opt.model.lsvm.LSVMStandardMIP"
        )
        mip    = LSVMStandardMIP(world, java_bidders)
        result = mip.calculateAllocation()
        return float(result.getTotalValue().doubleValue())
    except Exception as e:
        print(f"  [SATS MIP] Not available or failed: {e}")
        return None


def create_sats_lsvm_valuations(seed=101):

    model = LocalSynergyValueModel()
    model.setNumberOfRegionalBidders(5)
    model.setNumberOfNationalBidders(1)

    rng      = RNGSupplier(seed)
    world    = model.createWorld(rng)
    bidders  = model.createPopulation(world, rng)
    licenses = list(world.getLicenses())

    valuations = [
        SATSLSVMValuation(b, licenses)
        for b in bidders
    ]

    java_bidders = list(bidders)

    return valuations, world, java_bidders
"""
Regime Detection Models
========================
Detects latent market regimes using:
- Gaussian Hidden Markov Models (HMM)
- HDBSCAN clustering
- Gaussian Mixture Models (GMM)
"""

__all__ = [
    "HMMRegimeDetector",
    "HDBSCANRegimeDetector",
    "GMMRegimeDetector",
]

def __getattr__(name):
    if name == "HMMRegimeDetector":
        from src.models.regime.hmm import HMMRegimeDetector
        return HMMRegimeDetector
    elif name == "HDBSCANRegimeDetector":
        from src.models.regime.clustering import HDBSCANRegimeDetector
        return HDBSCANRegimeDetector
    elif name == "GMMRegimeDetector":
        from src.models.regime.clustering import GMMRegimeDetector
        return GMMRegimeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

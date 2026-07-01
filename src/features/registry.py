"""
Feature Registry
================
Central registry for all feature computation functions. Uses a decorator
pattern to register features with metadata, enabling automatic discovery,
documentation, and selective computation.

Usage:
    from src.features.registry import feature_registry, register_feature

    @register_feature("rsi_14", group="technical", description="RSI with period 14")
    def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
        return ta.rsi(df["close"], length=14)

    # Compute all registered features
    features_df = feature_registry.compute_all(ohlcv_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from src.core.logger import get_logger
from src.core.types import FeatureDefinition

logger = get_logger("features.registry")

# Type alias for feature functions
FeatureFunc = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass
class RegisteredFeature:
    """A registered feature with its computation function and metadata."""
    name: str
    func: FeatureFunc
    group: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=lambda: ["close"])


class FeatureRegistry:
    """
    Central registry for feature computation functions.

    Features are registered via the @register_feature decorator
    and can be computed individually or in bulk.
    """

    def __init__(self) -> None:
        self._features: dict[str, RegisteredFeature] = {}

    def register(
        self,
        name: str,
        func: FeatureFunc,
        group: str = "custom",
        description: str = "",
        params: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
    ) -> None:
        """Register a feature computation function."""
        self._features[name] = RegisteredFeature(
            name=name,
            func=func,
            group=group,
            description=description,
            params=params or {},
            dependencies=dependencies or ["close"],
        )

    def get(self, name: str) -> RegisteredFeature | None:
        """Get a registered feature by name."""
        return self._features.get(name)

    def list_features(self, group: str | None = None) -> list[str]:
        """List all registered feature names, optionally filtered by group."""
        if group:
            return [
                name for name, feat in self._features.items()
                if feat.group == group
            ]
        return list(self._features.keys())

    def list_groups(self) -> list[str]:
        """List all registered feature groups."""
        return sorted(set(f.group for f in self._features.values()))

    def get_definitions(self) -> list[FeatureDefinition]:
        """Get FeatureDefinition objects for all registered features."""
        return [
            FeatureDefinition(
                name=feat.name,
                group=feat.group,
                description=feat.description,
                params=feat.params,
                dependencies=feat.dependencies,
            )
            for feat in self._features.values()
        ]

    def compute(self, name: str, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Compute a single feature."""
        feat = self._features.get(name)
        if feat is None:
            raise KeyError(f"Feature '{name}' not registered")

        try:
            result = feat.func(df)
            return result
        except Exception as e:
            logger.warning(
                "feature_computation_error",
                feature=name,
                error=str(e),
            )
            # Return NaN series on error
            return pd.Series(float("nan"), index=df.index, name=name)

    def compute_group(self, group: str, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features in a group."""
        feature_names = self.list_features(group=group)
        return self.compute_many(feature_names, df)

    def compute_many(
        self, names: list[str], df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute multiple features and return as a DataFrame."""
        results: dict[str, pd.Series] = {}

        for name in names:
            result = self.compute(name, df)

            if isinstance(result, pd.DataFrame):
                # Feature function returned multiple columns
                for col in result.columns:
                    results[col] = result[col]
            else:
                results[name] = result

        return pd.DataFrame(results, index=df.index)

    def compute_all(
        self,
        df: pd.DataFrame,
        groups: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute all registered features.

        Args:
            df: OHLCV DataFrame.
            groups: Only compute features in these groups.
            exclude: Skip these feature names.

        Returns:
            DataFrame with all computed features.
        """
        exclude = exclude or []

        names = []
        for name, feat in self._features.items():
            if name in exclude:
                continue
            if groups and feat.group not in groups:
                continue
            names.append(name)

        logger.info(
            "computing_features",
            total=len(names),
            groups=groups or "all",
        )

        result = self.compute_many(names, df)

        logger.info(
            "features_computed",
            features=len(result.columns),
            samples=len(result),
            null_pct=f"{result.isnull().mean().mean():.1%}",
        )

        return result

    @property
    def count(self) -> int:
        return len(self._features)


# ============================================================
# Global Registry Singleton
# ============================================================

feature_registry = FeatureRegistry()


def register_feature(
    name: str,
    group: str = "custom",
    description: str = "",
    params: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> Callable:
    """
    Decorator to register a feature computation function.

    Usage:
        @register_feature("rsi_14", group="technical", description="RSI 14-period")
        def compute_rsi_14(df):
            return ta.rsi(df["close"], length=14)
    """
    def decorator(func: FeatureFunc) -> FeatureFunc:
        feature_registry.register(
            name=name,
            func=func,
            group=group,
            description=description,
            params=params,
            dependencies=dependencies,
        )
        return func
    return decorator

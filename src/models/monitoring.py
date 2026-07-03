"""
Model Monitoring
=================
Post-deployment model health checks:
- PSI drift detection (features and predictions)
- Probability calibration (Brier score + reliability bins)
- Per-regime performance breakdown
- Champion/challenger comparison

All functions are pure (DataFrames in, dicts/DataFrames out) so they can
run in the dashboard, CLI, or a scheduled job without extra wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("models.monitoring")

# Conventional PSI thresholds
PSI_STABLE = 0.10      # < 0.10: no significant change
PSI_MODERATE = 0.25    # 0.10-0.25: moderate shift, investigate; > 0.25: retrain


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index between a reference (training) distribution
    and a live distribution. Higher = more drift.
    """
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) < bins or len(actual) < bins:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:   # constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(expected, bins=edges)[0] / len(expected), 1e-6, None)
    a_pct = np.clip(np.histogram(actual, bins=edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def feature_drift(
    reference: pd.DataFrame, live: pd.DataFrame, bins: int = 10
) -> pd.DataFrame:
    """
    PSI per shared feature column, sorted worst-first, with a status label.

    Returns columns: feature, psi, status (stable | moderate | drifted).
    """
    shared = [c for c in reference.columns if c in live.columns]
    rows = []
    for col in shared:
        value = psi(reference[col], live[col], bins=bins)
        status = ("drifted" if value > PSI_MODERATE
                  else "moderate" if value > PSI_STABLE else "stable")
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    drifted = int((out["psi"] > PSI_MODERATE).sum()) if not out.empty else 0
    logger.info("feature_drift_computed", features=len(out), drifted=drifted)
    return out


def calibration_report(
    y_true: pd.Series, y_prob: pd.Series, bins: int = 10
) -> dict:
    """
    Probability calibration: Brier score plus a reliability table
    (predicted probability vs observed frequency per bin).
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if df.empty:
        return {"brier": None, "bins": pd.DataFrame()}
    brier = float(np.mean((df["p"] - df["y"]) ** 2))
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    table = df.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), count=("y", "size"),
    ).reset_index(drop=True)
    table["gap"] = (table["predicted"] - table["observed"]).abs()
    return {"brier": brier, "bins": table}


def regime_performance(
    returns: pd.Series, regimes: pd.Series, periods: int = 252
) -> pd.DataFrame:
    """
    Strategy performance broken down per regime label.

    Args:
        returns: strategy returns, datetime index.
        regimes: regime label per bar (same index).
    """
    joined = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
    rows = []
    for label, grp in joined.groupby("regime"):
        r = grp["ret"]
        sharpe = float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0
        rows.append({
            "regime": label,
            "bars": len(r),
            "total_return": float((1 + r).prod() - 1),
            "ann_return": float(r.mean() * periods),
            "sharpe": round(sharpe, 3),
            "win_rate": float((r > 0).mean()),
        })
    return pd.DataFrame(rows)


def champion_challenger(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    higher_is_better: tuple[str, ...] = ("sharpe_ratio", "total_return", "win_rate"),
    lower_is_better: tuple[str, ...] = ("max_drawdown",),
) -> dict:
    """
    Compare a challenger model/strategy against the current champion.
    Challenger is promoted only if it wins on a majority of metrics.
    """
    wins, comparisons = 0, []
    for key in higher_is_better:
        if key in champion_metrics and key in challenger_metrics:
            better = challenger_metrics[key] > champion_metrics[key]
            wins += better
            comparisons.append({"metric": key, "champion": champion_metrics[key],
                                "challenger": challenger_metrics[key],
                                "challenger_wins": better})
    for key in lower_is_better:
        if key in champion_metrics and key in challenger_metrics:
            better = abs(challenger_metrics[key]) < abs(champion_metrics[key])
            wins += better
            comparisons.append({"metric": key, "champion": champion_metrics[key],
                                "challenger": challenger_metrics[key],
                                "challenger_wins": better})
    promote = bool(comparisons) and wins > len(comparisons) / 2
    logger.info("champion_challenger", promote=promote, wins=wins, total=len(comparisons))
    return {"promote_challenger": promote, "wins": wins,
            "total": len(comparisons), "detail": pd.DataFrame(comparisons)}

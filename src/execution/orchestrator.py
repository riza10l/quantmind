"""
Paper-Trading Orchestrator
===========================
Automated single-cycle pipeline: stored data -> strategy signal ->
risk check -> paper order -> audit log. Each cycle is fully audited
in the trades_log table with the decision reason.

Live mode is deliberately blocked: it requires QUANTMIND_LIVE_APPROVED=yes
in the environment AND no kill-switch file present. Paper mode has no
such requirement.

Usage:
    orch = PaperOrchestrator(config, db)
    outcome = orch.run_cycle("BTC-USD", "1d", EMACrossStrategy(...))
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.core.config import AppConfig
from src.core.database import DatabaseManager
from src.core.logger import get_logger
from src.core.types import Side, SignalType
from src.execution.broker import BaseBroker, PaperBroker
from src.portfolio.risk_engine import RiskEngine, RiskLimits
from src.strategy.templates import BaseStrategy

logger = get_logger("execution.orchestrator")

KILL_SWITCH_FILE = "KILL_SWITCH"   # presence in project root halts live trading


@dataclass
class CycleOutcome:
    """Result of one orchestration cycle."""
    symbol: str
    signal: str
    action: str            # "order_filled" | "order_rejected" | "blocked" | "hold"
    reason: str = ""
    order_id: str | None = None
    fill_price: float | None = None
    quantity: float = 0.0


def assert_live_allowed(project_root: Path) -> None:
    """Guard for live execution: explicit approval + kill switch absent."""
    if (project_root / KILL_SWITCH_FILE).exists():
        raise RuntimeError(
            f"kill switch active: remove {KILL_SWITCH_FILE} file to re-enable live trading"
        )
    if os.getenv("QUANTMIND_LIVE_APPROVED", "").lower() != "yes":
        raise RuntimeError(
            "live execution requires QUANTMIND_LIVE_APPROVED=yes in the environment"
        )


class PaperOrchestrator:
    """
    Drives one full research-to-order cycle on paper. The broker is
    injectable for testing; defaults to PaperBroker.
    """

    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        broker: BaseBroker | None = None,
        position_pct: float = 0.1,   # fraction of equity per new position
    ) -> None:
        self.config = config
        self.db = db
        self.broker = broker or PaperBroker(
            initial_balance=config.backtest.initial_capital,
            commission_pct=config.backtest.commission_pct,
            slippage_pct=config.backtest.slippage_pct,
        )
        self.risk = RiskEngine(RiskLimits(
            max_position_pct=config.portfolio.max_position_pct,
        ))
        self.position_pct = position_pct

    def run_cycle(self, symbol: str, timeframe: str, strategy: BaseStrategy) -> CycleOutcome:
        """data -> features(strategy-internal) -> signal -> risk -> order -> audit."""
        df = self.db.query_ohlcv(symbol, timeframe)
        if len(df) < 50:
            return self._audit(CycleOutcome(
                symbol, "NONE", "blocked",
                reason=f"not enough data ({len(df)} bars)"), strategy)

        price = float(df["close"].iloc[-1])
        if isinstance(self.broker, PaperBroker):
            self.broker.update_price(symbol, price)

        sig = strategy.generate_signals(df).iloc[-1]
        sig = sig.value if isinstance(sig, SignalType) else sig
        has_position = any(p.symbol == symbol for p in self.broker.get_positions())

        if sig == SignalType.BUY.value and not has_position:
            side, qty = Side.BUY, self._size(price)
        elif sig == SignalType.SELL.value and has_position:
            pos = next(p for p in self.broker.get_positions() if p.symbol == symbol)
            side, qty = Side.SELL, pos.quantity
        else:
            return self._audit(CycleOutcome(symbol, str(sig), "hold",
                                            reason="signal matches current position"), strategy)

        # Risk gate (skips the sell-to-close path: reducing risk is always allowed)
        if side == Side.BUY:
            equity = getattr(self.broker, "equity", self.broker.get_balance())
            check = self.risk.check_order(
                qty, price, equity,
                recent_returns=df["close"].pct_change().dropna().tail(100),
            )
            if not check.approved:
                return self._audit(CycleOutcome(symbol, str(sig), "blocked",
                                                reason=check.reason), strategy)
            qty = check.adjusted_quantity

        order = self.broker.make_order(symbol, side, qty)
        order = self.broker.submit_order(order)
        outcome = CycleOutcome(
            symbol=symbol, signal=str(sig),
            action="order_filled" if order.filled_quantity else "order_rejected",
            reason=order.metadata.get("error", "") if hasattr(order, "metadata") else "",
            order_id=order.id, fill_price=order.filled_price, quantity=order.filled_quantity,
        )
        return self._audit(outcome, strategy, order=order)

    def _size(self, price: float) -> float:
        equity = getattr(self.broker, "equity", self.broker.get_balance())
        return (equity * self.position_pct) / price

    def _audit(self, outcome: CycleOutcome, strategy: BaseStrategy, order=None) -> CycleOutcome:
        """Every cycle decision lands in trades_log, filled or not."""
        self.db.log_trade({
            "trade_id": outcome.order_id or str(uuid.uuid4())[:12],
            "timestamp": datetime.now(UTC).replace(tzinfo=None),
            "symbol": outcome.symbol,
            "side": order.side.value if order else "none",
            "order_type": order.order_type.value if order else "none",
            "quantity": outcome.quantity,
            "entry_price": outcome.fill_price or 0.0,
            "commission": order.commission if order else 0.0,
            "latency_ms": order.latency_ms if order else None,
            "explanation": f"{strategy.name}: signal={outcome.signal} "
                           f"action={outcome.action} {outcome.reason}".strip(),
            "mode": "paper",
        })
        logger.info("cycle_complete", symbol=outcome.symbol, signal=outcome.signal,
                    action=outcome.action, reason=outcome.reason)
        return outcome

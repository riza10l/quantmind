"""
Circuit Breaker (Fase 4)
=========================
Auto-stops trading when risk limits are breached.

Partially implemented as it's critical for safety even in paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.events import EventTypes, event_bus
from src.core.logger import get_logger

logger = get_logger("portfolio.circuit_breaker")


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    is_triggered: bool = False
    triggered_at: datetime | None = None
    trigger_reason: str = ""
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    total_drawdown: float = 0.0


class CircuitBreaker:
    """
    Monitors risk metrics and halts trading when limits are breached.

    Checks:
    - Daily drawdown exceeds max_daily_drawdown_pct
    - Total drawdown exceeds max_total_drawdown_pct
    - Consecutive losses exceed max_consecutive_losses
    """

    def __init__(
        self,
        max_daily_drawdown_pct: float = 0.03,
        max_total_drawdown_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        enabled: bool = True,
    ) -> None:
        self.max_daily_dd = max_daily_drawdown_pct
        self.max_total_dd = max_total_drawdown_pct
        self.max_consec_losses = max_consecutive_losses
        self.enabled = enabled
        self.state = CircuitBreakerState()

    def check(
        self,
        daily_pnl_pct: float,
        total_drawdown_pct: float,
        last_trade_pnl: float | None = None,
    ) -> bool:
        """
        Check if circuit breaker should trigger.

        Args:
            daily_pnl_pct: Today's PnL as percentage.
            total_drawdown_pct: Current drawdown from peak.
            last_trade_pnl: PnL of the most recent trade (for consecutive loss tracking).

        Returns:
            True if trading should continue, False if halted.
        """
        if not self.enabled:
            return True

        if self.state.is_triggered:
            return False

        self.state.daily_pnl = daily_pnl_pct
        self.state.total_drawdown = total_drawdown_pct

        # Track consecutive losses
        if last_trade_pnl is not None:
            if last_trade_pnl < 0:
                self.state.consecutive_losses += 1
            else:
                self.state.consecutive_losses = 0

        # Check daily drawdown
        if abs(daily_pnl_pct) > self.max_daily_dd and daily_pnl_pct < 0:
            self._trigger(f"Daily drawdown {daily_pnl_pct:.2%} exceeds limit {self.max_daily_dd:.2%}")
            return False

        # Check total drawdown
        if abs(total_drawdown_pct) > self.max_total_dd:
            self._trigger(f"Total drawdown {total_drawdown_pct:.2%} exceeds limit {self.max_total_dd:.2%}")
            return False

        # Check consecutive losses
        if self.state.consecutive_losses >= self.max_consec_losses:
            self._trigger(f"Consecutive losses ({self.state.consecutive_losses}) exceeds limit ({self.max_consec_losses})")
            return False

        return True

    def _trigger(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        self.state.is_triggered = True
        self.state.triggered_at = datetime.utcnow()
        self.state.trigger_reason = reason

        logger.critical(
            "circuit_breaker_triggered",
            reason=reason,
            daily_pnl=f"{self.state.daily_pnl:.2%}",
            total_dd=f"{self.state.total_drawdown:.2%}",
            consecutive_losses=self.state.consecutive_losses,
        )

        event_bus.emit(
            EventTypes.CIRCUIT_BREAKER_TRIGGERED,
            {
                "reason": reason,
                "state": {
                    "daily_pnl": self.state.daily_pnl,
                    "total_drawdown": self.state.total_drawdown,
                    "consecutive_losses": self.state.consecutive_losses,
                },
            },
            source="portfolio.circuit_breaker",
        )

    def reset(self) -> None:
        """Reset the circuit breaker (manual override)."""
        logger.warning("circuit_breaker_reset")
        self.state = CircuitBreakerState()

"""
Live/Paper Trading Broker (Fase 4)
====================================
Abstract broker interface with:
- PaperBroker: local fill simulation with commission + slippage
- CCXTBroker: live/testnet exchange adapter (requires `pip install ccxt`)

Every fill is logged with the originating signal's XAI explanation
so trading decisions are auditable.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from src.core.logger import get_logger
from src.core.types import Order, OrderStatus, OrderType, Position, Side, Signal

logger = get_logger("execution.broker")


class BaseBroker(ABC):
    """Abstract broker interface shared by paper and live implementations."""

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order; returns it with updated status/fill info."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Free cash balance in quote currency."""

    def make_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        signal: Optional[Signal] = None,
    ) -> Order:
        """Convenience constructor for a pending order."""
        return Order(
            id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            signal=signal,
        )


class PaperBroker(BaseBroker):
    """
    Paper trading simulator: fills market orders immediately at the
    provided mark price with commission and slippage, tracks positions
    and cash locally.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        self.balance = initial_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.prices: dict[str, float] = {}

    def update_price(self, symbol: str, price: float) -> None:
        """Feed the latest mark price (from data pipeline or websocket)."""
        self.prices[symbol] = price
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = price
            direction = 1.0 if pos.side == Side.BUY else -1.0
            pos.unrealized_pnl = direction * (price - pos.entry_price) * pos.quantity

    def submit_order(self, order: Order) -> Order:
        start = time.perf_counter()
        order.status = OrderStatus.SUBMITTED
        self.orders[order.id] = order

        mark = self.prices.get(order.symbol) or order.price
        if mark is None:
            order.status = OrderStatus.REJECTED
            logger.warning("order_rejected_no_price", order_id=order.id, symbol=order.symbol)
            return order

        # Fill at mark price with adverse slippage
        slip = mark * self.slippage_pct
        fill = mark + slip if order.side == Side.BUY else mark - slip
        cost = order.quantity * fill
        commission = cost * self.commission_pct

        if order.side == Side.BUY and cost + commission > self.balance:
            order.status = OrderStatus.REJECTED
            logger.warning("order_rejected_insufficient_balance", order_id=order.id)
            return order

        self._apply_fill(order, fill, commission)
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill
        order.commission = commission
        order.latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "order_filled",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=fill,
            explanation=order.signal.explanation if order.signal else "",
        )
        return order

    def _apply_fill(self, order: Order, fill: float, commission: float) -> None:
        pos = self.positions.get(order.symbol)
        if order.side == Side.BUY:
            self.balance -= order.quantity * fill + commission
            if pos is None:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol, side=Side.BUY, quantity=order.quantity,
                    entry_price=fill, current_price=fill,
                    opened_at=order.timestamp,
                )
            else:
                total = pos.quantity + order.quantity
                pos.entry_price = (pos.cost_basis + order.quantity * fill) / total
                pos.quantity = total
        else:
            self.balance += order.quantity * fill - commission
            if pos is not None:
                sell_qty = min(order.quantity, pos.quantity)
                pos.realized_pnl += (fill - pos.entry_price) * sell_qty
                pos.quantity -= sell_qty
                if pos.quantity <= 1e-12:
                    del self.positions[order.symbol]
            # ponytail: naked shorts not tracked; add signed positions if shorting is needed

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    def get_balance(self) -> float:
        return self.balance

    @property
    def equity(self) -> float:
        return self.balance + sum(p.market_value for p in self.positions.values())


class CCXTBroker(BaseBroker):
    """
    Live/testnet broker via CCXT. Lazy-imports ccxt so the rest of the
    system works without it installed.

    Safety: defaults to testnet. Pass testnet=False explicitly (and knowingly)
    for real trading.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        quote_currency: str = "USDT",
    ) -> None:
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError("CCXTBroker requires ccxt: pip install ccxt") from exc

        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange = exchange_cls({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        self.quote_currency = quote_currency
        logger.info("ccxt_broker_init", exchange=exchange_id, testnet=testnet)

    def submit_order(self, order: Order) -> Order:
        start = time.perf_counter()
        order.status = OrderStatus.SUBMITTED
        try:
            params: dict[str, Any] = {}
            result = self.exchange.create_order(
                symbol=order.symbol,
                type=order.order_type.value if order.order_type != OrderType.STOP_LOSS else "market",
                side=order.side.value,
                amount=order.quantity,
                price=order.price,
                params=params,
            )
            order.metadata["exchange_order_id"] = result.get("id")
            order.filled_quantity = float(result.get("filled") or 0.0)
            order.filled_price = float(result.get("average") or result.get("price") or 0.0)
            fee = result.get("fee") or {}
            order.commission = float(fee.get("cost") or 0.0)
            status = result.get("status")
            order.status = {
                "closed": OrderStatus.FILLED,
                "open": OrderStatus.SUBMITTED,
                "canceled": OrderStatus.CANCELLED,
            }.get(status, OrderStatus.PARTIAL if order.filled_quantity > 0 else OrderStatus.SUBMITTED)
        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.metadata["error"] = str(exc)
            logger.error("live_order_failed", order_id=order.id, error=str(exc))
        order.latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "live_order_result",
            order_id=order.id,
            status=order.status.value,
            latency_ms=round(order.latency_ms, 1),
            explanation=order.signal.explanation if order.signal else "",
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.exchange.cancel_order(order_id)
            return True
        except Exception as exc:
            logger.error("cancel_failed", order_id=order_id, error=str(exc))
            return False

    def get_positions(self) -> list[Position]:
        balances = self.exchange.fetch_balance()
        positions = []
        for asset, amount in (balances.get("total") or {}).items():
            if asset == self.quote_currency or not amount:
                continue
            symbol = f"{asset}/{self.quote_currency}"
            try:
                price = float(self.exchange.fetch_ticker(symbol)["last"])
            except Exception:
                continue
            positions.append(Position(
                symbol=symbol, side=Side.BUY, quantity=float(amount),
                entry_price=price, current_price=price,
            ))
        return positions

    def get_balance(self) -> float:
        balances = self.exchange.fetch_balance()
        return float((balances.get("free") or {}).get(self.quote_currency) or 0.0)

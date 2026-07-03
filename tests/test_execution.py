"""Tests for the paper trading broker (Fase 4)."""

from __future__ import annotations

import pytest

from src.core.types import OrderStatus, Side
from src.execution.broker import PaperBroker


@pytest.fixture
def broker() -> PaperBroker:
    b = PaperBroker(initial_balance=10_000, commission_pct=0.001, slippage_pct=0.0)
    b.update_price("BTC/USDT", 100.0)
    return b


def test_market_buy_fills(broker):
    order = broker.make_order("BTC/USDT", Side.BUY, quantity=10)
    filled = broker.submit_order(order)
    assert filled.status == OrderStatus.FILLED
    assert filled.filled_price == 100.0
    assert broker.get_balance() == pytest.approx(10_000 - 1000 - 1.0)  # cost + 0.1% fee
    assert len(broker.get_positions()) == 1


def test_sell_realizes_pnl(broker):
    broker.submit_order(broker.make_order("BTC/USDT", Side.BUY, quantity=10))
    broker.update_price("BTC/USDT", 110.0)
    broker.submit_order(broker.make_order("BTC/USDT", Side.SELL, quantity=10))
    assert broker.get_positions() == []
    # 10_000 - 1000 - 1 (buy) + 1100 - 1.1 (sell) = 10_097.9
    assert broker.get_balance() == pytest.approx(10_097.9)


def test_rejects_insufficient_balance(broker):
    order = broker.submit_order(broker.make_order("BTC/USDT", Side.BUY, quantity=1000))
    assert order.status == OrderStatus.REJECTED
    assert broker.get_balance() == 10_000


def test_rejects_unknown_symbol(broker):
    order = broker.submit_order(broker.make_order("DOGE/USDT", Side.BUY, quantity=1))
    assert order.status == OrderStatus.REJECTED


def test_slippage_worsens_fill():
    b = PaperBroker(initial_balance=10_000, commission_pct=0.0, slippage_pct=0.01)
    b.update_price("BTC/USDT", 100.0)
    buy = b.submit_order(b.make_order("BTC/USDT", Side.BUY, quantity=1))
    assert buy.filled_price == pytest.approx(101.0)  # adverse fill


def test_averaging_into_position(broker):
    broker.submit_order(broker.make_order("BTC/USDT", Side.BUY, quantity=10))
    broker.update_price("BTC/USDT", 120.0)
    broker.submit_order(broker.make_order("BTC/USDT", Side.BUY, quantity=10))
    pos = broker.get_positions()[0]
    assert pos.quantity == 20
    assert pos.entry_price == pytest.approx(110.0)


def test_cancel_only_pending(broker):
    order = broker.submit_order(broker.make_order("BTC/USDT", Side.BUY, quantity=1))
    assert broker.cancel_order(order.id) is False  # already filled
    assert broker.cancel_order("nonexistent") is False


def test_equity_marks_to_market(broker):
    broker.submit_order(broker.make_order("BTC/USDT", Side.BUY, quantity=10))
    broker.update_price("BTC/USDT", 200.0)
    assert broker.equity == pytest.approx(broker.get_balance() + 2000.0)

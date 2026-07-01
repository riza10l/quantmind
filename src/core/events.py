"""
QuantMind Event Bus
===================
Simple in-process event bus for decoupled inter-module communication.
Modules publish events (e.g., "data updated") and other modules subscribe
to react accordingly.

In production, this can be swapped with Redis Pub/Sub or Kafka.

Usage:
    from src.core.events import event_bus, Event

    # Subscribe to an event
    @event_bus.on("data_updated")
    def handle_data_update(event: Event):
        print(f"New data for {event.data['symbol']}")

    # Publish an event
    event_bus.emit("data_updated", {"symbol": "BTC/USDT", "rows": 100})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.core.logger import get_logger

logger = get_logger("core.events")


@dataclass
class Event:
    """An event that flows through the event bus."""
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""


# Type alias for event handlers
EventHandler = Callable[[Event], None]


class EventBus:
    """
    Simple synchronous event bus for in-process pub/sub.

    Thread-safe for basic usage. For async workloads, consider
    using asyncio.Queue or Redis Pub/Sub.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._event_history: list[Event] = []
        self._max_history: int = 1000

    def on(self, event_type: str) -> Callable:
        """
        Decorator to subscribe a handler to an event type.

        Usage:
            @event_bus.on("data_updated")
            def handle(event: Event):
                ...
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler)
            return handler
        return decorator

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(
            "event_subscribed",
            event_type=event_type,
            handler=handler.__name__,
        )

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def emit(self, event_type: str, data: dict[str, Any] | None = None,
             source: str = "") -> None:
        """
        Publish an event to all subscribed handlers.

        Args:
            event_type: The type of event to emit.
            data: Event payload.
            source: The module that emitted the event.
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
        )

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        handlers = self._handlers.get(event_type, [])
        logger.debug(
            "event_emitted",
            event_type=event_type,
            handlers=len(handlers),
            source=source,
        )

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "event_handler_error",
                    event_type=event_type,
                    handler=handler.__name__,
                    error=str(e),
                )

    def get_history(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[Event]:
        """Get recent event history, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear(self) -> None:
        """Clear all handlers and history."""
        self._handlers.clear()
        self._event_history.clear()


# ============================================================
# Global Event Bus Singleton
# ============================================================

event_bus = EventBus()


# ============================================================
# Standard Event Types
# ============================================================

class EventTypes:
    """Constants for standard event types across the system."""
    # Data events
    DATA_DOWNLOADED = "data_downloaded"
    DATA_VALIDATED = "data_validated"
    DATA_STORED = "data_stored"

    # Feature events
    FEATURES_COMPUTED = "features_computed"
    FEATURES_SELECTED = "features_selected"
    FEATURES_STORED = "features_stored"

    # Model events
    MODEL_TRAINED = "model_trained"
    MODEL_EVALUATED = "model_evaluated"
    MODEL_REGISTERED = "model_registered"

    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_APPROVED = "signal_approved"

    # Trading events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_CLOSED = "trade_closed"

    # Risk events
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"

    # System events
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_ERROR = "pipeline_error"

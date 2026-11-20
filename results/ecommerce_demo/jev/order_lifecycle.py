"""
order_lifecycle.py
==================
E-commerce order lifecycle implementation following SWEBOK guidelines:

Design decisions (SWEBOK-driven):
- Separation of Concerns: validation, transition table, and business rules
  are isolated in distinct sections/helpers, each independently modifiable.
- Single Responsibility: each helper function does exactly one thing.
- Anticipate Change: the transition table (TRANSITIONS) is a plain dict —
  adding a new state or action requires only a new entry there plus,
  if needed, a new rule function. No existing code is touched.
- Encapsulation: Order exposes only the public interface; all internals
  (_state, _payment_method, _paid_on, _delivered_on, _last_event_on)
  are name-mangled or prefixed.
- Verifiability: every invariant is expressed as a small, testable function.
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------

class OrderState(Enum):
    CREATED   = "CREATED"
    PAID      = "PAID"
    PICKED    = "PICKED"
    SHIPPED   = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    REFUNDED  = "REFUNDED"


class PaymentMethod(Enum):
    CREDIT_CARD = "CREDIT_CARD"
    PIX         = "PIX"
    BOLETO      = "BOLETO"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class OrderError(Exception):
    """Base class for all order-related errors."""


class InvalidOrder(OrderError):
    """Invalid argument supplied to the order or any of its actions."""


class InvalidTransition(OrderError):
    """The requested action does not exist in the current state."""


class RuleViolation(OrderError):
    """The action exists in the current state but a business rule blocks it."""


# ---------------------------------------------------------------------------
# Argument validators  (Separation of Concerns — validation layer)
# ---------------------------------------------------------------------------

def _validate_order_id(order_id: object) -> None:
    if not isinstance(order_id, str) or not order_id:
        raise InvalidOrder("order_id must be a non-empty str.")


def _validate_total(total: object) -> None:
    if not isinstance(total, Decimal):
        raise InvalidOrder("total must be a Decimal (not float or int).")
    if total <= Decimal("0"):
        raise InvalidOrder("total must be greater than zero.")
    # At most two decimal places: quantize and compare
    try:
        quantized = total.quantize(Decimal("0.01"))
    except InvalidOperation:
        raise InvalidOrder("total has too many decimal places.")
    if quantized != total:
        raise InvalidOrder("total must have at most two decimal places.")


def _validate_date(value: object, name: str = "on") -> None:
    """Accept only datetime.date, reject datetime.datetime (which is a subclass)."""
    if isinstance(value, datetime.datetime) or not isinstance(value, datetime.date):
        raise InvalidOrder(f"{name} must be a datetime.date (not datetime or other type).")


def _validate_method(method: object) -> None:
    if not isinstance(method, PaymentMethod):
        raise InvalidOrder("method must be a PaymentMethod member.")


def _validate_not_before_last_event(on: datetime.date, last_event_on: datetime.date) -> None:
    if on < last_event_on:
        raise InvalidOrder(
            f"Action date {on} is before the last recorded event date {last_event_on}."
        )


# ---------------------------------------------------------------------------
# Business-rule predicates  (Separation of Concerns — rules layer)
# Each rule receives the Order instance and the action date; raises RuleViolation
# or returns normally.
# ---------------------------------------------------------------------------

def _rule_C(order: "Order", on: datetime.date) -> None:
    """Cancel a paid-but-not-yet-shipped order."""
    assert order._paid_on is not None  # invariant guaranteed by transition table
    days_since_payment = (on - order._paid_on).days

    if order._total > Decimal("5000.00"):
        limit = 1
    elif order._payment_method is PaymentMethod.BOLETO:
        limit = 2
    else:
        limit = 7

    if days_since_payment > limit:
        raise RuleViolation(
            f"Cannot cancel: {days_since_payment} day(s) since payment exceed the "
            f"{limit}-day limit for this order."
        )


def _rule_R(order: "Order", on: datetime.date) -> None:
    """Refund a delivered order."""
    assert order._delivered_on is not None  # invariant guaranteed by transition table

    if order._payment_method not in (PaymentMethod.CREDIT_CARD, PaymentMethod.PIX):
        raise RuleViolation(
            "Cannot refund: payment method must be CREDIT_CARD or PIX."
        )

    days_since_delivery = (on - order._delivered_on).days
    if days_since_delivery > 7:
        raise RuleViolation(
            f"Cannot refund: {days_since_delivery} day(s) since delivery exceed the 7-day limit."
        )

    if order._total > Decimal("2000.00"):
        raise RuleViolation(
            f"Cannot refund: order total {order._total} exceeds the 2000.00 limit."
        )


def _rule_R2(order: "Order", on: datetime.date) -> None:
    """Refund a cancelled order."""
    if order._paid_on is None:
        raise RuleViolation(
            "Cannot refund: order was cancelled before any payment was made."
        )

    if order._payment_method is PaymentMethod.BOLETO:
        days_since_payment = (on - order._paid_on).days
        if days_since_payment < 1:
            raise RuleViolation(
                "Cannot refund via BOLETO: at least 1 day must have passed since payment."
            )


# ---------------------------------------------------------------------------
# Transition table  (Separation of Concerns — state-machine layer)
#
# Structure:
#   TRANSITIONS[current_state][action_name] = (next_state, rule_fn | None)
#
# Adding a new state/action: add an entry here (and a rule function if needed).
# No other code changes required.
# ---------------------------------------------------------------------------

_RuleFn = Callable[["Order", datetime.date], None]

TRANSITIONS: dict[OrderState, dict[str, tuple[OrderState, Optional[_RuleFn]]]] = {
    OrderState.CREATED: {
        "pay":    (OrderState.PAID,      None),
        "cancel": (OrderState.CANCELLED, None),
    },
    OrderState.PAID: {
        "pick":   (OrderState.PICKED,    None),
        "cancel": (OrderState.CANCELLED, _rule_C),
    },
    OrderState.PICKED: {
        "ship":   (OrderState.SHIPPED,   None),
        "cancel": (OrderState.CANCELLED, _rule_C),
    },
    OrderState.SHIPPED: {
        "deliver": (OrderState.DELIVERED, None),
    },
    OrderState.DELIVERED: {
        "refund": (OrderState.REFUNDED,  _rule_R),
    },
    OrderState.CANCELLED: {
        "refund": (OrderState.REFUNDED,  _rule_R2),
    },
    OrderState.REFUNDED: {},   # terminal state — no outgoing transitions
}


# ---------------------------------------------------------------------------
# Order class
# ---------------------------------------------------------------------------

class Order:
    """
    Represents a single e-commerce order and manages its lifecycle.

    State is maintained internally; callers interact only through the
    public action methods.  Every method follows the mandated verification
    order: argument validation → transition existence → business rule.
    A failed call leaves the Order unchanged (strong exception safety).
    """

    def __init__(self, order_id: str, total: Decimal, created_on: datetime.date) -> None:
        # --- argument validation (raises InvalidOrder on failure) ---
        _validate_order_id(order_id)
        _validate_total(total)
        _validate_date(created_on, "created_on")

        self._order_id: str              = order_id
        self._total: Decimal             = total
        self._created_on: datetime.date  = created_on

        # Mutable lifecycle fields
        self._state: OrderState                    = OrderState.CREATED
        self._last_event_on: datetime.date         = created_on
        self._payment_method: Optional[PaymentMethod] = None
        self._paid_on: Optional[datetime.date]        = None
        self._delivered_on: Optional[datetime.date]   = None

    # ------------------------------------------------------------------
    # Public read-only property
    # ------------------------------------------------------------------

    @property
    def state(self) -> OrderState:
        return self._state

    # ------------------------------------------------------------------
    # Internal dispatcher  (DRY — single place for the verification order)
    # ------------------------------------------------------------------

    def _apply(self, action: str, on: datetime.date, **ctx) -> None:
        """
        Execute *action* on date *on*, honouring the mandated check order:
        1. Argument validation  → InvalidOrder
        2. Transition existence → InvalidTransition
        3. Business rule        → RuleViolation

        ctx carries action-specific extra data (e.g. payment method).
        Nothing is mutated until all checks pass (strong exception safety).
        """
        # 1. Argument validation
        _validate_date(on, "on")
        _validate_not_before_last_event(on, self._last_event_on)

        if action == "pay":
            _validate_method(ctx.get("method"))

        # 2. Transition existence
        state_transitions = TRANSITIONS.get(self._state, {})
        if action not in state_transitions:
            raise InvalidTransition(
                f"Action '{action}' is not allowed in state '{self._state.value}'."
            )

        next_state, rule_fn = state_transitions[action]

        # 3. Business rule
        if rule_fn is not None:
            rule_fn(self, on)

        # --- Commit (all checks passed) ---
        self._state = next_state
        self._last_event_on = on

        # Record action-specific timestamps / data
        if action == "pay":
            self._payment_method = ctx["method"]
            self._paid_on = on
        elif action == "deliver":
            self._delivered_on = on

    # ------------------------------------------------------------------
    # Public action methods
    # ------------------------------------------------------------------

    def pay(self, method: PaymentMethod, on: datetime.date) -> None:
        self._apply("pay", on, method=method)

    def pick(self, on: datetime.date) -> None:
        self._apply("pick", on)

    def ship(self, on: datetime.date) -> None:
        self._apply("ship", on)

    def deliver(self, on: datetime.date) -> None:
        self._apply("deliver", on)

    def cancel(self, on: datetime.date) -> None:
        self._apply("cancel", on)

    def refund(self, on: datetime.date) -> None:
        self._apply("refund", on)

    # ------------------------------------------------------------------
    # Repr for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Order(id={self._order_id!r}, state={self._state.value}, "
            f"total={self._total}, last_event={self._last_event_on})"
        )

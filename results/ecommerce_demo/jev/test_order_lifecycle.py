"""
test_order_lifecycle.py
=======================
Pytest suite for order_lifecycle.py.

Test-design strategy (SWEBOK-guided):
- State Transition Testing (FSM coverage): one positive test per transition.
- Negative / invalid-transition tests: every (state, action) pair NOT in the
  table triggers InvalidTransition.
- Boundary-value analysis on every business rule (at-limit, over-limit).
- Equivalence partitioning for argument validation (valid class, each invalid
  class).
- Scenario-based tests for complete happy paths and compound rule interactions.
- Strong exception-safety tests: a rejected call must leave Order unchanged.
"""

import datetime
from decimal import Decimal

import pytest

from order_lifecycle import (
    OrderError,
    InvalidOrder,
    InvalidTransition,
    Order,
    OrderState,
    PaymentMethod,
    RuleViolation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = Decimal          # shorthand
date = datetime.date # shorthand


def make_order(
    order_id: str = "ORD-001",
    total: Decimal = D("100.00"),
    created_on: datetime.date = date(2024, 1, 1),
) -> Order:
    return Order(order_id, total, created_on)


def paid_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    total: Decimal = D("100.00"),
    paid_on: datetime.date = date(2024, 1, 2),
) -> Order:
    o = make_order(total=total)
    o.pay(method, paid_on)
    return o


def picked_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    total: Decimal = D("100.00"),
    picked_on: datetime.date = date(2024, 1, 3),
) -> Order:
    o = paid_order(method=method, total=total)
    o.pick(picked_on)
    return o


def shipped_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    total: Decimal = D("100.00"),
) -> Order:
    o = picked_order(method=method, total=total)
    o.ship(date(2024, 1, 4))
    return o


def delivered_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    total: Decimal = D("100.00"),
    delivered_on: datetime.date = date(2024, 1, 5),
) -> Order:
    o = shipped_order(method=method, total=total)
    o.deliver(delivered_on)
    return o


# ===========================================================================
# 1. ARGUMENT VALIDATION (InvalidOrder)
# ===========================================================================

class TestOrderConstruction:
    """Validation of Order.__init__ arguments."""

    def test_valid_construction(self):
        o = make_order()
        assert o.state is OrderState.CREATED

    # --- order_id ---
    def test_order_id_empty_string(self):
        with pytest.raises(InvalidOrder):
            Order("", D("10.00"), date(2024, 1, 1))

    def test_order_id_not_string(self):
        with pytest.raises(InvalidOrder):
            Order(123, D("10.00"), date(2024, 1, 1))

    # --- total ---
    def test_total_float_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", 10.0, date(2024, 1, 1))

    def test_total_int_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", 10, date(2024, 1, 1))

    def test_total_zero_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", D("0.00"), date(2024, 1, 1))

    def test_total_negative_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", D("-1.00"), date(2024, 1, 1))

    def test_total_three_decimal_places_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", D("1.001"), date(2024, 1, 1))

    def test_total_two_decimal_places_accepted(self):
        o = Order("X", D("0.01"), date(2024, 1, 1))
        assert o.state is OrderState.CREATED

    def test_total_whole_number_decimal_accepted(self):
        o = Order("X", D("100"), date(2024, 1, 1))
        assert o.state is OrderState.CREATED

    def test_total_one_decimal_place_accepted(self):
        o = Order("X", D("9.9"), date(2024, 1, 1))
        assert o.state is OrderState.CREATED

    # --- created_on ---
    def test_created_on_datetime_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", D("10.00"), datetime.datetime(2024, 1, 1, 0, 0))

    def test_created_on_string_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("X", D("10.00"), "2024-01-01")


class TestActionDateValidation:
    """on= argument validation for action methods."""

    def test_on_datetime_rejected_in_pay(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.CREDIT_CARD, datetime.datetime(2024, 1, 2))

    def test_on_string_rejected_in_pick(self):
        o = paid_order()
        with pytest.raises(InvalidOrder):
            o.pick("2024-01-03")  # type: ignore[arg-type]

    def test_on_before_last_event_rejected(self):
        o = make_order(created_on=date(2024, 1, 10))
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.CREDIT_CARD, date(2024, 1, 9))

    def test_on_equal_to_last_event_accepted(self):
        o = make_order(created_on=date(2024, 1, 1))
        o.pay(PaymentMethod.CREDIT_CARD, date(2024, 1, 1))  # same day — OK
        assert o.state is OrderState.PAID

    def test_on_before_payment_rejected_in_pick(self):
        o = paid_order(paid_on=date(2024, 1, 5))
        with pytest.raises(InvalidOrder):
            o.pick(date(2024, 1, 4))

    def test_method_wrong_type_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay("CREDIT_CARD", date(2024, 1, 2))  # type: ignore[arg-type]

    def test_method_none_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(None, date(2024, 1, 2))  # type: ignore[arg-type]


# ===========================================================================
# 2. VALID TRANSITIONS (happy-path FSM coverage)
# ===========================================================================

class TestHappyPathTransitions:
    """One positive test per entry in the transition table."""

    def test_created_pay(self):
        o = make_order()
        o.pay(PaymentMethod.CREDIT_CARD, date(2024, 1, 2))
        assert o.state is OrderState.PAID

    def test_created_cancel(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        assert o.state is OrderState.CANCELLED

    def test_paid_pick(self):
        o = paid_order()
        o.pick(date(2024, 1, 3))
        assert o.state is OrderState.PICKED

    def test_paid_cancel_within_limit(self):
        o = paid_order(paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 7))  # 6 days — within 7-day limit
        assert o.state is OrderState.CANCELLED

    def test_picked_ship(self):
        o = picked_order()
        o.ship(date(2024, 1, 4))
        assert o.state is OrderState.SHIPPED

    def test_picked_cancel_within_limit(self):
        o = picked_order(picked_on=date(2024, 1, 3))
        # payment was date(2024,1,2); cancel at day 5 → 3 days since payment → within 7
        o.cancel(date(2024, 1, 5))
        assert o.state is OrderState.CANCELLED

    def test_shipped_deliver(self):
        o = shipped_order()
        o.deliver(date(2024, 1, 5))
        assert o.state is OrderState.DELIVERED

    def test_delivered_refund(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("500.00"))
        o.refund(date(2024, 1, 10))  # 5 days after delivery
        assert o.state is OrderState.REFUNDED

    def test_cancelled_refund_after_payment(self):
        # Pay then cancel immediately, then refund on same day
        o = paid_order(method=PaymentMethod.CREDIT_CARD, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        o.refund(date(2024, 1, 1))
        assert o.state is OrderState.REFUNDED

    def test_full_happy_path_credit_card(self):
        o = make_order()
        o.pay(PaymentMethod.CREDIT_CARD, date(2024, 1, 1))
        o.pick(date(2024, 1, 2))
        o.ship(date(2024, 1, 3))
        o.deliver(date(2024, 1, 4))
        assert o.state is OrderState.DELIVERED

    def test_full_happy_path_pix(self):
        o = Order("ORD-PIX", D("1999.99"), date(2024, 3, 1))
        o.pay(PaymentMethod.PIX, date(2024, 3, 2))
        o.pick(date(2024, 3, 3))
        o.ship(date(2024, 3, 4))
        o.deliver(date(2024, 3, 5))
        o.refund(date(2024, 3, 6))
        assert o.state is OrderState.REFUNDED


# ===========================================================================
# 3. INVALID TRANSITIONS (InvalidTransition — exhaustive coverage)
# ===========================================================================

class TestInvalidTransitions:
    """
    Every (state, action) combination NOT in the table must raise
    InvalidTransition.  Actions are: pay, pick, ship, deliver, cancel, refund.
    """

    # CREATED: only pay and cancel are valid
    def test_created_pick(self):
        with pytest.raises(InvalidTransition):
            make_order().pick(date(2024, 1, 2))

    def test_created_ship(self):
        with pytest.raises(InvalidTransition):
            make_order().ship(date(2024, 1, 2))

    def test_created_deliver(self):
        with pytest.raises(InvalidTransition):
            make_order().deliver(date(2024, 1, 2))

    def test_created_refund(self):
        with pytest.raises(InvalidTransition):
            make_order().refund(date(2024, 1, 2))

    # PAID: only pick and cancel are valid
    def test_paid_pay(self):
        with pytest.raises(InvalidTransition):
            paid_order().pay(PaymentMethod.PIX, date(2024, 1, 3))

    def test_paid_ship(self):
        with pytest.raises(InvalidTransition):
            paid_order().ship(date(2024, 1, 3))

    def test_paid_deliver(self):
        with pytest.raises(InvalidTransition):
            paid_order().deliver(date(2024, 1, 3))

    def test_paid_refund(self):
        with pytest.raises(InvalidTransition):
            paid_order().refund(date(2024, 1, 3))

    # PICKED: only ship and cancel are valid
    def test_picked_pay(self):
        with pytest.raises(InvalidTransition):
            picked_order().pay(PaymentMethod.PIX, date(2024, 1, 4))

    def test_picked_pick(self):
        with pytest.raises(InvalidTransition):
            picked_order().pick(date(2024, 1, 4))

    def test_picked_deliver(self):
        with pytest.raises(InvalidTransition):
            picked_order().deliver(date(2024, 1, 4))

    def test_picked_refund(self):
        with pytest.raises(InvalidTransition):
            picked_order().refund(date(2024, 1, 4))

    # SHIPPED: only deliver is valid
    def test_shipped_pay(self):
        with pytest.raises(InvalidTransition):
            shipped_order().pay(PaymentMethod.PIX, date(2024, 1, 5))

    def test_shipped_pick(self):
        with pytest.raises(InvalidTransition):
            shipped_order().pick(date(2024, 1, 5))

    def test_shipped_ship(self):
        with pytest.raises(InvalidTransition):
            shipped_order().ship(date(2024, 1, 5))

    def test_shipped_cancel(self):
        with pytest.raises(InvalidTransition):
            shipped_order().cancel(date(2024, 1, 5))

    def test_shipped_refund(self):
        with pytest.raises(InvalidTransition):
            shipped_order().refund(date(2024, 1, 5))

    # DELIVERED: only refund is valid
    def test_delivered_pay(self):
        with pytest.raises(InvalidTransition):
            delivered_order().pay(PaymentMethod.PIX, date(2024, 1, 6))

    def test_delivered_pick(self):
        with pytest.raises(InvalidTransition):
            delivered_order().pick(date(2024, 1, 6))

    def test_delivered_ship(self):
        with pytest.raises(InvalidTransition):
            delivered_order().ship(date(2024, 1, 6))

    def test_delivered_deliver(self):
        with pytest.raises(InvalidTransition):
            delivered_order().deliver(date(2024, 1, 6))

    def test_delivered_cancel(self):
        with pytest.raises(InvalidTransition):
            delivered_order().cancel(date(2024, 1, 6))

    # CANCELLED: only refund is valid (after payment)
    def test_cancelled_pay(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.CREDIT_CARD, date(2024, 1, 3))

    def test_cancelled_pick(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        with pytest.raises(InvalidTransition):
            o.pick(date(2024, 1, 3))

    def test_cancelled_ship(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        with pytest.raises(InvalidTransition):
            o.ship(date(2024, 1, 3))

    def test_cancelled_deliver(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        with pytest.raises(InvalidTransition):
            o.deliver(date(2024, 1, 3))

    def test_cancelled_cancel(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        with pytest.raises(InvalidTransition):
            o.cancel(date(2024, 1, 3))

    # REFUNDED: terminal — no actions at all
    def test_refunded_is_terminal_no_pay(self):
        o = delivered_order(method=PaymentMethod.PIX, total=D("100.00"))
        o.refund(date(2024, 1, 7))
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, date(2024, 1, 8))

    def test_refunded_is_terminal_no_cancel(self):
        o = delivered_order(method=PaymentMethod.PIX, total=D("100.00"))
        o.refund(date(2024, 1, 7))
        with pytest.raises(InvalidTransition):
            o.cancel(date(2024, 1, 8))

    def test_refunded_is_terminal_no_refund(self):
        o = delivered_order(method=PaymentMethod.PIX, total=D("100.00"))
        o.refund(date(2024, 1, 7))
        with pytest.raises(InvalidTransition):
            o.refund(date(2024, 1, 8))


# ===========================================================================
# 4. RULE C — Cancel paid/picked order (boundary-value analysis)
# ===========================================================================

class TestRuleC:
    """
    Boundaries:
      total > 5000  → limit 1 day  (0 OK, 1 OK, 2 fail)
      total ≤ 5000, BOLETO → limit 2 days (0,1,2 OK, 3 fail)
      total ≤ 5000, other  → limit 7 days (0..7 OK, 8 fail)
    """

    # --- total > 5000, limit = 1 ---

    def test_rule_c_high_value_day0_ok(self):
        o = paid_order(total=D("5000.01"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))   # 0 days — OK
        assert o.state is OrderState.CANCELLED

    def test_rule_c_high_value_day1_ok(self):
        o = paid_order(total=D("5000.01"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 2))   # 1 day — at limit
        assert o.state is OrderState.CANCELLED

    def test_rule_c_high_value_day2_fail(self):
        o = paid_order(total=D("5000.01"), paid_on=date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 3))   # 2 days — over limit

    def test_rule_c_exactly_5000_not_high_value(self):
        """5000.00 is NOT > 5000, so normal rules apply."""
        o = paid_order(total=D("5000.00"), method=PaymentMethod.CREDIT_CARD, paid_on=date(2024, 1, 1))
        # Should use 7-day limit → day 7 OK
        o.cancel(date(2024, 1, 8))   # 7 days
        assert o.state is OrderState.CANCELLED

    # --- total ≤ 5000, BOLETO, limit = 2 ---

    def test_rule_c_boleto_day0_ok(self):
        o = paid_order(method=PaymentMethod.BOLETO, total=D("100.00"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        assert o.state is OrderState.CANCELLED

    def test_rule_c_boleto_day2_ok(self):
        o = paid_order(method=PaymentMethod.BOLETO, total=D("100.00"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 3))   # 2 days — at limit
        assert o.state is OrderState.CANCELLED

    def test_rule_c_boleto_day3_fail(self):
        o = paid_order(method=PaymentMethod.BOLETO, total=D("100.00"), paid_on=date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 4))   # 3 days — over limit

    # --- total ≤ 5000, CREDIT_CARD / PIX, limit = 7 ---

    def test_rule_c_credit_card_day7_ok(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 8))   # 7 days — at limit
        assert o.state is OrderState.CANCELLED

    def test_rule_c_credit_card_day8_fail(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), paid_on=date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 9))   # 8 days — over limit

    def test_rule_c_pix_day7_ok(self):
        o = paid_order(method=PaymentMethod.PIX, total=D("100.00"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 8))   # 7 days — at limit
        assert o.state is OrderState.CANCELLED

    def test_rule_c_pix_day8_fail(self):
        o = paid_order(method=PaymentMethod.PIX, total=D("100.00"), paid_on=date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 9))   # 8 days — over limit

    # --- Rule C applies at PICKED state too ---

    def test_rule_c_in_picked_state_within_limit(self):
        # payment date is date(2024,1,2) (from paid_order default)
        o = picked_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), picked_on=date(2024, 1, 3))
        o.cancel(date(2024, 1, 9))   # 7 days since payment (1/2 → 1/9) — at limit
        assert o.state is OrderState.CANCELLED

    def test_rule_c_in_picked_state_over_limit(self):
        o = picked_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), picked_on=date(2024, 1, 3))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 10))   # 8 days since payment

    def test_rule_c_high_value_picked_day1_ok(self):
        o = paid_order(total=D("9999.99"), paid_on=date(2024, 1, 1))
        o.pick(date(2024, 1, 1))
        o.cancel(date(2024, 1, 2))   # 1 day — at limit
        assert o.state is OrderState.CANCELLED

    def test_rule_c_high_value_picked_day2_fail(self):
        o = paid_order(total=D("9999.99"), paid_on=date(2024, 1, 1))
        o.pick(date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 3))   # 2 days — over limit


# ===========================================================================
# 5. RULE R — Refund delivered order (boundary-value analysis)
# ===========================================================================

class TestRuleR:
    """
    Three conjunctive conditions:
      1. payment in {CREDIT_CARD, PIX}
      2. days since delivery ≤ 7
      3. total ≤ 2000.00
    All must hold; failing any one raises RuleViolation.
    """

    # --- Condition 1: payment method ---

    def test_rule_r_boleto_rejected(self):
        o = delivered_order(method=PaymentMethod.BOLETO, total=D("100.00"))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 6))

    def test_rule_r_credit_card_accepted(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"))
        o.refund(date(2024, 1, 6))
        assert o.state is OrderState.REFUNDED

    def test_rule_r_pix_accepted(self):
        o = delivered_order(method=PaymentMethod.PIX, total=D("100.00"))
        o.refund(date(2024, 1, 6))
        assert o.state is OrderState.REFUNDED

    # --- Condition 2: days since delivery ---

    def test_rule_r_delivery_day0_ok(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), delivered_on=date(2024, 1, 5))
        o.refund(date(2024, 1, 5))   # 0 days
        assert o.state is OrderState.REFUNDED

    def test_rule_r_delivery_day7_ok(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), delivered_on=date(2024, 1, 5))
        o.refund(date(2024, 1, 12))   # 7 days — at limit
        assert o.state is OrderState.REFUNDED

    def test_rule_r_delivery_day8_fail(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("100.00"), delivered_on=date(2024, 1, 5))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 13))   # 8 days — over limit

    # --- Condition 3: total ≤ 2000 ---

    def test_rule_r_total_2000_ok(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("2000.00"), delivered_on=date(2024, 1, 5))
        o.refund(date(2024, 1, 6))
        assert o.state is OrderState.REFUNDED

    def test_rule_r_total_2000_01_fail(self):
        o = delivered_order(method=PaymentMethod.CREDIT_CARD, total=D("2000.01"), delivered_on=date(2024, 1, 5))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 6))

    # --- Interaction: fail on method even if other conditions hold ---

    def test_rule_r_fail_method_despite_ok_days_and_total(self):
        o = delivered_order(method=PaymentMethod.BOLETO, total=D("100.00"), delivered_on=date(2024, 1, 5))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 6))   # method wrong

    # --- Interaction: fail on total even if method and days OK ---

    def test_rule_r_fail_total_despite_ok_method_and_days(self):
        o = delivered_order(method=PaymentMethod.PIX, total=D("5000.00"), delivered_on=date(2024, 1, 5))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 6))   # total wrong


# ===========================================================================
# 6. RULE R2 — Refund cancelled order (boundary-value analysis)
# ===========================================================================

class TestRuleR2:
    """
    Two conditions:
      1. Order was paid before cancellation.
      2. If BOLETO, at least 1 day since payment.
    """

    # --- Condition 1: must have been paid ---

    def test_rule_r2_cancelled_without_payment_rejected(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))   # cancelled from CREATED (never paid)
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 3))

    def test_rule_r2_cancelled_after_payment_accepted_credit(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        o.refund(date(2024, 1, 1))   # same day — OK for CREDIT_CARD
        assert o.state is OrderState.REFUNDED

    def test_rule_r2_cancelled_after_payment_accepted_pix(self):
        o = paid_order(method=PaymentMethod.PIX, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        o.refund(date(2024, 1, 1))
        assert o.state is OrderState.REFUNDED

    # --- Condition 2: BOLETO needs ≥ 1 day since payment ---

    def test_rule_r2_boleto_day0_since_payment_rejected(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 1))   # 0 days — not enough

    def test_rule_r2_boleto_day1_since_payment_ok(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        o.refund(date(2024, 1, 2))   # 1 day — at limit
        assert o.state is OrderState.REFUNDED

    def test_rule_r2_boleto_day2_since_payment_ok(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 2))
        o.refund(date(2024, 1, 3))   # 2 days — above limit
        assert o.state is OrderState.REFUNDED

    def test_rule_r2_boleto_cancel_at_picked_day1_ok(self):
        # Payment at day 1, pick at day 1, cancel at day 2 (1 day since pay)
        # then refund at day 2 — 1 day since payment → OK
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=date(2024, 1, 1))
        o.pick(date(2024, 1, 1))
        o.cancel(date(2024, 1, 2))   # rule C: 1 day — at boleto limit → OK
        o.refund(date(2024, 1, 2))   # 1 day since payment → OK
        assert o.state is OrderState.REFUNDED


# ===========================================================================
# 7. VERIFICATION ORDER — argument → transition → rule
# ===========================================================================

class TestVerificationOrder:
    """
    When multiple things are wrong, the mandated check order must hold:
    InvalidOrder before InvalidTransition before RuleViolation.
    """

    def test_bad_date_before_invalid_transition(self):
        """REFUNDED state + past date → InvalidOrder (not InvalidTransition)."""
        o = delivered_order(method=PaymentMethod.PIX, total=D("100.00"), delivered_on=date(2024, 1, 5))
        o.refund(date(2024, 1, 7))
        # Now in REFUNDED; use a past date AND invalid action
        with pytest.raises(InvalidOrder):
            o.cancel(date(2024, 1, 1))   # past date → InvalidOrder first

    def test_invalid_transition_before_rule_violation(self):
        """SHIPPED state + cancel (invalid) → InvalidTransition (not RuleViolation)."""
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.cancel(date(2024, 1, 5))

    def test_bad_on_type_before_rule(self):
        """datetime.datetime for on → InvalidOrder even if rule would also fail."""
        o = delivered_order(method=PaymentMethod.BOLETO, total=D("5000.00"), delivered_on=date(2024, 1, 5))
        with pytest.raises(InvalidOrder):
            o.refund(datetime.datetime(2024, 1, 6))  # type: ignore[arg-type]


# ===========================================================================
# 8. EXCEPTION SAFETY — failed calls must not alter state
# ===========================================================================

class TestExceptionSafety:
    """A rejected call must leave the Order in its original state."""

    def test_failed_pay_leaves_created(self):
        o = make_order()
        try:
            o.pay("bad", date(2024, 1, 2))  # type: ignore[arg-type]
        except OrderError:
            pass
        assert o.state is OrderState.CREATED

    def test_failed_cancel_rule_c_leaves_paid(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, paid_on=date(2024, 1, 1))
        try:
            o.cancel(date(2024, 1, 10))   # 9 days — rule C violated
        except RuleViolation:
            pass
        assert o.state is OrderState.PAID

    def test_failed_refund_rule_r_leaves_delivered(self):
        o = delivered_order(method=PaymentMethod.BOLETO, total=D("100.00"), delivered_on=date(2024, 1, 5))
        try:
            o.refund(date(2024, 1, 6))   # method wrong
        except RuleViolation:
            pass
        assert o.state is OrderState.DELIVERED

    def test_failed_refund_rule_r2_leaves_cancelled(self):
        o = make_order()
        o.cancel(date(2024, 1, 2))
        try:
            o.refund(date(2024, 1, 3))   # never paid
        except RuleViolation:
            pass
        assert o.state is OrderState.CANCELLED

    def test_last_event_date_not_updated_on_failure(self):
        o = paid_order(paid_on=date(2024, 1, 5))
        try:
            o.pick(date(2024, 1, 4))   # before last event
        except InvalidOrder:
            pass
        # last event date should still be 2024-01-05 (payment)
        # verify by checking a pick with 2024-01-05 still works
        o.pick(date(2024, 1, 5))
        assert o.state is OrderState.PICKED


# ===========================================================================
# 9. EDGE CASES & ADDITIONAL SCENARIOS
# ===========================================================================

class TestEdgeCases:
    """Additional scenarios covering corner cases and cross-rule interactions."""

    def test_pay_on_same_day_as_created(self):
        o = Order("X", D("50.00"), date(2024, 6, 1))
        o.pay(PaymentMethod.PIX, date(2024, 6, 1))
        assert o.state is OrderState.PAID

    def test_all_actions_on_same_day(self):
        d = date(2024, 6, 15)
        o = Order("SAME-DAY", D("100.00"), d)
        o.pay(PaymentMethod.PIX, d)
        o.pick(d)
        o.ship(d)
        o.deliver(d)
        assert o.state is OrderState.DELIVERED

    def test_boleto_cancel_at_exactly_limit_high_value(self):
        """BOLETO + total > 5000 → limit 1 (high-value rule wins)."""
        o = paid_order(method=PaymentMethod.BOLETO, total=D("5000.01"), paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 2))   # 1 day — at limit
        assert o.state is OrderState.CANCELLED

    def test_boleto_cancel_over_limit_high_value(self):
        """BOLETO + total > 5000 → limit 1; day 2 → fail."""
        o = paid_order(method=PaymentMethod.BOLETO, total=D("5000.01"), paid_on=date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(date(2024, 1, 3))   # 2 days

    def test_refund_delivered_pix_total_exactly_2000(self):
        o = delivered_order(method=PaymentMethod.PIX, total=D("2000.00"), delivered_on=date(2024, 2, 1))
        o.refund(date(2024, 2, 1))
        assert o.state is OrderState.REFUNDED

    def test_refund_delivered_requires_all_three_conditions(self):
        """Method OK, days OK, but total over limit → RuleViolation."""
        o = delivered_order(method=PaymentMethod.PIX, total=D("2000.01"), delivered_on=date(2024, 2, 1))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 2, 2))

    def test_cancel_from_created_always_allowed(self):
        """No business rule on cancelling a CREATED order."""
        o = Order("Z", D("99999.99"), date(2024, 1, 1))
        o.cancel(date(2024, 1, 1))
        assert o.state is OrderState.CANCELLED

    def test_cancel_from_created_then_refund_blocked(self):
        """Cancelled without payment → R2 blocks refund."""
        o = make_order()
        o.cancel(date(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.refund(date(2024, 1, 2))

    def test_decimal_vs_float_in_total(self):
        """Confirm float is specifically rejected (not just non-Decimal)."""
        with pytest.raises(InvalidOrder):
            Order("X", 100.0, date(2024, 1, 1))

    def test_decimal_subclass_rejected_as_int(self):
        """int is not Decimal."""
        with pytest.raises(InvalidOrder):
            Order("X", 100, date(2024, 1, 1))   # type: ignore[arg-type]

    def test_state_property_initial(self):
        o = make_order()
        assert o.state is OrderState.CREATED

    def test_payment_method_pix_full_cancel_refund_cycle(self):
        o = paid_order(method=PaymentMethod.PIX, paid_on=date(2024, 5, 1))
        o.cancel(date(2024, 5, 3))   # 2 days — within 7
        o.refund(date(2024, 5, 3))   # PIX, paid before cancel → OK
        assert o.state is OrderState.REFUNDED

    def test_pick_after_cancel_invalid(self):
        o = paid_order()
        o.cancel(date(2024, 1, 3))
        with pytest.raises(InvalidTransition):
            o.pick(date(2024, 1, 4))

    def test_multiple_payment_methods_at_rule_r2_boleto_boundary(self):
        """BOLETO refund on day 0 fails; on day 1 it passes."""
        for paid_on, refund_on, should_pass in [
            (date(2024, 1, 1), date(2024, 1, 1), False),  # 0 days → fail
            (date(2024, 1, 1), date(2024, 1, 2), True),   # 1 day  → pass
        ]:
            o = paid_order(method=PaymentMethod.BOLETO, paid_on=paid_on)
            o.cancel(paid_on)
            if should_pass:
                o.refund(refund_on)
                assert o.state is OrderState.REFUNDED
            else:
                with pytest.raises(RuleViolation):
                    o.refund(refund_on)

    def test_rule_r2_boleto_refund_date_must_not_precede_last_event(self):
        """
        After cancelling on day 1, refund date must be >= day 1 (last event).
        Attempting to pass a date before the cancel raises InvalidOrder (not RuleViolation).
        """
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=date(2024, 1, 1))
        o.cancel(date(2024, 1, 2))
        with pytest.raises(InvalidOrder):
            o.refund(date(2024, 1, 1))   # before cancel date



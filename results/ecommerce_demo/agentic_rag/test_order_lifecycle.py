"""
test_order_lifecycle.py
=======================
Suíte de testes para order_lifecycle.py (pytest).

Estratégia (baseada em SWEBOK):
  * Particionamento de equivalência: classes válidas e inválidas para cada argumento.
  * Análise de valores-limite: fronteiras exatas das regras C e R (dias, total).
  * Teste de transição de estados: todas as transições válidas e inválidas cobertas.
  * Teste de exceção forçada (§3.1.11 SWEBOK): cada caminho de erro é exercitado.
  * Atomicidade: pedidos que levantam exceção não devem alterar o estado.
  * Tabelas de decisão: combinações de forma de pagamento × total × dias para R e C.
"""

import pytest
from datetime import date, datetime
from decimal import Decimal

from order_lifecycle import (
    Order,
    OrderState,
    OrderError,
    InvalidOrder,
    InvalidTransition,
    RuleViolation,
    PaymentMethod,
)


# ===========================================================================
# Fixtures auxiliares
# ===========================================================================

D = date  # alias para reduzir verbosidade

def make_order(
    order_id: str = "ORD-001",
    total: Decimal = Decimal("100.00"),
    created_on: date = D(2024, 1, 10),
) -> Order:
    return Order(order_id=order_id, total=total, created_on=created_on)


def paid_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    pay_date: date = D(2024, 1, 11),
    total: Decimal = Decimal("100.00"),
    created_on: date = D(2024, 1, 10),
) -> Order:
    o = make_order(total=total, created_on=created_on)
    o.pay(method, on=pay_date)
    return o


def picked_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    pay_date: date = D(2024, 1, 11),
    pick_date: date = D(2024, 1, 12),
    total: Decimal = Decimal("100.00"),
) -> Order:
    o = paid_order(method=method, pay_date=pay_date, total=total)
    o.pick(on=pick_date)
    return o


def shipped_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    pay_date: date = D(2024, 1, 11),
    pick_date: date = D(2024, 1, 12),
    ship_date: date = D(2024, 1, 13),
    total: Decimal = Decimal("100.00"),
) -> Order:
    o = picked_order(method=method, pay_date=pay_date, pick_date=pick_date, total=total)
    o.ship(on=ship_date)
    return o


def delivered_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    pay_date: date = D(2024, 1, 11),
    ship_date: date = D(2024, 1, 13),
    deliver_date: date = D(2024, 1, 15),
    total: Decimal = Decimal("100.00"),
) -> Order:
    o = shipped_order(
        method=method, pay_date=pay_date, ship_date=ship_date, total=total
    )
    o.deliver(on=deliver_date)
    return o


# ===========================================================================
# 1. Validação de construção – InvalidOrder
# ===========================================================================

class TestOrderConstruction:

    def test_valid_construction(self):
        o = make_order()
        assert o.state == OrderState.CREATED

    def test_order_id_empty_string(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="", total=Decimal("10.00"), created_on=D(2024, 1, 1))

    def test_order_id_not_string(self):
        with pytest.raises(InvalidOrder):
            Order(order_id=123, total=Decimal("10.00"), created_on=D(2024, 1, 1))

    def test_total_must_be_decimal_not_float(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=10.0, created_on=D(2024, 1, 1))

    def test_total_must_be_decimal_not_int(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=10, created_on=D(2024, 1, 1))

    def test_total_zero_rejected(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("0"), created_on=D(2024, 1, 1))

    def test_total_negative_rejected(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("-1.00"), created_on=D(2024, 1, 1))

    def test_total_three_decimal_places_rejected(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("1.001"), created_on=D(2024, 1, 1))

    def test_total_two_decimal_places_accepted(self):
        o = Order(order_id="X", total=Decimal("1.99"), created_on=D(2024, 1, 1))
        assert o.state == OrderState.CREATED

    def test_total_one_decimal_place_accepted(self):
        o = Order(order_id="X", total=Decimal("5.5"), created_on=D(2024, 1, 1))
        assert o.state == OrderState.CREATED

    def test_total_whole_number_accepted(self):
        o = Order(order_id="X", total=Decimal("100"), created_on=D(2024, 1, 1))
        assert o.state == OrderState.CREATED

    def test_created_on_datetime_rejected(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("10.00"), created_on=datetime(2024, 1, 1))

    def test_created_on_not_date_rejected(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("10.00"), created_on="2024-01-01")


# ===========================================================================
# 2. Validação de argumentos nas ações – InvalidOrder
# ===========================================================================

class TestActionArgumentValidation:

    def test_pay_datetime_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, on=datetime(2024, 1, 11))

    def test_pay_invalid_method_type(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay("CREDIT_CARD", on=D(2024, 1, 11))

    def test_pay_method_none_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(None, on=D(2024, 1, 11))

    def test_pick_datetime_rejected(self):
        o = paid_order()
        with pytest.raises(InvalidOrder):
            o.pick(on=datetime(2024, 1, 12))

    def test_ship_datetime_rejected(self):
        o = picked_order()
        with pytest.raises(InvalidOrder):
            o.ship(on=datetime(2024, 1, 13))

    def test_deliver_datetime_rejected(self):
        o = shipped_order()
        with pytest.raises(InvalidOrder):
            o.deliver(on=datetime(2024, 1, 14))

    def test_cancel_datetime_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.cancel(on=datetime(2024, 1, 11))

    def test_refund_datetime_rejected(self):
        o = delivered_order()
        with pytest.raises(InvalidOrder):
            o.refund(on=datetime(2024, 1, 15))

    # Datas anteriores ao último evento
    def test_pay_before_creation_rejected(self):
        o = make_order(created_on=D(2024, 1, 10))
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, on=D(2024, 1, 9))

    def test_pay_on_creation_date_accepted(self):
        o = make_order(created_on=D(2024, 1, 10))
        o.pay(PaymentMethod.PIX, on=D(2024, 1, 10))
        assert o.state == OrderState.PAID

    def test_pick_before_payment_date_rejected(self):
        o = paid_order(pay_date=D(2024, 1, 11))
        with pytest.raises(InvalidOrder):
            o.pick(on=D(2024, 1, 10))

    def test_ship_before_pick_date_rejected(self):
        o = picked_order(pick_date=D(2024, 1, 12))
        with pytest.raises(InvalidOrder):
            o.ship(on=D(2024, 1, 11))

    def test_deliver_before_ship_date_rejected(self):
        o = shipped_order(ship_date=D(2024, 1, 13))
        with pytest.raises(InvalidOrder):
            o.deliver(on=D(2024, 1, 12))

    def test_cancel_before_last_event_rejected(self):
        o = paid_order(pay_date=D(2024, 1, 11))
        with pytest.raises(InvalidOrder):
            o.cancel(on=D(2024, 1, 10))

    def test_refund_before_delivery_rejected(self):
        o = delivered_order(deliver_date=D(2024, 1, 15))
        with pytest.raises(InvalidOrder):
            o.refund(on=D(2024, 1, 14))


# ===========================================================================
# 3. Transições válidas (happy path)
# ===========================================================================

class TestValidTransitions:

    def test_created_to_paid(self):
        o = make_order()
        o.pay(PaymentMethod.CREDIT_CARD, on=D(2024, 1, 11))
        assert o.state == OrderState.PAID

    def test_paid_to_picked(self):
        o = paid_order()
        o.pick(on=D(2024, 1, 12))
        assert o.state == OrderState.PICKED

    def test_picked_to_shipped(self):
        o = picked_order()
        o.ship(on=D(2024, 1, 13))
        assert o.state == OrderState.SHIPPED

    def test_shipped_to_delivered(self):
        o = shipped_order()
        o.deliver(on=D(2024, 1, 14))
        assert o.state == OrderState.DELIVERED

    def test_delivered_to_refunded(self):
        o = delivered_order(
            deliver_date=D(2024, 1, 15),
            total=Decimal("500.00"),
        )
        o.refund(on=D(2024, 1, 20))
        assert o.state == OrderState.REFUNDED

    def test_created_to_cancelled(self):
        o = make_order()
        o.cancel(on=D(2024, 1, 11))
        assert o.state == OrderState.CANCELLED

    def test_paid_to_cancelled_within_rule_c(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_date=D(2024, 1, 11))
        o.cancel(on=D(2024, 1, 15))  # 4 dias — limite 7
        assert o.state == OrderState.CANCELLED

    def test_picked_to_cancelled_within_rule_c(self):
        o = picked_order(method=PaymentMethod.CREDIT_CARD, pay_date=D(2024, 1, 11))
        o.cancel(on=D(2024, 1, 15))  # 4 dias — limite 7
        assert o.state == OrderState.CANCELLED

    def test_cancelled_to_refunded(self):
        # Pago e depois cancelado no mesmo dia → R2 sem boleto
        o = paid_order(method=PaymentMethod.PIX, pay_date=D(2024, 1, 11))
        o.cancel(on=D(2024, 1, 15))
        o.refund(on=D(2024, 1, 15))
        assert o.state == OrderState.REFUNDED

    def test_full_lifecycle(self):
        o = make_order()
        o.pay(PaymentMethod.PIX, on=D(2024, 1, 11))
        o.pick(on=D(2024, 1, 12))
        o.ship(on=D(2024, 1, 13))
        o.deliver(on=D(2024, 1, 14))
        o.refund(on=D(2024, 1, 20))
        assert o.state == OrderState.REFUNDED

    def test_all_payment_methods_accepted(self):
        for method in PaymentMethod:
            o = make_order()
            o.pay(method, on=D(2024, 1, 11))
            assert o.state == OrderState.PAID


# ===========================================================================
# 4. Transições inválidas – InvalidTransition
# ===========================================================================

class TestInvalidTransitions:

    def test_pick_on_created(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.pick(on=D(2024, 1, 11))

    def test_ship_on_created(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.ship(on=D(2024, 1, 11))

    def test_deliver_on_created(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.deliver(on=D(2024, 1, 11))

    def test_refund_on_created(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.refund(on=D(2024, 1, 11))

    def test_pay_on_paid(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, on=D(2024, 1, 15))

    def test_ship_on_paid(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.ship(on=D(2024, 1, 15))

    def test_deliver_on_paid(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.deliver(on=D(2024, 1, 15))

    def test_refund_on_paid(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.refund(on=D(2024, 1, 15))

    def test_pay_on_picked(self):
        o = picked_order()
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, on=D(2024, 1, 15))

    def test_deliver_on_picked(self):
        o = picked_order()
        with pytest.raises(InvalidTransition):
            o.deliver(on=D(2024, 1, 15))

    def test_refund_on_picked(self):
        o = picked_order()
        with pytest.raises(InvalidTransition):
            o.refund(on=D(2024, 1, 15))

    def test_cancel_on_shipped(self):
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.cancel(on=D(2024, 1, 15))

    def test_refund_on_shipped(self):
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.refund(on=D(2024, 1, 15))

    def test_cancel_on_delivered(self):
        o = delivered_order()
        with pytest.raises(InvalidTransition):
            o.cancel(on=D(2024, 1, 20))

    def test_pay_on_cancelled(self):
        o = make_order()
        o.cancel(on=D(2024, 1, 11))
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.CREDIT_CARD, on=D(2024, 1, 12))

    def test_cancel_on_cancelled(self):
        o = make_order()
        o.cancel(on=D(2024, 1, 11))
        with pytest.raises(InvalidTransition):
            o.cancel(on=D(2024, 1, 12))

    def test_any_action_on_refunded(self):
        o = delivered_order(total=Decimal("500.00"), deliver_date=D(2024, 1, 15))
        o.refund(on=D(2024, 1, 20))
        for action_name, kwargs in [
            ("pay",     {"method": PaymentMethod.PIX}),
            ("pick",    {}),
            ("ship",    {}),
            ("deliver", {}),
            ("cancel",  {}),
            ("refund",  {}),
        ]:
            method = getattr(o, action_name)
            call_kwargs = {**kwargs, "on": D(2024, 1, 21)}
            with pytest.raises(InvalidTransition):
                method(**call_kwargs)


# ===========================================================================
# 5. Regra C — cancelar pedido pago/picked
# ===========================================================================

class TestRuleC:

    # ---- Total > 5000: limite 1 dia ----

    @pytest.mark.parametrize("method", list(PaymentMethod))
    def test_rule_c_high_value_exactly_1_day_allowed(self, method):
        o = paid_order(method=method, pay_date=D(2024, 1, 10), total=Decimal("5001.00"))
        o.cancel(on=D(2024, 1, 11))  # 1 dia — no limite
        assert o.state == OrderState.CANCELLED

    @pytest.mark.parametrize("method", list(PaymentMethod))
    def test_rule_c_high_value_2_days_rejected(self, method):
        o = paid_order(method=method, pay_date=D(2024, 1, 10), total=Decimal("5001.00"))
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 1, 12))  # 2 dias — acima do limite

    def test_rule_c_high_value_exactly_5000_not_high(self):
        """5000.00 não é > 5000, então cai na regra de método."""
        o = paid_order(
            method=PaymentMethod.CREDIT_CARD,
            pay_date=D(2024, 1, 10),
            total=Decimal("5000.00"),
        )
        # Com CC, limite é 7 dias
        o.cancel(on=D(2024, 1, 17))  # 7 dias — no limite
        assert o.state == OrderState.CANCELLED

    # ---- Total <= 5000, BOLETO: limite 2 dias ----

    def test_rule_c_boleto_exactly_2_days_allowed(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10), total=Decimal("100.00"))
        o.cancel(on=D(2024, 1, 12))  # 2 dias — no limite
        assert o.state == OrderState.CANCELLED

    def test_rule_c_boleto_3_days_rejected(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10), total=Decimal("100.00"))
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 1, 13))  # 3 dias — acima do limite

    def test_rule_c_boleto_1_day_allowed(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10), total=Decimal("100.00"))
        o.cancel(on=D(2024, 1, 11))  # 1 dia — abaixo do limite
        assert o.state == OrderState.CANCELLED

    # ---- Total <= 5000, CC ou PIX: limite 7 dias ----

    @pytest.mark.parametrize("method", [PaymentMethod.CREDIT_CARD, PaymentMethod.PIX])
    def test_rule_c_cc_pix_exactly_7_days_allowed(self, method):
        o = paid_order(method=method, pay_date=D(2024, 1, 10), total=Decimal("100.00"))
        o.cancel(on=D(2024, 1, 17))  # 7 dias — no limite
        assert o.state == OrderState.CANCELLED

    @pytest.mark.parametrize("method", [PaymentMethod.CREDIT_CARD, PaymentMethod.PIX])
    def test_rule_c_cc_pix_8_days_rejected(self, method):
        o = paid_order(method=method, pay_date=D(2024, 1, 10), total=Decimal("100.00"))
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 1, 18))  # 8 dias — acima do limite

    @pytest.mark.parametrize("method", [PaymentMethod.CREDIT_CARD, PaymentMethod.PIX])
    def test_rule_c_cc_pix_0_days_allowed(self, method):
        o = paid_order(method=method, pay_date=D(2024, 1, 10), total=Decimal("100.00"))
        o.cancel(on=D(2024, 1, 10))  # 0 dias — bem dentro do limite
        assert o.state == OrderState.CANCELLED

    # ---- Regra C no estado PICKED ----

    def test_rule_c_from_picked_boleto_within_limit(self):
        o = picked_order(
            method=PaymentMethod.BOLETO,
            pay_date=D(2024, 1, 10),
            pick_date=D(2024, 1, 11),
        )
        o.cancel(on=D(2024, 1, 12))  # 2 dias desde pagamento
        assert o.state == OrderState.CANCELLED

    def test_rule_c_from_picked_boleto_outside_limit(self):
        o = picked_order(
            method=PaymentMethod.BOLETO,
            pay_date=D(2024, 1, 10),
            pick_date=D(2024, 1, 11),
        )
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 1, 13))  # 3 dias desde pagamento

    def test_rule_c_from_picked_high_value_within_limit(self):
        o = picked_order(
            method=PaymentMethod.CREDIT_CARD,
            pay_date=D(2024, 1, 10),
            pick_date=D(2024, 1, 11),
            total=Decimal("9999.99"),
        )
        o.cancel(on=D(2024, 1, 11))  # 1 dia desde pagamento
        assert o.state == OrderState.CANCELLED

    def test_rule_c_from_picked_high_value_outside_limit(self):
        o = picked_order(
            method=PaymentMethod.CREDIT_CARD,
            pay_date=D(2024, 1, 10),
            pick_date=D(2024, 1, 11),
            total=Decimal("9999.99"),
        )
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 1, 12))  # 2 dias desde pagamento

    # Cancelamento de CREATED nunca exige regra C
    def test_cancel_created_always_allowed(self):
        for method in PaymentMethod:
            o = make_order(total=Decimal("9999.99"), created_on=D(2024, 1, 1))
            o.cancel(on=D(2024, 1, 31))
            assert o.state == OrderState.CANCELLED


# ===========================================================================
# 6. Regra R — reembolsar pedido entregue
# ===========================================================================

class TestRuleR:

    # ---- Condição: forma de pagamento ----

    @pytest.mark.parametrize("method", [PaymentMethod.CREDIT_CARD, PaymentMethod.PIX])
    def test_rule_r_allowed_methods(self, method):
        o = delivered_order(method=method, deliver_date=D(2024, 1, 15), total=Decimal("500.00"))
        o.refund(on=D(2024, 1, 20))
        assert o.state == OrderState.REFUNDED

    def test_rule_r_boleto_rejected(self):
        o = delivered_order(method=PaymentMethod.BOLETO, deliver_date=D(2024, 1, 15), total=Decimal("500.00"))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 20))

    # ---- Condição: prazo ----

    def test_rule_r_exactly_7_days_allowed(self):
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("500.00"))
        o.refund(on=D(2024, 1, 22))  # 7 dias — no limite
        assert o.state == OrderState.REFUNDED

    def test_rule_r_8_days_rejected(self):
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("500.00"))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 23))  # 8 dias

    def test_rule_r_0_days_allowed(self):
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("500.00"))
        o.refund(on=D(2024, 1, 15))  # mesmo dia
        assert o.state == OrderState.REFUNDED

    # ---- Condição: total ----

    def test_rule_r_exactly_2000_allowed(self):
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("2000.00"))
        o.refund(on=D(2024, 1, 20))
        assert o.state == OrderState.REFUNDED

    def test_rule_r_total_above_2000_rejected(self):
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("2000.01"))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 20))

    def test_rule_r_all_three_conditions_must_hold(self):
        """BOLETO + > 7 dias + > 2000 → testa falha em qualquer combinação."""
        # Só BOLETO falha (outros OK)
        o = delivered_order(method=PaymentMethod.BOLETO, deliver_date=D(2024, 1, 15), total=Decimal("100.00"))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 15))

        # Só prazo falha (outros OK)
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("100.00"))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 23))  # 8 dias

        # Só total falha (outros OK)
        o = delivered_order(method=PaymentMethod.PIX, deliver_date=D(2024, 1, 15), total=Decimal("2500.00"))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 15))


# ===========================================================================
# 7. Regra R2 — reembolsar pedido cancelado
# ===========================================================================

class TestRuleR2:

    def test_r2_paid_then_cancelled_cc(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 15))
        o.refund(on=D(2024, 1, 15))
        assert o.state == OrderState.REFUNDED

    def test_r2_paid_then_cancelled_pix(self):
        o = paid_order(method=PaymentMethod.PIX, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 15))
        o.refund(on=D(2024, 1, 20))
        assert o.state == OrderState.REFUNDED

    def test_r2_cancelled_without_payment_rejected(self):
        """Pedido cancelado sem nunca ter sido pago → R2 rejeita."""
        o = make_order(created_on=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 11))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 11))

    def test_r2_boleto_same_day_as_payment_rejected(self):
        """BOLETO exige ao menos 1 dia desde o pagamento."""
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 10))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 10))  # 0 dias desde pagamento

    def test_r2_boleto_exactly_1_day_allowed(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 11))
        o.refund(on=D(2024, 1, 11))  # 1 dia desde pagamento
        assert o.state == OrderState.REFUNDED

    def test_r2_boleto_many_days_allowed(self):
        # BOLETO: cancela dentro do prazo C (2 dias) e depois reembolsa
        # (R2 não impõe prazo máximo desde pagamento para BOLETO — só mínimo de 1 dia)
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 12))   # 2 dias — dentro do limite da Regra C
        o.refund(on=D(2024, 1, 20))   # 10 dias desde pagamento — R2 apenas exige >= 1 dia
        assert o.state == OrderState.REFUNDED

    def test_r2_boleto_cancel_before_1_day_then_refund_next_day(self):
        """Cancela no mesmo dia do pagamento, reembolsa no dia seguinte → OK."""
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 10))
        o.refund(on=D(2024, 1, 11))  # 1 dia desde pagamento
        assert o.state == OrderState.REFUNDED

    # R2 não impõe limite de valor nem de prazo além do pagamento
    def test_r2_high_value_cc_allowed(self):
        o = paid_order(
            method=PaymentMethod.CREDIT_CARD,
            pay_date=D(2024, 1, 10),
            total=Decimal("9999.99"),
        )
        o.cancel(on=D(2024, 1, 11))
        o.refund(on=D(2024, 1, 11))
        assert o.state == OrderState.REFUNDED

    def test_r2_many_days_after_payment_cc_allowed(self):
        """R2 não impõe prazo máximo desde pagamento (apenas BOLETO tem prazo mínimo)."""
        o = Order("X", Decimal("100.00"), D(2024, 1, 1))
        o.pay(PaymentMethod.CREDIT_CARD, on=D(2024, 1, 1))
        o.cancel(on=D(2024, 1, 7))    # dentro do limite C (7 dias CC)
        o.refund(on=D(2024, 6, 1))    # R2: sem prazo máximo para CC
        assert o.state == OrderState.REFUNDED


# ===========================================================================
# 8. Atomicidade — exceção não altera o estado do pedido
# ===========================================================================

class TestAtomicity:

    def test_invalid_transition_does_not_change_state(self):
        o = make_order()
        try:
            o.pick(on=D(2024, 1, 11))
        except InvalidTransition:
            pass
        assert o.state == OrderState.CREATED

    def test_rule_violation_does_not_change_state(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10))
        try:
            o.cancel(on=D(2024, 1, 20))  # > 2 dias → RuleViolation
        except RuleViolation:
            pass
        assert o.state == OrderState.PAID

    def test_invalid_order_does_not_change_state(self):
        o = make_order()
        try:
            o.pay("not_a_method", on=D(2024, 1, 11))
        except InvalidOrder:
            pass
        assert o.state == OrderState.CREATED

    def test_state_preserved_after_multiple_failed_attempts(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_date=D(2024, 1, 10))
        for _ in range(3):
            try:
                o.ship(on=D(2024, 1, 15))  # InvalidTransition: não foi para PICKED
            except InvalidTransition:
                pass
        assert o.state == OrderState.PAID


# ===========================================================================
# 9. Ordem das verificações (InvalidOrder > InvalidTransition > RuleViolation)
# ===========================================================================

class TestVerificationOrder:

    def test_invalid_date_takes_priority_over_invalid_transition(self):
        """Estado SHIPPED não tem cancel, mas o erro deve ser de argumento (datetime)."""
        o = shipped_order()
        with pytest.raises(InvalidOrder):
            o.cancel(on=datetime(2024, 1, 15))

    def test_invalid_date_takes_priority_over_rule_violation(self):
        """Data anterior tem prioridade sobre regra R."""
        o = delivered_order(deliver_date=D(2024, 1, 15), total=Decimal("500.00"))
        with pytest.raises(InvalidOrder):
            o.refund(on=D(2024, 1, 14))  # anterior ao último evento

    def test_invalid_transition_takes_priority_over_rule_violation(self):
        """Estado SHIPPED não tem refund — deve ser InvalidTransition, não RuleViolation."""
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.refund(on=D(2024, 1, 20))

    def test_invalid_method_type_raises_invalid_order_not_transition(self):
        """pay() com method inválido → InvalidOrder (antes de checar transição)."""
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay("PIX", on=D(2024, 1, 11))


# ===========================================================================
# 10. Hierarquia de exceções
# ===========================================================================

class TestExceptionHierarchy:

    def test_invalid_order_is_order_error(self):
        with pytest.raises(OrderError):
            Order(order_id="", total=Decimal("10.00"), created_on=D(2024, 1, 1))

    def test_invalid_transition_is_order_error(self):
        o = make_order()
        with pytest.raises(OrderError):
            o.pick(on=D(2024, 1, 11))

    def test_rule_violation_is_order_error(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_date=D(2024, 1, 10))
        with pytest.raises(OrderError):
            o.cancel(on=D(2024, 1, 20))

    def test_exceptions_are_distinct(self):
        assert not issubclass(InvalidOrder, InvalidTransition)
        assert not issubclass(InvalidTransition, RuleViolation)
        assert not issubclass(RuleViolation, InvalidOrder)


# ===========================================================================
# 11. Testes de fronteira adicionais (boundary value analysis)
# ===========================================================================

class TestBoundaryValues:

    def test_total_minimum_valid_cent(self):
        o = Order(order_id="X", total=Decimal("0.01"), created_on=D(2024, 1, 1))
        assert o.state == OrderState.CREATED

    def test_rule_c_boundary_high_value_exactly_5000_01(self):
        """5000.01 é > 5000 → limite 1 dia."""
        o = paid_order(
            method=PaymentMethod.CREDIT_CARD,
            pay_date=D(2024, 1, 10),
            total=Decimal("5000.01"),
        )
        o.cancel(on=D(2024, 1, 11))  # 1 dia — no limite
        assert o.state == OrderState.CANCELLED

    def test_rule_c_boundary_high_value_exactly_5000_01_day2(self):
        o = paid_order(
            method=PaymentMethod.CREDIT_CARD,
            pay_date=D(2024, 1, 10),
            total=Decimal("5000.01"),
        )
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 1, 12))  # 2 dias

    def test_rule_r_boundary_exactly_2000_00(self):
        o = delivered_order(
            method=PaymentMethod.PIX,
            deliver_date=D(2024, 1, 15),
            total=Decimal("2000.00"),
        )
        o.refund(on=D(2024, 1, 15))
        assert o.state == OrderState.REFUNDED

    def test_rule_r_boundary_2000_01_rejected(self):
        o = delivered_order(
            method=PaymentMethod.PIX,
            deliver_date=D(2024, 1, 15),
            total=Decimal("2000.01"),
        )
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 15))

    def test_same_day_actions_accepted(self):
        """Todas as transições podem ocorrer no mesmo dia."""
        d = D(2024, 1, 10)
        o = make_order(created_on=d)
        o.pay(PaymentMethod.CREDIT_CARD, on=d)
        o.pick(on=d)
        o.ship(on=d)
        o.deliver(on=d)
        o.refund(on=d)
        assert o.state == OrderState.REFUNDED

    def test_total_with_scientific_notation_decimal(self):
        """Decimal('1E+2') é 100 inteiros — sem casas decimais (exponent=2 >= 0)."""
        o = Order(order_id="X", total=Decimal("1E+2"), created_on=D(2024, 1, 1))
        assert o.state == OrderState.CREATED

    def test_total_decimal_negative_exponent_3_rejected(self):
        """Decimal('1.001') tem 3 casas decimais → rejeitado."""
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("1.001"), created_on=D(2024, 1, 1))

    def test_cancel_exactly_on_last_event_date_ok(self):
        """Cancelar exatamente na data do pagamento (0 dias) deve ser aceito."""
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_date=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 10))
        assert o.state == OrderState.CANCELLED


# ===========================================================================
# 12. Cobertura de estados terminais
# ===========================================================================

class TestTerminalStates:

    def test_refunded_state_has_no_valid_transitions(self):
        """REFUNDED não tem saída para nenhuma ação."""
        o = delivered_order(total=Decimal("500.00"), deliver_date=D(2024, 1, 15))
        o.refund(on=D(2024, 1, 20))
        assert o.state == OrderState.REFUNDED
        for action_name, kwargs in [
            ("pay",     {"method": PaymentMethod.PIX, "on": D(2024, 1, 21)}),
            ("pick",    {"on": D(2024, 1, 21)}),
            ("ship",    {"on": D(2024, 1, 21)}),
            ("deliver", {"on": D(2024, 1, 21)}),
            ("cancel",  {"on": D(2024, 1, 21)}),
            ("refund",  {"on": D(2024, 1, 21)}),
        ]:
            with pytest.raises(InvalidTransition):
                getattr(o, action_name)(**kwargs)

    def test_cancelled_without_payment_has_no_valid_refund(self):
        o = make_order(created_on=D(2024, 1, 10))
        o.cancel(on=D(2024, 1, 10))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 1, 10))


# ===========================================================================
# 13. Testes de integração / cenários realistas
# ===========================================================================

class TestIntegrationScenarios:

    def test_scenario_boleto_cancel_within_limit(self):
        """Compra de baixo valor com boleto, cancelada no dia seguinte."""
        o = Order("B-001", Decimal("49.90"), D(2024, 3, 1))
        o.pay(PaymentMethod.BOLETO, on=D(2024, 3, 1))
        o.cancel(on=D(2024, 3, 3))  # 2 dias — no limite
        assert o.state == OrderState.CANCELLED

    def test_scenario_pix_full_refund_fast(self):
        """Compra via PIX, entregue e reembolsada no mesmo dia."""
        o = Order("P-100", Decimal("199.99"), D(2024, 3, 1))
        o.pay(PaymentMethod.PIX, on=D(2024, 3, 2))
        o.pick(on=D(2024, 3, 3))
        o.ship(on=D(2024, 3, 4))
        o.deliver(on=D(2024, 3, 7))
        o.refund(on=D(2024, 3, 7))
        assert o.state == OrderState.REFUNDED

    def test_scenario_expensive_order_fast_cancel(self):
        """Pedido caro: apenas 1 dia para cancelar."""
        o = Order("E-999", Decimal("15000.00"), D(2024, 3, 1))
        o.pay(PaymentMethod.CREDIT_CARD, on=D(2024, 3, 1))
        o.pick(on=D(2024, 3, 1))
        o.cancel(on=D(2024, 3, 2))  # 1 dia desde pagamento — no limite
        assert o.state == OrderState.CANCELLED

    def test_scenario_expensive_order_cancel_too_late(self):
        o = Order("E-999", Decimal("15000.00"), D(2024, 3, 1))
        o.pay(PaymentMethod.CREDIT_CARD, on=D(2024, 3, 1))
        o.pick(on=D(2024, 3, 1))
        with pytest.raises(RuleViolation):
            o.cancel(on=D(2024, 3, 3))  # 2 dias desde pagamento — acima do limite

    def test_scenario_boleto_refund_after_cancel(self):
        """Boleto cancelado → reembolso só após 1 dia do pagamento."""
        o = Order("BL-5", Decimal("80.00"), D(2024, 4, 1))
        o.pay(PaymentMethod.BOLETO, on=D(2024, 4, 2))
        o.cancel(on=D(2024, 4, 3))
        o.refund(on=D(2024, 4, 3))  # 1 dia desde pagamento — OK
        assert o.state == OrderState.REFUNDED

    def test_scenario_refund_on_cc_too_late(self):
        """Entrega com CC, mas reembolso requisitado 8 dias depois."""
        o = Order("CC-7", Decimal("300.00"), D(2024, 5, 1))
        o.pay(PaymentMethod.CREDIT_CARD, on=D(2024, 5, 2))
        o.pick(on=D(2024, 5, 3))
        o.ship(on=D(2024, 5, 4))
        o.deliver(on=D(2024, 5, 10))
        with pytest.raises(RuleViolation):
            o.refund(on=D(2024, 5, 18))  # 8 dias — acima do limite

    def test_scenario_state_property_reflects_each_step(self):
        """Propriedade state reflete fielmente cada transição."""
        o = Order("ORD-SEQ", Decimal("500.00"), D(2024, 1, 1))
        assert o.state == OrderState.CREATED
        o.pay(PaymentMethod.PIX, on=D(2024, 1, 2))
        assert o.state == OrderState.PAID
        o.pick(on=D(2024, 1, 3))
        assert o.state == OrderState.PICKED
        o.ship(on=D(2024, 1, 4))
        assert o.state == OrderState.SHIPPED
        o.deliver(on=D(2024, 1, 5))
        assert o.state == OrderState.DELIVERED
        o.refund(on=D(2024, 1, 10))
        assert o.state == OrderState.REFUNDED

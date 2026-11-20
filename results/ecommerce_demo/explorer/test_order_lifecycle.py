"""
test_order_lifecycle.py
=======================
Suíte de testes para order_lifecycle.py (pytest).

Estratégia (SWEBOK v4):
- Cobertura de transições: todo arco do grafo FSM é percorrido ao menos
  uma vez (State Transition Testing).
- Partição de equivalência: classes válidas e inválidas para cada argumento
  e para cada regra de negócio.
- Análise de valor limite: limites exatos (0, 1, 2, 7 dias; 2000, 5000)
  e valores imediatamente acima/abaixo.
- Ordem de verificação: cada nível da pilha (argumento → transição → regra)
  é verificado de forma isolada para garantir que a ordem correta é mantida.
- Atomicidade: após uma exceção, estado e datas permanecem inalterados.
- Cenários completos: fluxos fim-a-fim cobrem combinações de regras.
"""

import datetime
from decimal import Decimal

import pytest

from order_lifecycle import (
    InvalidOrder,
    InvalidTransition,
    Order,
    OrderState,
    PaymentMethod,
    RuleViolation,
)

# ---------------------------------------------------------------------------
# Factories e datas fixas
# ---------------------------------------------------------------------------

D = datetime.date          # atalho

BASE = D(2024, 1, 1)       # data de criação padrão nos testes


def make_order(
    order_id: str = "ORD-001",
    total: Decimal = Decimal("100.00"),
    created_on: datetime.date = BASE,
) -> Order:
    """Cria um pedido CREATED com defaults seguros."""
    return Order(order_id=order_id, total=total, created_on=created_on)


def paid_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    paid_on: datetime.date = BASE,
    total: Decimal = Decimal("100.00"),
    created_on: datetime.date = BASE,
) -> Order:
    """Pedido já em estado PAID."""
    o = make_order(total=total, created_on=created_on)
    o.pay(method, paid_on)
    return o


def picked_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    paid_on: datetime.date = BASE,
    picked_on: datetime.date = BASE,
    total: Decimal = Decimal("100.00"),
    created_on: datetime.date = BASE,
) -> Order:
    """Pedido já em estado PICKED."""
    o = paid_order(method=method, paid_on=paid_on, total=total, created_on=created_on)
    o.pick(picked_on)
    return o


def shipped_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    paid_on: datetime.date = BASE,
    picked_on: datetime.date = BASE,
    shipped_on: datetime.date = BASE,
    total: Decimal = Decimal("100.00"),
    created_on: datetime.date = BASE,
) -> Order:
    """Pedido já em estado SHIPPED."""
    o = picked_order(
        method=method,
        paid_on=paid_on,
        picked_on=picked_on,
        total=total,
        created_on=created_on,
    )
    o.ship(shipped_on)
    return o


def delivered_order(
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    paid_on: datetime.date = BASE,
    picked_on: datetime.date = BASE,
    shipped_on: datetime.date = BASE,
    delivered_on: datetime.date = BASE,
    total: Decimal = Decimal("100.00"),
    created_on: datetime.date = BASE,
) -> Order:
    """Pedido já em estado DELIVERED."""
    o = shipped_order(
        method=method,
        paid_on=paid_on,
        picked_on=picked_on,
        shipped_on=shipped_on,
        total=total,
        created_on=created_on,
    )
    o.deliver(delivered_on)
    return o


# ===========================================================================
# 1. Validação do construtor
# ===========================================================================

class TestConstructorValidation:

    def test_valid_order_created_state(self):
        o = make_order()
        assert o.state is OrderState.CREATED

    # --- order_id ---

    @pytest.mark.parametrize("bad_id", [
        "",          # vazio
        123,         # não string
        None,        # None
        [],          # lista
    ])
    def test_invalid_order_id(self, bad_id):
        with pytest.raises(InvalidOrder):
            Order(order_id=bad_id, total=Decimal("1.00"), created_on=BASE)

    # --- total ---

    @pytest.mark.parametrize("bad_total", [
        1.0,            # float
        1,              # int
        "1.00",         # string
        None,           # None
    ])
    def test_total_wrong_type(self, bad_total):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=bad_total, created_on=BASE)

    def test_total_zero(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("0"), created_on=BASE)

    def test_total_negative(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("-0.01"), created_on=BASE)

    def test_total_too_many_decimals(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("1.001"), created_on=BASE)

    def test_total_exactly_two_decimals_ok(self):
        o = Order(order_id="X", total=Decimal("1.99"), created_on=BASE)
        assert o.state is OrderState.CREATED

    def test_total_one_decimal_ok(self):
        o = Order(order_id="X", total=Decimal("1.5"), created_on=BASE)
        assert o.state is OrderState.CREATED

    def test_total_integer_decimal_ok(self):
        # Decimal("100") tem exponent 0 — válido
        o = Order(order_id="X", total=Decimal("100"), created_on=BASE)
        assert o.state is OrderState.CREATED

    # --- created_on ---

    def test_created_on_datetime_rejected(self):
        with pytest.raises(InvalidOrder):
            Order(
                order_id="X",
                total=Decimal("1.00"),
                created_on=datetime.datetime(2024, 1, 1),
            )

    def test_created_on_wrong_type(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("1.00"), created_on="2024-01-01")

    def test_created_on_none(self):
        with pytest.raises(InvalidOrder):
            Order(order_id="X", total=Decimal("1.00"), created_on=None)


# ===========================================================================
# 2. Validação de argumentos nas ações
# ===========================================================================

class TestActionArgumentValidation:

    # --- on deve ser date, não datetime ---

    def test_pay_datetime_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, datetime.datetime(2024, 1, 1))

    def test_pick_datetime_rejected(self):
        o = paid_order()
        with pytest.raises(InvalidOrder):
            o.pick(datetime.datetime(2024, 1, 2))

    def test_ship_datetime_rejected(self):
        o = picked_order()
        with pytest.raises(InvalidOrder):
            o.ship(datetime.datetime(2024, 1, 2))

    def test_deliver_datetime_rejected(self):
        o = shipped_order()
        with pytest.raises(InvalidOrder):
            o.deliver(datetime.datetime(2024, 1, 2))

    def test_cancel_datetime_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.cancel(datetime.datetime(2024, 1, 1))

    def test_refund_datetime_rejected(self):
        o = delivered_order()
        with pytest.raises(InvalidOrder):
            o.refund(datetime.datetime(2024, 1, 1))

    # --- on não pode ser anterior ao último evento ---

    def test_pay_date_before_creation(self):
        o = make_order(created_on=D(2024, 1, 10))
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, D(2024, 1, 9))

    def test_pick_date_before_payment(self):
        o = paid_order(paid_on=D(2024, 1, 5))
        with pytest.raises(InvalidOrder):
            o.pick(D(2024, 1, 4))

    def test_on_same_as_last_event_ok(self):
        o = make_order(created_on=BASE)
        o.pay(PaymentMethod.PIX, BASE)   # mesmo dia — deve passar
        assert o.state is OrderState.PAID

    # --- method para pay ---

    def test_pay_invalid_method_type(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay("CREDIT_CARD", BASE)   # string, não enum

    def test_pay_none_method(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(None, BASE)

    # --- Atomicidade: exceção não muda o estado ---

    def test_atomicity_on_invalid_date(self):
        o = make_order(created_on=D(2024, 1, 10))
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, D(2024, 1, 9))
        assert o.state is OrderState.CREATED  # estado intacto

    def test_atomicity_on_invalid_method(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay("PIX", BASE)
        assert o.state is OrderState.CREATED


# ===========================================================================
# 3. Transições válidas (happy path — cobertura de todos os arcos)
# ===========================================================================

class TestValidTransitions:

    def test_created_to_paid(self):
        o = make_order()
        o.pay(PaymentMethod.CREDIT_CARD, BASE)
        assert o.state is OrderState.PAID

    def test_created_to_cancelled(self):
        o = make_order()
        o.cancel(BASE)
        assert o.state is OrderState.CANCELLED

    def test_paid_to_picked(self):
        o = paid_order()
        o.pick(BASE)
        assert o.state is OrderState.PICKED

    def test_paid_to_cancelled_within_limit(self):
        o = paid_order(paid_on=BASE)
        o.cancel(BASE)   # 0 dias <= 7 (CREDIT_CARD)
        assert o.state is OrderState.CANCELLED

    def test_picked_to_shipped(self):
        o = picked_order()
        o.ship(BASE)
        assert o.state is OrderState.SHIPPED

    def test_picked_to_cancelled_within_limit(self):
        o = picked_order(paid_on=BASE)
        o.cancel(BASE)   # 0 dias <= 7 (CREDIT_CARD)
        assert o.state is OrderState.CANCELLED

    def test_shipped_to_delivered(self):
        o = shipped_order()
        o.deliver(BASE)
        assert o.state is OrderState.DELIVERED

    def test_delivered_to_refunded(self):
        o = delivered_order()
        o.refund(BASE)   # 0 dias, CREDIT_CARD, total 100 <= 2000
        assert o.state is OrderState.REFUNDED

    def test_cancelled_to_refunded_paid_first(self):
        o = paid_order(paid_on=BASE)
        o.cancel(BASE)
        o.refund(BASE)   # pago antes, não BOLETO
        assert o.state is OrderState.REFUNDED

    def test_full_happy_path(self):
        o = make_order()
        o.pay(PaymentMethod.PIX, BASE)
        o.pick(BASE)
        o.ship(BASE)
        o.deliver(BASE)
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED

    def test_all_payment_methods_allowed_in_pay(self):
        for method in PaymentMethod:
            o = make_order()
            o.pay(method, BASE)
            assert o.state is OrderState.PAID


# ===========================================================================
# 4. Transições inválidas (InvalidTransition)
# ===========================================================================

class TestInvalidTransitions:
    """
    Cada linha da tabela de transições proíbe a maioria das ações.
    Verificamos as combinações inválidas mais relevantes.
    """

    # Estado CREATED
    def test_created_pick(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.pick(BASE)

    def test_created_ship(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.ship(BASE)

    def test_created_deliver(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.deliver(BASE)

    def test_created_refund(self):
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.refund(BASE)

    def test_created_pay_twice(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, BASE)

    # Estado PAID
    def test_paid_ship(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.ship(BASE)

    def test_paid_deliver(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.deliver(BASE)

    def test_paid_refund(self):
        o = paid_order()
        with pytest.raises(InvalidTransition):
            o.refund(BASE)

    # Estado PICKED
    def test_picked_pay(self):
        o = picked_order()
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, BASE)

    def test_picked_deliver(self):
        o = picked_order()
        with pytest.raises(InvalidTransition):
            o.deliver(BASE)

    def test_picked_refund(self):
        o = picked_order()
        with pytest.raises(InvalidTransition):
            o.refund(BASE)

    # Estado SHIPPED
    def test_shipped_pay(self):
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, BASE)

    def test_shipped_pick(self):
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.pick(BASE)

    def test_shipped_cancel(self):
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.cancel(BASE)

    def test_shipped_refund(self):
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.refund(BASE)

    # Estado DELIVERED
    def test_delivered_pay(self):
        o = delivered_order()
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, BASE)

    def test_delivered_cancel(self):
        o = delivered_order()
        with pytest.raises(InvalidTransition):
            o.cancel(BASE)

    # Estado CANCELLED
    def test_cancelled_pay(self):
        o = make_order()
        o.cancel(BASE)
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, BASE)

    def test_cancelled_pick(self):
        o = make_order()
        o.cancel(BASE)
        with pytest.raises(InvalidTransition):
            o.pick(BASE)

    def test_cancelled_ship(self):
        o = make_order()
        o.cancel(BASE)
        with pytest.raises(InvalidTransition):
            o.ship(BASE)

    def test_cancelled_deliver(self):
        o = make_order()
        o.cancel(BASE)
        with pytest.raises(InvalidTransition):
            o.deliver(BASE)

    # Estado REFUNDED (terminal — nenhuma ação)
    def test_refunded_pay(self):
        o = delivered_order()
        o.refund(BASE)
        with pytest.raises(InvalidTransition):
            o.pay(PaymentMethod.PIX, BASE)

    def test_refunded_cancel(self):
        o = delivered_order()
        o.refund(BASE)
        with pytest.raises(InvalidTransition):
            o.cancel(BASE)

    def test_refunded_refund(self):
        o = delivered_order()
        o.refund(BASE)
        with pytest.raises(InvalidTransition):
            o.refund(BASE)


# ===========================================================================
# 5. Regra C — cancelar pedido pago/picked
# ===========================================================================

class TestRuleC:
    """
    Limites (valor limite + partição de equivalência):
      - total > 5000  → limite 1 dia
      - BOLETO        → limite 2 dias
      - demais        → limite 7 dias
    """

    # --- CREDIT_CARD / PIX: limite 7 dias ---

    @pytest.mark.parametrize("method", [
        PaymentMethod.CREDIT_CARD, PaymentMethod.PIX
    ])
    def test_cancel_day_0_allowed(self, method):
        o = paid_order(method=method, paid_on=D(2024, 1, 1))
        o.cancel(D(2024, 1, 1))
        assert o.state is OrderState.CANCELLED

    @pytest.mark.parametrize("method", [
        PaymentMethod.CREDIT_CARD, PaymentMethod.PIX
    ])
    def test_cancel_day_7_allowed(self, method):
        o = paid_order(method=method, paid_on=D(2024, 1, 1))
        o.cancel(D(2024, 1, 8))   # 7 dias
        assert o.state is OrderState.CANCELLED

    @pytest.mark.parametrize("method", [
        PaymentMethod.CREDIT_CARD, PaymentMethod.PIX
    ])
    def test_cancel_day_8_forbidden(self, method):
        o = paid_order(method=method, paid_on=D(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 1, 9))   # 8 dias

    # --- BOLETO: limite 2 dias ---

    def test_cancel_boleto_day_2_allowed(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=D(2024, 1, 1))
        o.cancel(D(2024, 1, 3))   # 2 dias
        assert o.state is OrderState.CANCELLED

    def test_cancel_boleto_day_3_forbidden(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=D(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 1, 4))   # 3 dias

    # --- total > 5000: limite 1 dia (qualquer pagamento) ---

    @pytest.mark.parametrize("method", list(PaymentMethod))
    def test_cancel_high_value_day_1_allowed(self, method):
        o = paid_order(
            method=method,
            paid_on=D(2024, 1, 1),
            total=Decimal("5000.01"),
        )
        o.cancel(D(2024, 1, 2))   # 1 dia
        assert o.state is OrderState.CANCELLED

    @pytest.mark.parametrize("method", list(PaymentMethod))
    def test_cancel_high_value_day_2_forbidden(self, method):
        o = paid_order(
            method=method,
            paid_on=D(2024, 1, 1),
            total=Decimal("5000.01"),
        )
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 1, 3))   # 2 dias

    def test_cancel_exactly_5000_uses_boleto_limit(self):
        # 5000.00 NÃO é maior que 5000; método BOLETO → limite 2
        o = paid_order(
            method=PaymentMethod.BOLETO,
            paid_on=D(2024, 1, 1),
            total=Decimal("5000.00"),
        )
        o.cancel(D(2024, 1, 3))   # 2 dias — OK
        assert o.state is OrderState.CANCELLED

    def test_cancel_exactly_5000_non_boleto_uses_7_limit(self):
        o = paid_order(
            method=PaymentMethod.CREDIT_CARD,
            paid_on=D(2024, 1, 1),
            total=Decimal("5000.00"),
        )
        o.cancel(D(2024, 1, 8))   # 7 dias — OK
        assert o.state is OrderState.CANCELLED

    # --- Regra C a partir de PICKED ---

    def test_cancel_from_picked_within_limit(self):
        o = picked_order(
            method=PaymentMethod.CREDIT_CARD,
            paid_on=D(2024, 1, 1),
            picked_on=D(2024, 1, 1),
        )
        o.cancel(D(2024, 1, 5))   # 4 dias desde pagamento
        assert o.state is OrderState.CANCELLED

    def test_cancel_from_picked_exceeds_limit(self):
        o = picked_order(
            method=PaymentMethod.CREDIT_CARD,
            paid_on=D(2024, 1, 1),
            picked_on=D(2024, 1, 3),
        )
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 1, 10))  # 9 dias desde pagamento

    # --- Atomicidade após RuleViolation na Regra C ---

    def test_rule_c_violation_does_not_change_state(self):
        o = paid_order(
            method=PaymentMethod.CREDIT_CARD,
            paid_on=D(2024, 1, 1),
        )
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 1, 10))
        assert o.state is OrderState.PAID


# ===========================================================================
# 6. Regra R — reembolso de pedido entregue
# ===========================================================================

class TestRuleR:
    """
    As três condições devem valer simultaneamente.
    """

    def _make_delivered(
        self,
        method: PaymentMethod = PaymentMethod.CREDIT_CARD,
        total: Decimal = Decimal("100.00"),
        delivered_on: datetime.date = BASE,
    ) -> Order:
        return delivered_order(
            method=method,
            total=total,
            delivered_on=delivered_on,
        )

    # --- Condição 1: forma de pagamento ---

    def test_refund_credit_card_allowed(self):
        o = self._make_delivered(method=PaymentMethod.CREDIT_CARD)
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED

    def test_refund_pix_allowed(self):
        o = self._make_delivered(method=PaymentMethod.PIX)
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED

    def test_refund_boleto_forbidden(self):
        o = self._make_delivered(method=PaymentMethod.BOLETO)
        with pytest.raises(RuleViolation):
            o.refund(BASE)

    # --- Condição 2: prazo 7 dias ---

    def test_refund_day_7_allowed(self):
        delivered = D(2024, 1, 1)
        o = self._make_delivered(delivered_on=delivered)
        o.refund(D(2024, 1, 8))   # 7 dias
        assert o.state is OrderState.REFUNDED

    def test_refund_day_8_forbidden(self):
        delivered = D(2024, 1, 1)
        o = self._make_delivered(delivered_on=delivered)
        with pytest.raises(RuleViolation):
            o.refund(D(2024, 1, 9))   # 8 dias

    def test_refund_day_0_allowed(self):
        delivered = D(2024, 1, 5)
        o = self._make_delivered(delivered_on=delivered)
        o.refund(D(2024, 1, 5))   # 0 dias
        assert o.state is OrderState.REFUNDED

    # --- Condição 3: total <= 2000 ---

    def test_refund_total_2000_allowed(self):
        o = self._make_delivered(total=Decimal("2000.00"))
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED

    def test_refund_total_above_2000_forbidden(self):
        o = self._make_delivered(total=Decimal("2000.01"))
        with pytest.raises(RuleViolation):
            o.refund(BASE)

    # --- Múltiplas condições violadas simultaneamente ---

    def test_refund_boleto_and_late_and_high(self):
        delivered = D(2024, 1, 1)
        o = self._make_delivered(
            method=PaymentMethod.BOLETO,
            total=Decimal("3000.00"),
            delivered_on=delivered,
        )
        with pytest.raises(RuleViolation):
            o.refund(D(2024, 1, 10))

    # --- Atomicidade ---

    def test_rule_r_violation_does_not_change_state(self):
        o = self._make_delivered(method=PaymentMethod.BOLETO)
        with pytest.raises(RuleViolation):
            o.refund(BASE)
        assert o.state is OrderState.DELIVERED


# ===========================================================================
# 7. Regra R2 — reembolso de pedido cancelado
# ===========================================================================

class TestRuleR2:

    # --- Cancelado sem pagamento ---

    def test_refund_cancelled_never_paid_forbidden(self):
        o = make_order()
        o.cancel(BASE)
        with pytest.raises(RuleViolation):
            o.refund(BASE)

    # --- Cancelado após pagamento (não BOLETO) ---

    def test_refund_cancelled_after_payment_allowed(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, paid_on=BASE)
        o.cancel(BASE)
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED

    def test_refund_cancelled_pix_allowed(self):
        o = paid_order(method=PaymentMethod.PIX, paid_on=BASE)
        o.cancel(BASE)
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED

    # --- BOLETO: exige pelo menos 1 dia desde o pagamento ---

    def test_refund_cancelled_boleto_day_0_forbidden(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=D(2024, 1, 1))
        o.cancel(D(2024, 1, 1))
        with pytest.raises(RuleViolation):
            o.refund(D(2024, 1, 1))   # 0 dias desde pagamento

    def test_refund_cancelled_boleto_day_1_allowed(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=D(2024, 1, 1))
        o.cancel(D(2024, 1, 1))
        o.refund(D(2024, 1, 2))   # 1 dia desde pagamento
        assert o.state is OrderState.REFUNDED

    def test_refund_cancelled_boleto_day_7_allowed(self):
        o = paid_order(method=PaymentMethod.BOLETO, paid_on=D(2024, 1, 1))
        o.cancel(D(2024, 1, 2))   # 1 dia — dentro do limite C (2)
        o.refund(D(2024, 1, 8))   # 7 dias desde pagamento — OK
        assert o.state is OrderState.REFUNDED

    # --- Atomicidade ---

    def test_rule_r2_violation_does_not_change_state(self):
        o = make_order()
        o.cancel(BASE)
        with pytest.raises(RuleViolation):
            o.refund(BASE)
        assert o.state is OrderState.CANCELLED


# ===========================================================================
# 8. Ordem de verificação (argumento → transição → regra)
# ===========================================================================

class TestVerificationOrder:
    """
    Garante que a ordem do contrato é respeitada:
    argumento inválido sobrepõe transição inválida;
    transição inválida sobrepõe regra.
    """

    def test_invalid_date_type_takes_priority_over_invalid_transition(self):
        # Em SHIPPED, cancel não existe.  Mas o 'on' é datetime.datetime.
        # Deve levantar InvalidOrder (argumento), não InvalidTransition.
        o = shipped_order()
        with pytest.raises(InvalidOrder):
            o.cancel(datetime.datetime(2024, 1, 2))

    def test_date_before_last_event_takes_priority_over_invalid_transition(self):
        # Em SHIPPED, cancel não existe. Mas 'on' é anterior ao último evento.
        # Deve levantar InvalidOrder (data anterior), não InvalidTransition.
        o = shipped_order(shipped_on=D(2024, 6, 1))
        with pytest.raises(InvalidOrder):
            o.cancel(D(2024, 1, 1))   # antes do último evento

    def test_invalid_transition_takes_priority_over_rule(self):
        # Em PAID, 'deliver' não existe (InvalidTransition).
        # Deve levantar InvalidTransition, não RuleViolation.
        # (Regra R exigiria estado DELIVERED — se chegasse lá.)
        o = paid_order(paid_on=BASE)
        with pytest.raises(InvalidTransition):
            o.deliver(BASE)

    def test_invalid_transition_on_shipped_cancel(self):
        # SHIPPED não tem 'cancel' — InvalidTransition mesmo com args válidos.
        o = shipped_order()
        with pytest.raises(InvalidTransition):
            o.cancel(BASE)

    def test_invalid_method_type_before_invalid_transition(self):
        # Em PAID, pay não existe (InvalidTransition), mas method inválido
        # deve ser detectado antes (InvalidOrder).
        o = paid_order()
        with pytest.raises(InvalidOrder):
            o.pay("PIX", BASE)

    def test_invalid_date_before_rule_violation(self):
        # Regra C seria violada (8 dias), mas o 'on' é datetime → InvalidOrder primeiro.
        o = paid_order(paid_on=D(2024, 1, 1))
        with pytest.raises(InvalidOrder):
            o.cancel(datetime.datetime(2024, 1, 9))

    def test_invalid_transition_refunded_terminal(self):
        # REFUNDED é terminal — qualquer ação levanta InvalidTransition.
        o = delivered_order()
        o.refund(BASE)
        assert o.state is OrderState.REFUNDED
        with pytest.raises(InvalidTransition):
            o.pick(BASE)

    def test_invalid_transition_on_created_no_rule(self):
        # CREATED não tem 'pick' — InvalidTransition puro (sem regra envolvida).
        o = make_order()
        with pytest.raises(InvalidTransition):
            o.pick(BASE)


# ===========================================================================
# 9. Cenários end-to-end
# ===========================================================================

class TestEndToEndScenarios:

    def test_full_lifecycle_pix(self):
        o = Order("E2E-001", Decimal("500.00"), D(2024, 3, 1))
        assert o.state is OrderState.CREATED
        o.pay(PaymentMethod.PIX, D(2024, 3, 1))
        assert o.state is OrderState.PAID
        o.pick(D(2024, 3, 2))
        assert o.state is OrderState.PICKED
        o.ship(D(2024, 3, 3))
        assert o.state is OrderState.SHIPPED
        o.deliver(D(2024, 3, 7))
        assert o.state is OrderState.DELIVERED
        o.refund(D(2024, 3, 10))   # 3 dias após entrega, dentro do limite
        assert o.state is OrderState.REFUNDED

    def test_cancel_before_ship_boleto_day_1(self):
        o = Order("E2E-002", Decimal("300.00"), D(2024, 3, 1))
        o.pay(PaymentMethod.BOLETO, D(2024, 3, 1))
        o.pick(D(2024, 3, 2))
        o.cancel(D(2024, 3, 2))   # 1 dia — OK para BOLETO (limite 2)
        assert o.state is OrderState.CANCELLED

    def test_refund_cancelled_boleto_scenario(self):
        o = Order("E2E-003", Decimal("150.00"), D(2024, 3, 1))
        o.pay(PaymentMethod.BOLETO, D(2024, 3, 1))
        o.cancel(D(2024, 3, 2))   # 1 dia — OK
        o.refund(D(2024, 3, 2))   # 1 dia desde pagamento — OK para R2
        assert o.state is OrderState.REFUNDED

    def test_high_value_order_strict_cancel_window(self):
        o = Order("E2E-004", Decimal("10000.00"), D(2024, 3, 1))
        o.pay(PaymentMethod.CREDIT_CARD, D(2024, 3, 1))
        o.cancel(D(2024, 3, 2))   # 1 dia — OK (limite 1 para >5000)
        assert o.state is OrderState.CANCELLED

    def test_high_value_order_cannot_cancel_after_1_day(self):
        o = Order("E2E-005", Decimal("10000.00"), D(2024, 3, 1))
        o.pay(PaymentMethod.CREDIT_CARD, D(2024, 3, 1))
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 3, 3))  # 2 dias — limite é 1

    def test_cannot_refund_delivered_high_value(self):
        o = Order("E2E-006", Decimal("2000.01"), D(2024, 3, 1))
        o.pay(PaymentMethod.CREDIT_CARD, D(2024, 3, 1))
        o.pick(D(2024, 3, 1))
        o.ship(D(2024, 3, 1))
        o.deliver(D(2024, 3, 1))
        with pytest.raises(RuleViolation):
            o.refund(D(2024, 3, 1))

    def test_multiple_orders_independent(self):
        """Duas ordens não compartilham estado."""
        o1 = make_order(order_id="A")
        o2 = make_order(order_id="B")
        o1.pay(PaymentMethod.PIX, BASE)
        assert o1.state is OrderState.PAID
        assert o2.state is OrderState.CREATED

    def test_dates_monotonically_increase_across_lifecycle(self):
        o = Order("MONO", Decimal("99.99"), D(2024, 1, 1))
        o.pay(PaymentMethod.PIX, D(2024, 1, 2))
        o.pick(D(2024, 1, 3))
        o.ship(D(2024, 1, 4))
        o.deliver(D(2024, 1, 5))
        o.refund(D(2024, 1, 6))
        assert o.state is OrderState.REFUNDED

    def test_cancel_created_no_rule_needed(self):
        """Cancela direto do CREATED — sem regra C, sempre permitido."""
        o = make_order()
        o.cancel(BASE)
        assert o.state is OrderState.CANCELLED

    def test_cancel_paid_at_exact_boleto_boundary(self):
        """BOLETO: cancela no dia 2 exato (limite) — deve passar."""
        o = Order("BND", Decimal("499.99"), D(2024, 5, 1))
        o.pay(PaymentMethod.BOLETO, D(2024, 5, 1))
        o.cancel(D(2024, 5, 3))   # 2 dias
        assert o.state is OrderState.CANCELLED

    def test_cancel_paid_just_over_boleto_boundary(self):
        """BOLETO: cancela no dia 3 — um a mais do limite → RuleViolation."""
        o = Order("BND2", Decimal("499.99"), D(2024, 5, 1))
        o.pay(PaymentMethod.BOLETO, D(2024, 5, 1))
        with pytest.raises(RuleViolation):
            o.cancel(D(2024, 5, 4))   # 3 dias

    def test_refund_delivered_on_last_valid_day(self):
        """Regra R: reembolso no sétimo dia exato."""
        o = Order("R7", Decimal("1000.00"), D(2024, 2, 1))
        o.pay(PaymentMethod.PIX, D(2024, 2, 1))
        o.pick(D(2024, 2, 1))
        o.ship(D(2024, 2, 1))
        o.deliver(D(2024, 2, 1))
        o.refund(D(2024, 2, 8))   # 7 dias — no limite
        assert o.state is OrderState.REFUNDED

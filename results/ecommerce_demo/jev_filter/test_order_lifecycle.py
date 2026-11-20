"""
test_order_lifecycle.py
=======================
Suíte pytest para order_lifecycle.

Estratégia (guiada pelo SWEBOK):

* **State-transition testing** – cobre cada aresta da tabela de transições,
  incluindo todos os caminhos felizes e todas as transições inválidas.
* **Boundary Value Analysis** (SWEBOK – BVA): os valores de limite das regras
  C, R e R2 são testados em n-1, n e n+1 dias, e nos limiares de total
  (5000,00 / 2000,00).
* **Combinação de regras**: todas as combinações de forma de pagamento,
  total e dias relevantes para as regras C e R são exercitadas.
* **Ordem das verificações**: testes que garantem que InvalidOrder vem antes
  de InvalidTransition, que por sua vez vem antes de RuleViolation.
* **Atomicidade**: uma chamada que falha NÃO deve mudar o estado do pedido.
"""

import pytest
from datetime import date, datetime, timedelta
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


# ============================================================
# Helpers
# ============================================================

D = date  # alias curto


def d(offset: int = 0) -> date:
    """Data-base conveniente: 2024-01-01 + offset dias."""
    return D(2024, 1, 1) + timedelta(days=offset)


def make_order(
    total: Decimal = Decimal("100.00"),
    order_id: str = "ORD-1",
    created_on: date = None,
) -> Order:
    return Order(order_id, total, created_on or d(0))


def paid_order(
    total: Decimal = Decimal("100.00"),
    method: PaymentMethod = PaymentMethod.CREDIT_CARD,
    pay_offset: int = 1,
) -> Order:
    o = make_order(total=total)
    o.pay(method, d(pay_offset))
    return o


def picked_order(**kw) -> Order:
    o = paid_order(**kw)
    o.pick(d(2))
    return o


def shipped_order(**kw) -> Order:
    o = picked_order(**kw)
    o.ship(d(3))
    return o


def delivered_order(**kw) -> Order:
    o = shipped_order(**kw)
    o.deliver(d(4))
    return o


# ============================================================
# 1. Validação do construtor
# ============================================================

class TestOrderCreation:

    def test_valid_creation(self):
        o = Order("ORD-1", Decimal("1.00"), d())
        assert o.state is OrderState.CREATED

    # --- order_id ---

    @pytest.mark.parametrize("bad_id", [
        "", 123, None, 3.14, b"id",
    ])
    def test_invalid_order_id(self, bad_id):
        with pytest.raises(InvalidOrder):
            Order(bad_id, Decimal("1.00"), d())

    # --- total ---

    @pytest.mark.parametrize("bad_total", [
        1,          # int
        1.0,        # float
        "1.00",     # str
        None,
    ])
    def test_total_wrong_type(self, bad_total):
        with pytest.raises(InvalidOrder):
            Order("ORD-1", bad_total, d())

    def test_total_zero(self):
        with pytest.raises(InvalidOrder):
            Order("ORD-1", Decimal("0"), d())

    def test_total_negative(self):
        with pytest.raises(InvalidOrder):
            Order("ORD-1", Decimal("-0.01"), d())

    def test_total_too_many_decimals(self):
        with pytest.raises(InvalidOrder):
            Order("ORD-1", Decimal("1.001"), d())

    def test_total_two_decimals_ok(self):
        o = Order("ORD-1", Decimal("9999.99"), d())
        assert o.state is OrderState.CREATED

    def test_total_one_decimal_ok(self):
        o = Order("ORD-1", Decimal("50.5"), d())
        assert o.state is OrderState.CREATED

    def test_total_whole_decimal_ok(self):
        o = Order("ORD-1", Decimal("100"), d())
        assert o.state is OrderState.CREATED

    # --- created_on ---

    def test_created_on_datetime_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("ORD-1", Decimal("1.00"), datetime(2024, 1, 1))

    def test_created_on_string_rejected(self):
        with pytest.raises(InvalidOrder):
            Order("ORD-1", Decimal("1.00"), "2024-01-01")


# ============================================================
# 2. Validação de argumentos nas ações
# ============================================================

class TestActionArgumentValidation:

    # --- tipo de 'on' ---

    def test_pay_datetime_on_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, datetime(2024, 1, 2))

    def test_pick_datetime_on_rejected(self):
        o = paid_order()
        with pytest.raises(InvalidOrder):
            o.pick(datetime(2024, 1, 2))

    def test_ship_datetime_on_rejected(self):
        o = picked_order()
        with pytest.raises(InvalidOrder):
            o.ship(datetime(2024, 1, 4))

    def test_deliver_datetime_on_rejected(self):
        o = shipped_order()
        with pytest.raises(InvalidOrder):
            o.deliver(datetime(2024, 1, 5))

    def test_cancel_datetime_on_rejected(self):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.cancel(datetime(2024, 1, 2))

    def test_refund_datetime_on_rejected(self):
        o = delivered_order()
        with pytest.raises(InvalidOrder):
            o.refund(datetime(2024, 1, 12))

    # --- método de pagamento ---

    @pytest.mark.parametrize("bad_method", [
        "CREDIT_CARD", 1, None,
    ])
    def test_pay_bad_method_type(self, bad_method):
        o = make_order()
        with pytest.raises(InvalidOrder):
            o.pay(bad_method, d(1))

    # --- data anterior ao último evento ---

    def test_pay_before_created_on(self):
        o = make_order()           # criado em d(0)
        with pytest.raises(InvalidOrder):
            o.pay(PaymentMethod.PIX, d(-1))

    def test_pick_before_paid_on(self):
        o = paid_order(pay_offset=5)   # pago em d(5)
        with pytest.raises(InvalidOrder):
            o.pick(d(4))

    def test_ship_before_picked_on(self):
        o = picked_order()         # picked em d(2)
        with pytest.raises(InvalidOrder):
            o.ship(d(1))

    def test_deliver_before_shipped_on(self):
        o = shipped_order()        # shipped em d(3)
        with pytest.raises(InvalidOrder):
            o.deliver(d(2))

    def test_cancel_before_last_event(self):
        o = paid_order(pay_offset=3)   # paid em d(3)
        with pytest.raises(InvalidOrder):
            o.cancel(d(2))

    def test_same_day_actions_allowed(self):
        """on == last_event_date é aceito."""
        o = make_order()
        o.pay(PaymentMethod.PIX, d(0))   # mesmo dia da criação
        assert o.state is OrderState.PAID

    # --- state não alterado quando há InvalidOrder ---

    def test_state_unchanged_on_invalid_order(self):
        o = make_order()
        try:
            o.pay(PaymentMethod.PIX, datetime(2024, 1, 2))
        except InvalidOrder:
            pass
        assert o.state is OrderState.CREATED


# ============================================================
# 3. Transições válidas (caminhos felizes)
# ============================================================

class TestHappyPaths:

    def test_full_lifecycle_credit_card(self):
        o = Order("ORD-1", Decimal("500.00"), d(0))
        o.pay(PaymentMethod.CREDIT_CARD, d(1))
        assert o.state is OrderState.PAID
        o.pick(d(2))
        assert o.state is OrderState.PICKED
        o.ship(d(3))
        assert o.state is OrderState.SHIPPED
        o.deliver(d(4))
        assert o.state is OrderState.DELIVERED
        o.refund(d(5))
        assert o.state is OrderState.REFUNDED

    def test_full_lifecycle_pix_low_total(self):
        o = Order("ORD-2", Decimal("99.99"), d(0))
        o.pay(PaymentMethod.PIX, d(1))
        o.pick(d(2))
        o.ship(d(3))
        o.deliver(d(4))
        o.refund(d(11))   # 7 dias após entrega = limite exato
        assert o.state is OrderState.REFUNDED

    def test_cancel_from_created(self):
        o = make_order()
        o.cancel(d(1))
        assert o.state is OrderState.CANCELLED

    def test_cancel_from_paid(self):
        o = paid_order()
        o.cancel(d(2))
        assert o.state is OrderState.CANCELLED

    def test_cancel_from_picked(self):
        o = picked_order()
        o.cancel(d(3))
        assert o.state is OrderState.CANCELLED

    def test_refund_from_cancelled_after_pay(self):
        o = paid_order()
        o.cancel(d(2))
        o.refund(d(3))
        assert o.state is OrderState.REFUNDED


# ============================================================
# 4. Transições inválidas (InvalidTransition)
# ============================================================

class TestInvalidTransitions:

    # De CREATED
    @pytest.mark.parametrize("action", ["pick", "ship", "deliver", "refund"])
    def test_created_invalid_actions(self, action):
        o = make_order()
        with pytest.raises(InvalidTransition):
            getattr(o, action)(d(1))

    # De PAID
    @pytest.mark.parametrize("action", ["pay", "ship", "deliver", "refund"])
    def test_paid_invalid_actions(self, action):
        o = paid_order()
        if action == "pay":
            with pytest.raises(InvalidTransition):
                o.pay(PaymentMethod.PIX, d(2))
        else:
            with pytest.raises(InvalidTransition):
                getattr(o, action)(d(2))

    # De PICKED
    @pytest.mark.parametrize("action", ["pay", "pick", "deliver", "refund"])
    def test_picked_invalid_actions(self, action):
        o = picked_order()
        if action == "pay":
            with pytest.raises(InvalidTransition):
                o.pay(PaymentMethod.PIX, d(3))
        else:
            with pytest.raises(InvalidTransition):
                getattr(o, action)(d(3))

    # De SHIPPED
    @pytest.mark.parametrize("action", ["pay", "pick", "ship", "cancel", "refund"])
    def test_shipped_invalid_actions(self, action):
        o = shipped_order()
        if action == "pay":
            with pytest.raises(InvalidTransition):
                o.pay(PaymentMethod.PIX, d(4))
        else:
            with pytest.raises(InvalidTransition):
                getattr(o, action)(d(4))

    # De DELIVERED
    @pytest.mark.parametrize("action", ["pay", "pick", "ship", "deliver", "cancel"])
    def test_delivered_invalid_actions(self, action):
        o = delivered_order()
        if action == "pay":
            with pytest.raises(InvalidTransition):
                o.pay(PaymentMethod.PIX, d(5))
        else:
            with pytest.raises(InvalidTransition):
                getattr(o, action)(d(5))

    # De CANCELLED
    @pytest.mark.parametrize("action", ["pay", "pick", "ship", "deliver", "cancel"])
    def test_cancelled_invalid_actions(self, action):
        o = make_order()
        o.cancel(d(1))
        if action == "pay":
            with pytest.raises(InvalidTransition):
                o.pay(PaymentMethod.PIX, d(2))
        else:
            with pytest.raises(InvalidTransition):
                getattr(o, action)(d(2))

    # De REFUNDED – absolutamente nenhuma ação disponível
    @pytest.mark.parametrize("action", ["pay", "pick", "ship", "deliver", "cancel", "refund"])
    def test_refunded_no_actions(self, action):
        o = delivered_order(total=Decimal("500.00"))
        o.refund(d(5))
        assert o.state is OrderState.REFUNDED
        if action == "pay":
            with pytest.raises(InvalidTransition):
                o.pay(PaymentMethod.PIX, d(6))
        else:
            with pytest.raises(InvalidTransition):
                getattr(o, action)(d(6))

    def test_state_unchanged_on_invalid_transition(self):
        o = make_order()
        try:
            o.pick(d(1))
        except InvalidTransition:
            pass
        assert o.state is OrderState.CREATED


# ============================================================
# 5. Regra C – cancelar pedido pago/picked
# ============================================================

class TestRuleC:
    """BVA nos limiares de dias e de total."""

    # ---- CREDIT_CARD, total <= 5000 → limite 7 dias ----

    def test_cc_7_days_ok(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_offset=1)
        o.cancel(d(8))   # 7 dias
        assert o.state is OrderState.CANCELLED

    def test_cc_8_days_fail(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_offset=1)
        with pytest.raises(RuleViolation):
            o.cancel(d(9))   # 8 dias

    def test_cc_0_days_ok(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_offset=1)
        o.cancel(d(1))   # 0 dias
        assert o.state is OrderState.CANCELLED

    # ---- PIX, total <= 5000 → limite 7 dias ----

    def test_pix_7_days_ok(self):
        o = paid_order(method=PaymentMethod.PIX, pay_offset=0)
        o.cancel(d(7))
        assert o.state is OrderState.CANCELLED

    def test_pix_8_days_fail(self):
        o = paid_order(method=PaymentMethod.PIX, pay_offset=0)
        with pytest.raises(RuleViolation):
            o.cancel(d(8))

    # ---- BOLETO, total <= 5000 → limite 2 dias ----

    def test_boleto_2_days_ok(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_offset=1)
        o.cancel(d(3))   # 2 dias
        assert o.state is OrderState.CANCELLED

    def test_boleto_3_days_fail(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_offset=1)
        with pytest.raises(RuleViolation):
            o.cancel(d(4))   # 3 dias

    def test_boleto_1_day_ok(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_offset=0)
        o.cancel(d(1))
        assert o.state is OrderState.CANCELLED

    # ---- Total > 5000 → limite 1 dia (qualquer pagamento) ----

    @pytest.mark.parametrize("method", list(PaymentMethod))
    def test_high_total_1_day_ok(self, method):
        o = paid_order(total=Decimal("5000.01"), method=method, pay_offset=0)
        o.cancel(d(1))
        assert o.state is OrderState.CANCELLED

    @pytest.mark.parametrize("method", list(PaymentMethod))
    def test_high_total_2_days_fail(self, method):
        o = paid_order(total=Decimal("5000.01"), method=method, pay_offset=0)
        with pytest.raises(RuleViolation):
            o.cancel(d(2))

    def test_exactly_5000_boleto_limit_is_2(self):
        """total == 5000,00 não entra na faixa > 5000; usa limite BOLETO (2 dias)."""
        o = paid_order(total=Decimal("5000.00"), method=PaymentMethod.BOLETO, pay_offset=0)
        o.cancel(d(2))
        assert o.state is OrderState.CANCELLED

    def test_exactly_5000_cc_limit_is_7(self):
        o = paid_order(total=Decimal("5000.00"), method=PaymentMethod.CREDIT_CARD, pay_offset=0)
        o.cancel(d(7))
        assert o.state is OrderState.CANCELLED

    # ---- Regra C também se aplica a PICKED ----

    def test_cancel_from_picked_within_limit(self):
        """O prazo conta a partir do pagamento, mesmo em PICKED."""
        o = make_order()
        o.pay(PaymentMethod.PIX, d(1))
        o.pick(d(2))
        o.cancel(d(8))   # 7 dias após pagamento
        assert o.state is OrderState.CANCELLED

    def test_cancel_from_picked_exceeds_limit(self):
        o = make_order()
        o.pay(PaymentMethod.PIX, d(1))
        o.pick(d(2))
        with pytest.raises(RuleViolation):
            o.cancel(d(9))   # 8 dias após pagamento

    # ---- Atomicidade: falha em Regra C não muda estado ----

    def test_rule_c_failure_does_not_change_state(self):
        o = paid_order(method=PaymentMethod.BOLETO, pay_offset=0)
        try:
            o.cancel(d(10))
        except RuleViolation:
            pass
        assert o.state is OrderState.PAID


# ============================================================
# 6. Regra R – reembolsar pedido entregue
# ============================================================

class TestRuleR:

    # ---- Condição 1: forma de pagamento ----

    def test_boleto_refund_delivered_blocked(self):
        o = delivered_order(
            total=Decimal("100.00"),
            method=PaymentMethod.BOLETO,
        )
        with pytest.raises(RuleViolation):
            o.refund(d(5))

    def test_credit_card_refund_delivered_ok(self):
        o = delivered_order(
            total=Decimal("100.00"),
            method=PaymentMethod.CREDIT_CARD,
        )
        o.refund(d(5))
        assert o.state is OrderState.REFUNDED

    def test_pix_refund_delivered_ok(self):
        o = delivered_order(
            total=Decimal("100.00"),
            method=PaymentMethod.PIX,
        )
        o.refund(d(5))
        assert o.state is OrderState.REFUNDED

    # ---- Condição 2: dias desde a entrega ≤ 7 (BVA) ----
    # entrega em d(4)

    def test_refund_7_days_after_delivery_ok(self):
        o = delivered_order(total=Decimal("100.00"), method=PaymentMethod.CREDIT_CARD)
        o.refund(d(11))   # 7 dias após d(4)
        assert o.state is OrderState.REFUNDED

    def test_refund_8_days_after_delivery_fail(self):
        o = delivered_order(total=Decimal("100.00"), method=PaymentMethod.CREDIT_CARD)
        with pytest.raises(RuleViolation):
            o.refund(d(12))   # 8 dias após d(4)

    def test_refund_same_day_delivery_ok(self):
        o = delivered_order(total=Decimal("100.00"), method=PaymentMethod.PIX)
        o.refund(d(4))   # 0 dias após entrega
        assert o.state is OrderState.REFUNDED

    # ---- Condição 3: total ≤ 2000,00 (BVA) ----

    def test_refund_total_exactly_2000_ok(self):
        o = delivered_order(total=Decimal("2000.00"), method=PaymentMethod.CREDIT_CARD)
        o.refund(d(5))
        assert o.state is OrderState.REFUNDED

    def test_refund_total_2000_01_fail(self):
        o = delivered_order(total=Decimal("2000.01"), method=PaymentMethod.CREDIT_CARD)
        with pytest.raises(RuleViolation):
            o.refund(d(5))

    def test_refund_total_1999_99_ok(self):
        o = delivered_order(total=Decimal("1999.99"), method=PaymentMethod.CREDIT_CARD)
        o.refund(d(5))
        assert o.state is OrderState.REFUNDED

    # ---- Combinação: duas condições falham ao mesmo tempo ----

    def test_boleto_and_high_total_fail(self):
        """Boleto + total alto: RuleViolation (primeira condição que falha)."""
        o = delivered_order(
            total=Decimal("3000.00"),
            method=PaymentMethod.BOLETO,
        )
        with pytest.raises(RuleViolation):
            o.refund(d(5))

    # ---- Atomicidade ----

    def test_rule_r_failure_does_not_change_state(self):
        o = delivered_order(total=Decimal("100.00"), method=PaymentMethod.BOLETO)
        try:
            o.refund(d(5))
        except RuleViolation:
            pass
        assert o.state is OrderState.DELIVERED


# ============================================================
# 7. Regra R2 – reembolsar pedido cancelado
# ============================================================

class TestRuleR2:

    # ---- Condição 1: pedido deve ter sido pago ----

    def test_cancel_without_pay_no_refund(self):
        o = make_order()
        o.cancel(d(1))
        with pytest.raises(RuleViolation):
            o.refund(d(2))

    def test_cancel_after_pay_allows_refund(self):
        o = paid_order(method=PaymentMethod.CREDIT_CARD, pay_offset=1)
        o.cancel(d(2))
        o.refund(d(3))
        assert o.state is OrderState.REFUNDED

    # ---- Condição 2: BOLETO exige >= 1 dia desde pagamento (BVA) ----

    def test_boleto_cancel_same_day_as_pay_refund_blocked(self):
        """BOLETO pago e cancelado no mesmo dia: refund no mesmo dia (0 dias desde pag.) → RuleViolation."""
        o = Order("ORD-1", Decimal("100.00"), d(0))
        o.pay(PaymentMethod.BOLETO, d(0))
        o.cancel(d(0))
        with pytest.raises(RuleViolation):
            o.refund(d(0))

    def test_boleto_cancel_1_day_after_pay_refund_ok(self):
        """BOLETO com 1 dia desde pagamento: permitido."""
        o = Order("ORD-1", Decimal("100.00"), d(0))
        o.pay(PaymentMethod.BOLETO, d(0))
        o.cancel(d(1))
        o.refund(d(1))
        assert o.state is OrderState.REFUNDED

    def test_pix_cancel_same_day_as_pay_refund_ok(self):
        """PIX não exige 1 dia; mesmo dia de pagamento é aceito."""
        o = Order("ORD-1", Decimal("100.00"), d(0))
        o.pay(PaymentMethod.PIX, d(0))
        o.cancel(d(0))
        o.refund(d(0))
        assert o.state is OrderState.REFUNDED

    def test_cc_cancel_same_day_as_pay_refund_ok(self):
        o = Order("ORD-1", Decimal("100.00"), d(0))
        o.pay(PaymentMethod.CREDIT_CARD, d(0))
        o.cancel(d(0))
        o.refund(d(0))
        assert o.state is OrderState.REFUNDED

    # ---- Atomicidade ----

    def test_rule_r2_failure_does_not_change_state(self):
        o = make_order()
        o.cancel(d(1))
        try:
            o.refund(d(2))
        except RuleViolation:
            pass
        assert o.state is OrderState.CANCELLED


# ============================================================
# 8. Ordem das verificações
# ============================================================

class TestVerificationOrder:
    """
    Garante que, quando múltiplos problemas coexistem, o erro mais
    prioritário é levantado (InvalidOrder > InvalidTransition > RuleViolation).
    """

    def test_invalid_date_beats_invalid_transition(self):
        """
        'pick' em CREATED seria InvalidTransition, mas a data anterior
        ao último evento deve ser detectada primeiro (InvalidOrder).
        """
        o = make_order()   # criado em d(0)
        with pytest.raises(InvalidOrder):
            o.pick(d(-1))   # data inválida + transição inválida

    def test_invalid_date_beats_rule_violation(self):
        """
        'cancel' em PAID com data passada: InvalidOrder deve vir primeiro,
        mesmo que a Regra C também fosse violada.
        """
        # Pago em d(10), data de criação d(0)
        o = Order("ORD-1", Decimal("100.00"), d(0))
        o.pay(PaymentMethod.PIX, d(10))
        with pytest.raises(InvalidOrder):
            o.cancel(d(9))   # antes do last_event (d(10)) e viola Regra C

    def test_invalid_transition_beats_rule_violation(self):
        """
        Tentar cancelar de SHIPPED (inválido) com data futura:
        InvalidTransition deve vir antes de qualquer RuleViolation.
        """
        o = shipped_order()   # shipped em d(3)
        with pytest.raises(InvalidTransition):
            o.cancel(d(10))   # data válida mas transição inexistente

    def test_invalid_method_type_beats_invalid_transition(self):
        """
        Passar método inválido para 'pay' em PAID: InvalidOrder antes de
        InvalidTransition.
        """
        o = paid_order()
        with pytest.raises(InvalidOrder):
            o.pay("PIX", d(5))   # método errado; pay em PAID também seria inválido


# ============================================================
# 9. Exceções são subclasses de OrderError
# ============================================================

class TestExceptionHierarchy:

    def test_invalid_order_is_order_error(self):
        with pytest.raises(OrderError):
            Order("", Decimal("1.00"), d())

    def test_invalid_transition_is_order_error(self):
        o = make_order()
        with pytest.raises(OrderError):
            o.pick(d(1))

    def test_rule_violation_is_order_error(self):
        o = delivered_order(total=Decimal("100.00"), method=PaymentMethod.BOLETO)
        with pytest.raises(OrderError):
            o.refund(d(5))


# ============================================================
# 10. Testes adicionais de atomicidade e idempotência
# ============================================================

class TestAtomicity:

    def test_multiple_failures_do_not_accumulate_state(self):
        o = make_order()
        for _ in range(5):
            try:
                o.pick(d(1))
            except InvalidTransition:
                pass
        assert o.state is OrderState.CREATED

    def test_state_after_full_failure_sequence(self):
        """
        Vários erros diferentes em sequência: o estado final deve ser apenas
        o resultado das ações que tiveram sucesso.
        """
        o = paid_order(method=PaymentMethod.PIX, pay_offset=1)

        # tenta cancelar com data inválida → InvalidOrder
        try:
            o.cancel(d(0))
        except InvalidOrder:
            pass

        # tenta cancelar com transição inválida impossível: pick primeiro
        o.pick(d(2))

        # tenta cancelar após prazo → RuleViolation
        try:
            o.cancel(d(100))
        except RuleViolation:
            pass

        # estado deve ser PICKED (único sucesso foi pick)
        assert o.state is OrderState.PICKED

    def test_pay_on_same_day_as_creation_allowed(self):
        o = Order("ORD-1", Decimal("1.00"), d(5))
        o.pay(PaymentMethod.PIX, d(5))
        assert o.state is OrderState.PAID

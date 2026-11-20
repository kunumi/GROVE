"""
order_lifecycle.py
==================
Ciclo de vida de um pedido de e-commerce.

Design (guiado pelo SWEBOK):
- **State Pattern** (SWEBOK – behavioral design patterns): cada estado concreto
  sabe quais transições aceita e delega as verificações de regras de negócio
  para objetos de regra independentes (Strategy / Rule objects).
- **Modular / anticipating-change** (SWEBOK – construction): adicionar um novo
  estado exige apenas criar uma nova subclasse de `_State` e registrá-la na
  tabela `TRANSITIONS`; nenhum código existente precisa ser alterado.
- **Component-Based Design** (SWEBOK – CBD): a máquina de estados é uma unidade
  isolada atrás da interface pública `Order`; o chamador nunca toca nos objetos
  de estado internos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerações públicas
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
# Hierarquia de exceções públicas
# ---------------------------------------------------------------------------

class OrderError(Exception):
    """Raiz de todas as exceções do módulo."""

class InvalidOrder(OrderError):
    """Argumento inválido."""

class InvalidTransition(OrderError):
    """A ação não existe no estado atual."""

class RuleViolation(OrderError):
    """A ação existe, mas uma regra de negócio a impede."""


# ---------------------------------------------------------------------------
# Utilitários de validação de argumentos
# ---------------------------------------------------------------------------

def _require_date(value: object, name: str) -> None:
    """Aceita somente datetime.date puro; rejeita datetime.datetime."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise InvalidOrder(
            f"'{name}' deve ser datetime.date (não datetime.datetime nem outro tipo); "
            f"recebido: {type(value).__name__!r}"
        )


def _require_payment_method(value: object) -> None:
    if not isinstance(value, PaymentMethod):
        raise InvalidOrder(
            f"'method' deve ser PaymentMethod; recebido: {type(value).__name__!r}"
        )


def _validate_order_id(order_id: object) -> None:
    if not isinstance(order_id, str) or not order_id:
        raise InvalidOrder(
            f"'order_id' deve ser str não vazia; recebido: {order_id!r}"
        )


def _validate_total(total: object) -> None:
    if not isinstance(total, Decimal):
        raise InvalidOrder(
            f"'total' deve ser Decimal (não float, nem int); "
            f"recebido: {type(total).__name__!r}"
        )
    if total <= Decimal("0"):
        raise InvalidOrder(f"'total' deve ser maior que zero; recebido: {total!r}")
    # Verifica casas decimais
    sign, digits, exponent = total.as_tuple()
    if isinstance(exponent, int) and exponent < -2:
        raise InvalidOrder(
            f"'total' deve ter no máximo duas casas decimais; recebido: {total!r}"
        )


def _require_not_before_last_event(on: date, last_event_date: date) -> None:
    if on < last_event_date:
        raise InvalidOrder(
            f"A data da ação ({on}) não pode ser anterior à data do último evento "
            f"({last_event_date})."
        )


# ---------------------------------------------------------------------------
# Regras de negócio (Strategy objects)
# ---------------------------------------------------------------------------

class _Rule(ABC):
    """Interface comum para uma regra de negócio verificável."""

    @abstractmethod
    def check(self, ctx: "_OrderContext", on: date) -> None:
        """Levanta RuleViolation se a regra não for satisfeita."""


class _RuleC(_Rule):
    """
    Regra C – cancelar pedido pago ainda não enviado.

    Limite de dias desde o pagamento:
      - total > 5000,00  → 1 dia (qualquer pagamento)
      - BOLETO           → 2 dias
      - demais           → 7 dias
    """

    def check(self, ctx: "_OrderContext", on: date) -> None:
        days = (on - ctx.paid_on).days  # type: ignore[operator]
        limit = self._limit(ctx)
        if days > limit:
            raise RuleViolation(
                f"Cancelamento negado: {days} dia(s) desde o pagamento "
                f"(limite para este pedido: {limit} dia(s))."
            )

    @staticmethod
    def _limit(ctx: "_OrderContext") -> int:
        if ctx.total > Decimal("5000.00"):
            return 1
        if ctx.payment_method is PaymentMethod.BOLETO:
            return 2
        return 7


class _RuleR(_Rule):
    """
    Regra R – reembolsar pedido entregue.

    Todas as três condições devem ser satisfeitas:
      1. forma de pagamento é CREDIT_CARD ou PIX;
      2. dias desde a entrega ≤ 7;
      3. total ≤ 2000,00.
    """

    def check(self, ctx: "_OrderContext", on: date) -> None:
        if ctx.payment_method not in (PaymentMethod.CREDIT_CARD, PaymentMethod.PIX):
            raise RuleViolation(
                "Reembolso negado: forma de pagamento não elegível "
                f"({ctx.payment_method.value}); apenas CREDIT_CARD e PIX são aceitos."
            )
        days = (on - ctx.delivered_on).days  # type: ignore[operator]
        if days > 7:
            raise RuleViolation(
                f"Reembolso negado: {days} dia(s) desde a entrega (limite: 7 dias)."
            )
        if ctx.total > Decimal("2000.00"):
            raise RuleViolation(
                f"Reembolso negado: total {ctx.total} excede o limite de 2000,00."
            )


class _RuleR2(_Rule):
    """
    Regra R2 – reembolsar pedido cancelado.

    Condições:
      1. O pedido deve ter sido pago antes de ser cancelado (paid_on não-nulo).
      2. Se BOLETO, exige pelo menos 1 dia desde o pagamento.
    """

    def check(self, ctx: "_OrderContext", on: date) -> None:
        if ctx.paid_on is None:
            raise RuleViolation(
                "Reembolso negado: o pedido foi cancelado sem ter sido pago."
            )
        if ctx.payment_method is PaymentMethod.BOLETO:
            days = (on - ctx.paid_on).days
            if days < 1:
                raise RuleViolation(
                    f"Reembolso negado (BOLETO): é necessário pelo menos 1 dia desde "
                    f"o pagamento; passaram-se {days} dia(s)."
                )


# Instâncias singleton das regras (sem estado próprio)
_RULE_C  = _RuleC()
_RULE_R  = _RuleR()
_RULE_R2 = _RuleR2()


# ---------------------------------------------------------------------------
# Contexto interno (dados mutáveis do pedido)
# ---------------------------------------------------------------------------

class _OrderContext:
    """
    Carrega todo o estado mutável do pedido.
    Os objetos de estado (_State) recebem uma referência a este contexto.
    """

    __slots__ = (
        "order_id",
        "total",
        "created_on",
        "state",
        "payment_method",
        "paid_on",
        "picked_on",
        "shipped_on",
        "delivered_on",
        "cancelled_on",
        "refunded_on",
        "last_event_date",
    )

    def __init__(self, order_id: str, total: Decimal, created_on: date) -> None:
        self.order_id: str                       = order_id
        self.total: Decimal                      = total
        self.created_on: date                    = created_on
        self.state: "_State"                     = _StateCreated()
        self.payment_method: Optional[PaymentMethod] = None
        self.paid_on:        Optional[date]      = None
        self.picked_on:      Optional[date]      = None
        self.shipped_on:     Optional[date]      = None
        self.delivered_on:   Optional[date]      = None
        self.cancelled_on:   Optional[date]      = None
        self.refunded_on:    Optional[date]      = None
        self.last_event_date: date               = created_on


# ---------------------------------------------------------------------------
# State Pattern – estados concretos
# ---------------------------------------------------------------------------

class _State(ABC):
    """
    Classe-base para todos os estados do pedido.

    Cada método de ação, por padrão, levanta InvalidTransition.
    Os estados concretos sobrescrevem apenas as ações que suportam.
    Graças ao SWEBOK (State Pattern + anticipating change), adicionar um novo
    estado é só criar uma nova subclasse aqui.
    """

    @property
    @abstractmethod
    def order_state(self) -> OrderState: ...

    # ---- ações (padrão = transição inválida) ----------------------------

    def pay(self, ctx: _OrderContext, method: PaymentMethod, on: date) -> None:
        raise InvalidTransition(
            f"Ação 'pay' não permitida no estado {self.order_state.value}."
        )

    def pick(self, ctx: _OrderContext, on: date) -> None:
        raise InvalidTransition(
            f"Ação 'pick' não permitida no estado {self.order_state.value}."
        )

    def ship(self, ctx: _OrderContext, on: date) -> None:
        raise InvalidTransition(
            f"Ação 'ship' não permitida no estado {self.order_state.value}."
        )

    def deliver(self, ctx: _OrderContext, on: date) -> None:
        raise InvalidTransition(
            f"Ação 'deliver' não permitida no estado {self.order_state.value}."
        )

    def cancel(self, ctx: _OrderContext, on: date) -> None:
        raise InvalidTransition(
            f"Ação 'cancel' não permitida no estado {self.order_state.value}."
        )

    def refund(self, ctx: _OrderContext, on: date) -> None:
        raise InvalidTransition(
            f"Ação 'refund' não permitida no estado {self.order_state.value}."
        )


class _StateCreated(_State):
    order_state = OrderState.CREATED

    def pay(self, ctx: _OrderContext, method: PaymentMethod, on: date) -> None:
        ctx.payment_method = method
        ctx.paid_on        = on
        ctx.last_event_date = on
        ctx.state          = _StatePaid()

    def cancel(self, ctx: _OrderContext, on: date) -> None:
        ctx.cancelled_on    = on
        ctx.last_event_date = on
        ctx.state           = _StateCancelled()


class _StatePaid(_State):
    order_state = OrderState.PAID

    def pick(self, ctx: _OrderContext, on: date) -> None:
        ctx.picked_on       = on
        ctx.last_event_date = on
        ctx.state           = _StatePicked()

    def cancel(self, ctx: _OrderContext, on: date) -> None:
        _RULE_C.check(ctx, on)
        ctx.cancelled_on    = on
        ctx.last_event_date = on
        ctx.state           = _StateCancelled()


class _StatePicked(_State):
    order_state = OrderState.PICKED

    def ship(self, ctx: _OrderContext, on: date) -> None:
        ctx.shipped_on      = on
        ctx.last_event_date = on
        ctx.state           = _StateShipped()

    def cancel(self, ctx: _OrderContext, on: date) -> None:
        _RULE_C.check(ctx, on)
        ctx.cancelled_on    = on
        ctx.last_event_date = on
        ctx.state           = _StateCancelled()


class _StateShipped(_State):
    order_state = OrderState.SHIPPED

    def deliver(self, ctx: _OrderContext, on: date) -> None:
        ctx.delivered_on    = on
        ctx.last_event_date = on
        ctx.state           = _StateDelivered()


class _StateDelivered(_State):
    order_state = OrderState.DELIVERED

    def refund(self, ctx: _OrderContext, on: date) -> None:
        _RULE_R.check(ctx, on)
        ctx.refunded_on     = on
        ctx.last_event_date = on
        ctx.state           = _StateRefunded()


class _StateCancelled(_State):
    order_state = OrderState.CANCELLED

    def refund(self, ctx: _OrderContext, on: date) -> None:
        _RULE_R2.check(ctx, on)
        ctx.refunded_on     = on
        ctx.last_event_date = on
        ctx.state           = _StateRefunded()


class _StateRefunded(_State):
    order_state = OrderState.REFUNDED
    # Nenhuma ação disponível; todos os métodos usam o padrão (InvalidTransition).


# ---------------------------------------------------------------------------
# Fachada pública: Order
# ---------------------------------------------------------------------------

class Order:
    """
    Fachada pública que encapsula o contexto e a máquina de estados.

    Toda chamada de ação segue a ordem de verificação mandatória:
      1. Validação de argumentos  → InvalidOrder
      2. Existência da transição  → InvalidTransition  (delegado ao estado)
      3. Regra de negócio         → RuleViolation      (delegado à regra)

    Uma chamada que levanta exceção NÃO altera o estado do pedido.
    """

    def __init__(self, order_id: str, total: Decimal, created_on: date) -> None:
        # Validação de argumentos do construtor
        _validate_order_id(order_id)
        _validate_total(total)
        _require_date(created_on, "created_on")

        self._ctx = _OrderContext(order_id, total, created_on)

    # ---- propriedade pública de estado -----------------------------------

    @property
    def state(self) -> OrderState:
        return self._ctx.state.order_state

    # ---- ações públicas --------------------------------------------------

    def pay(self, method: PaymentMethod, on: date) -> None:
        # 1. Argumentos
        _require_payment_method(method)
        _require_date(on, "on")
        _require_not_before_last_event(on, self._ctx.last_event_date)
        # 2 & 3. Estado atual (pode levantar InvalidTransition ou RuleViolation)
        self._ctx.state.pay(self._ctx, method, on)

    def pick(self, on: date) -> None:
        # 1. Argumentos
        _require_date(on, "on")
        _require_not_before_last_event(on, self._ctx.last_event_date)
        # 2 & 3.
        self._ctx.state.pick(self._ctx, on)

    def ship(self, on: date) -> None:
        # 1. Argumentos
        _require_date(on, "on")
        _require_not_before_last_event(on, self._ctx.last_event_date)
        # 2 & 3.
        self._ctx.state.ship(self._ctx, on)

    def deliver(self, on: date) -> None:
        # 1. Argumentos
        _require_date(on, "on")
        _require_not_before_last_event(on, self._ctx.last_event_date)
        # 2 & 3.
        self._ctx.state.deliver(self._ctx, on)

    def cancel(self, on: date) -> None:
        # 1. Argumentos
        _require_date(on, "on")
        _require_not_before_last_event(on, self._ctx.last_event_date)
        # 2 & 3.
        self._ctx.state.cancel(self._ctx, on)

    def refund(self, on: date) -> None:
        # 1. Argumentos
        _require_date(on, "on")
        _require_not_before_last_event(on, self._ctx.last_event_date)
        # 2 & 3.
        self._ctx.state.refund(self._ctx, on)

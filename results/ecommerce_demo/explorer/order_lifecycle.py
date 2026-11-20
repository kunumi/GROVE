"""
order_lifecycle.py
==================
Ciclo de vida de um pedido de e-commerce.

Design (SWEBOK v4):
- SRP / OCP: validação, tabela de transições e regras de negócio são
  responsabilidades separadas; adicionar estados/regras não exige mexer
  em código existente — apenas estender a tabela e os checkers.
- State-Based + Table-Driven: a FSM vive em _TRANSITIONS; nenhuma
  lógica condicional acopla estados entre si.
- Design by Contract: pré-condições verificadas em ordem definida pelo
  contrato (argumento → transição → regra) antes de qualquer mutação.
- Exceções informativas: mensagens descrevem o que aconteceu e por quê.
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Callable, Optional


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
    """Raiz da hierarquia de erros de pedido."""


class InvalidOrder(OrderError):
    """Argumento inválido para criar ou operar um pedido."""


class InvalidTransition(OrderError):
    """A ação não existe no estado atual do pedido."""


class RuleViolation(OrderError):
    """A ação existe, mas uma regra de negócio a impede."""


# ---------------------------------------------------------------------------
# Helpers internos de validação (Design by Contract — pré-condições)
# ---------------------------------------------------------------------------

def _require_strict_date(value: object, name: str) -> datetime.date:
    """Aceita datetime.date mas rejeita datetime.datetime (subtipo)."""
    if isinstance(value, datetime.datetime):  # datetime é subtipo de date
        raise InvalidOrder(
            f"'{name}' deve ser datetime.date, não datetime.datetime."
        )
    if not isinstance(value, datetime.date):
        raise InvalidOrder(
            f"'{name}' deve ser datetime.date; recebido {type(value).__name__}."
        )
    return value


def _validate_constructor_args(
    order_id: object,
    total: object,
    created_on: object,
) -> tuple[str, Decimal, datetime.date]:
    """Valida e devolve os argumentos do construtor já tipados."""
    if not isinstance(order_id, str) or not order_id:
        raise InvalidOrder(
            f"'order_id' deve ser str não vazia; recebido {order_id!r}."
        )
    # Decimal — rejeita float e int explicitamente
    if not isinstance(total, Decimal):
        raise InvalidOrder(
            f"'total' deve ser Decimal; recebido {type(total).__name__}."
        )
    if total <= Decimal("0"):
        raise InvalidOrder(
            f"'total' deve ser maior que zero; recebido {total}."
        )
    # Verifica no máximo duas casas decimais: exponent() == -2 ou superior
    sign, digits, exponent = total.as_tuple()
    if exponent < -2:
        raise InvalidOrder(
            f"'total' deve ter no máximo duas casas decimais; recebido {total}."
        )
    created_on = _require_strict_date(created_on, "created_on")
    return order_id, total, created_on


def _validate_action_date(on: object, last_event_date: datetime.date) -> datetime.date:
    """Valida que 'on' é date válido e não anterior ao último evento."""
    on = _require_strict_date(on, "on")
    if on < last_event_date:
        raise InvalidOrder(
            f"'on' ({on}) não pode ser anterior à data do último evento "
            f"({last_event_date})."
        )
    return on


def _validate_payment_method(method: object) -> PaymentMethod:
    if not isinstance(method, PaymentMethod):
        raise InvalidOrder(
            f"'method' deve ser PaymentMethod; recebido {type(method).__name__}."
        )
    return method


# ---------------------------------------------------------------------------
# Regras de negócio (isoladas — OCP: adicionar regra = adicionar função)
# ---------------------------------------------------------------------------

def _check_rule_c(order: "Order", on: datetime.date) -> None:
    """
    Regra C — cancelar pedido pago e ainda não enviado.
    Levanta RuleViolation se o prazo foi excedido.
    """
    assert order._paid_on is not None  # invariante
    days = (on - order._paid_on).days
    total = order._total
    method = order._payment_method

    if total > Decimal("5000.00"):
        limit = 1
    elif method is PaymentMethod.BOLETO:
        limit = 2
    else:
        limit = 7

    if days > limit:
        raise RuleViolation(
            f"Cancelamento fora do prazo: {days} dia(s) desde o pagamento; "
            f"limite para este pedido é {limit} dia(s)."
        )


def _check_rule_r(order: "Order", on: datetime.date) -> None:
    """
    Regra R — reembolsar pedido entregue.
    As três condições devem valer simultaneamente.
    """
    assert order._delivered_on is not None  # invariante
    errors = []

    if order._payment_method not in (PaymentMethod.CREDIT_CARD, PaymentMethod.PIX):
        errors.append("forma de pagamento deve ser CREDIT_CARD ou PIX")

    days = (on - order._delivered_on).days
    if days > 7:
        errors.append(
            f"{days} dia(s) desde a entrega excedem o limite de 7"
        )

    if order._total > Decimal("2000.00"):
        errors.append(
            f"total {order._total} excede o limite de 2000,00"
        )

    if errors:
        raise RuleViolation(
            "Reembolso de pedido entregue negado: " + "; ".join(errors) + "."
        )


def _check_rule_r2(order: "Order", on: datetime.date) -> None:
    """
    Regra R2 — reembolsar pedido cancelado.
    Exige que o pedido tenha sido pago antes do cancelamento.
    Se BOLETO, exige pelo menos 1 dia desde o pagamento.
    """
    if order._paid_on is None:
        raise RuleViolation(
            "Reembolso de pedido cancelado negado: o pedido nunca foi pago."
        )

    if order._payment_method is PaymentMethod.BOLETO:
        days = (on - order._paid_on).days
        if days < 1:
            raise RuleViolation(
                f"Reembolso via BOLETO exige pelo menos 1 dia desde o pagamento; "
                f"decorrido {days} dia(s)."
            )


# ---------------------------------------------------------------------------
# Tabela de transições (State-Based Table-Driven — SWEBOK)
#
# Estrutura:
#   _TRANSITIONS[estado_atual][nome_da_ação] = (novo_estado, checker_ou_None)
#
# checker é uma função (order, on) -> None que levanta RuleViolation se
# a regra de negócio falhar.  None significa "sem restrição adicional".
# ---------------------------------------------------------------------------

_RuleChecker = Optional[Callable[["Order", datetime.date], None]]

_TRANSITIONS: dict[
    OrderState,
    dict[str, tuple[OrderState, _RuleChecker]]
] = {
    OrderState.CREATED: {
        "pay":    (OrderState.PAID,      None),
        "cancel": (OrderState.CANCELLED, None),
    },
    OrderState.PAID: {
        "pick":   (OrderState.PICKED,    None),
        "cancel": (OrderState.CANCELLED, _check_rule_c),
    },
    OrderState.PICKED: {
        "ship":   (OrderState.SHIPPED,   None),
        "cancel": (OrderState.CANCELLED, _check_rule_c),
    },
    OrderState.SHIPPED: {
        "deliver": (OrderState.DELIVERED, None),
    },
    OrderState.DELIVERED: {
        "refund": (OrderState.REFUNDED, _check_rule_r),
    },
    OrderState.CANCELLED: {
        "refund": (OrderState.REFUNDED, _check_rule_r2),
    },
    OrderState.REFUNDED: {},   # estado terminal — sem saídas
}


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class Order:
    """
    Representa um pedido de e-commerce com ciclo de vida controlado por FSM.

    Atributos internos são prefixados com _ (encapsulamento).
    Mutação só ocorre após todas as verificações passarem (atomicidade).
    """

    def __init__(self, order_id: str, total: Decimal, created_on: datetime.date) -> None:
        order_id, total, created_on = _validate_constructor_args(
            order_id, total, created_on
        )
        self._order_id: str            = order_id
        self._total: Decimal           = total
        self._created_on: datetime.date = created_on

        # Estado e datas de eventos (imutáveis após registro)
        self._state: OrderState             = OrderState.CREATED
        self._last_event_date: datetime.date = created_on

        # Dados opcionais preenchidos ao longo do ciclo de vida
        self._payment_method: Optional[PaymentMethod] = None
        self._paid_on: Optional[datetime.date]        = None
        self._delivered_on: Optional[datetime.date]   = None

    # ------------------------------------------------------------------
    # Propriedade pública
    # ------------------------------------------------------------------

    @property
    def state(self) -> OrderState:
        return self._state

    # ------------------------------------------------------------------
    # Métodos de ação públicos — delegam para _execute_transition
    # ------------------------------------------------------------------

    def pay(self, method: PaymentMethod, on: datetime.date) -> None:
        # Valida method antes do date para manter a ordem: argumentos primeiro
        method = _validate_payment_method(method)
        on = _validate_action_date(on, self._last_event_date)
        self._execute_transition("pay", on, payment_method=method)

    def pick(self, on: datetime.date) -> None:
        on = _validate_action_date(on, self._last_event_date)
        self._execute_transition("pick", on)

    def ship(self, on: datetime.date) -> None:
        on = _validate_action_date(on, self._last_event_date)
        self._execute_transition("ship", on)

    def deliver(self, on: datetime.date) -> None:
        on = _validate_action_date(on, self._last_event_date)
        self._execute_transition("deliver", on)

    def cancel(self, on: datetime.date) -> None:
        on = _validate_action_date(on, self._last_event_date)
        self._execute_transition("cancel", on)

    def refund(self, on: datetime.date) -> None:
        on = _validate_action_date(on, self._last_event_date)
        self._execute_transition("refund", on)

    # ------------------------------------------------------------------
    # Núcleo da FSM — verificação e mutação
    # ------------------------------------------------------------------

    def _execute_transition(
        self,
        action: str,
        on: datetime.date,
        **extra: object,
    ) -> None:
        """
        Executa uma transição seguindo a ordem de verificação do contrato:
          1. argumentos (já validados pelos chamadores)
          2. existência da transição no estado atual
          3. regra de negócio
        Apenas após todas as verificações o estado é mutado.
        """
        # 2. Verifica se a ação existe no estado atual
        available = _TRANSITIONS.get(self._state, {})
        if action not in available:
            raise InvalidTransition(
                f"Ação '{action}' não permitida no estado {self._state.value}."
            )

        new_state, checker = available[action]

        # 3. Regra de negócio
        if checker is not None:
            checker(self, on)

        # --- Mutação (só alcançada se tudo passou) ---
        self._state = new_state
        self._last_event_date = on

        # Registra dados específicos de cada ação
        if action == "pay":
            self._payment_method = extra["payment_method"]  # type: ignore[assignment]
            self._paid_on = on
        elif action == "deliver":
            self._delivered_on = on

    # ------------------------------------------------------------------
    # Representação (útil em depuração)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Order(id={self._order_id!r}, state={self._state.value}, "
            f"total={self._total}, last_event={self._last_event_date})"
        )

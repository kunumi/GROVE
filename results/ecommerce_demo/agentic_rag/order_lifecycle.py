"""
order_lifecycle.py
==================
Ciclo de vida de um pedido de e-commerce.

Design interno
--------------
* A máquina de estados é codificada como um dicionário de tabela de transições
  (SWEBOK §4.7 "table-driven method") em vez de if/elif aninhados, tornando
  trivial adicionar novos estados ou regras: basta inserir nova entrada na tabela
  e, se necessário, registrar um predicado de guarda em GUARDS.

* Cada ação executa a sequência obrigatória de verificações:
    1. Validação de argumentos  → InvalidOrder
    2. Existência da transição  → InvalidTransition
    3. Regra de negócio         → RuleViolation

* O Order não mantém clock interno; toda data é injetada pela chamada (on=).

* Atomicidade: estado e datas só são atualizados depois de todas as verificações
  passarem, garantindo que uma chamada que levanta exceção não altera o pedido.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Callable, NamedTuple, Optional

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
    """Argumento inválido (tipo, valor ou data anterior ao último evento)."""


class InvalidTransition(OrderError):
    """A ação não existe no estado atual."""


class RuleViolation(OrderError):
    """A ação existe, mas uma regra de negócio a impede."""


# ---------------------------------------------------------------------------
# Tabela de transições e guardas
# ---------------------------------------------------------------------------

# Chave: (estado_atual, nome_da_ação)
# Valor: estado_destino
_TRANSITIONS: dict[tuple[OrderState, str], OrderState] = {
    (OrderState.CREATED,   "pay"):    OrderState.PAID,
    (OrderState.CREATED,   "cancel"): OrderState.CANCELLED,
    (OrderState.PAID,      "pick"):   OrderState.PICKED,
    (OrderState.PAID,      "cancel"): OrderState.CANCELLED,
    (OrderState.PICKED,    "ship"):   OrderState.SHIPPED,
    (OrderState.PICKED,    "cancel"): OrderState.CANCELLED,
    (OrderState.SHIPPED,   "deliver"):OrderState.DELIVERED,
    (OrderState.DELIVERED, "refund"): OrderState.REFUNDED,
    (OrderState.CANCELLED, "refund"): OrderState.REFUNDED,
}

# ---------------------------------------------------------------------------
# Implementação das regras de negócio (predicados de guarda)
# Cada guarda recebe o Order e o 'on' da ação; levanta RuleViolation se inválido.
# ---------------------------------------------------------------------------

def _rule_C(order: "Order", on: date) -> None:
    """Regra C — cancelar pedido pago ainda não enviado."""
    assert order._payment_date is not None, "Regra C pressupõe pagamento registrado"
    days = (on - order._payment_date).days

    if order._total > Decimal("5000.00"):
        limit = 1
    elif order._payment_method is PaymentMethod.BOLETO:
        limit = 2
    else:
        limit = 7

    if days > limit:
        raise RuleViolation(
            f"Cancelamento fora do prazo: {days} dia(s) desde o pagamento "
            f"(limite {limit})."
        )


def _rule_R(order: "Order", on: date) -> None:
    """Regra R — reembolsar pedido entregue."""
    assert order._delivery_date is not None, "Regra R pressupõe entrega registrada"

    if order._payment_method not in (PaymentMethod.CREDIT_CARD, PaymentMethod.PIX):
        raise RuleViolation(
            "Reembolso só é permitido para pagamentos com CREDIT_CARD ou PIX."
        )

    days = (on - order._delivery_date).days
    if days > 7:
        raise RuleViolation(
            f"Reembolso fora do prazo: {days} dia(s) desde a entrega (limite 7)."
        )

    if order._total > Decimal("2000.00"):
        raise RuleViolation(
            f"Reembolso não permitido: total {order._total} excede 2000,00."
        )


def _rule_R2(order: "Order", on: date) -> None:
    """Regra R2 — reembolsar pedido cancelado."""
    if order._payment_date is None:
        raise RuleViolation(
            "Reembolso de pedido cancelado exige que o pedido tenha sido pago."
        )

    if order._payment_method is PaymentMethod.BOLETO:
        days = (on - order._payment_date).days
        if days < 1:
            raise RuleViolation(
                "Reembolso via BOLETO exige pelo menos 1 dia desde o pagamento."
            )


# Mapeamento de guarda por transição (opcional; None significa sem restrição).
_GUARDS: dict[tuple[OrderState, str], Callable[["Order", date], None]] = {
    (OrderState.PAID,      "cancel"): _rule_C,
    (OrderState.PICKED,    "cancel"): _rule_C,
    (OrderState.DELIVERED, "refund"): _rule_R,
    (OrderState.CANCELLED, "refund"): _rule_R2,
}


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class Order:
    """
    Representa o ciclo de vida de um pedido de e-commerce.

    Atributos internos relevantes para as regras de negócio:
        _state           – estado atual (OrderState)
        _total           – valor total (Decimal)
        _payment_method  – forma de pagamento (PaymentMethod | None)
        _payment_date    – data do pagamento (date | None)
        _delivery_date   – data da entrega (date | None)
        _last_event_date – data do último evento bem-sucedido
    """

    def __init__(self, order_id: str, total: Decimal, created_on: date) -> None:
        # --- validação de argumentos ---
        if not isinstance(order_id, str) or not order_id:
            raise InvalidOrder("order_id deve ser uma str não vazia.")

        if not isinstance(total, Decimal):
            raise InvalidOrder(
                "total deve ser Decimal (não float nem int)."
            )
        if total <= Decimal("0"):
            raise InvalidOrder("total deve ser maior que zero.")
        # Verifica casas decimais
        sign, digits, exponent = total.as_tuple()
        if exponent < -2:
            raise InvalidOrder(
                "total deve ter no máximo duas casas decimais."
            )

        _validate_date(created_on, "created_on")

        # --- inicialização ---
        self._order_id:        str                    = order_id
        self._total:           Decimal                = total
        self._created_on:      date                   = created_on
        self._state:           OrderState             = OrderState.CREATED
        self._last_event_date: date                   = created_on
        self._payment_method:  Optional[PaymentMethod] = None
        self._payment_date:    Optional[date]          = None
        self._delivery_date:   Optional[date]          = None

    # ------------------------------------------------------------------
    # Propriedade pública
    # ------------------------------------------------------------------

    @property
    def state(self) -> OrderState:
        return self._state

    # ------------------------------------------------------------------
    # Ações públicas — cada uma delega para _apply_action
    # ------------------------------------------------------------------

    def pay(self, method: PaymentMethod, on: date) -> None:
        # Validação extra: method deve ser PaymentMethod
        if not isinstance(method, PaymentMethod):
            raise InvalidOrder(
                f"method deve ser um membro de PaymentMethod, recebido: {method!r}"
            )
        self._apply_action("pay", on, method=method)

    def pick(self, on: date) -> None:
        self._apply_action("pick", on)

    def ship(self, on: date) -> None:
        self._apply_action("ship", on)

    def deliver(self, on: date) -> None:
        self._apply_action("deliver", on)

    def cancel(self, on: date) -> None:
        self._apply_action("cancel", on)

    def refund(self, on: date) -> None:
        self._apply_action("refund", on)

    # ------------------------------------------------------------------
    # Motor interno da máquina de estados
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        action: str,
        on: date,
        method: Optional[PaymentMethod] = None,
    ) -> None:
        """
        Executa uma ação seguindo estritamente a ordem de verificações:
          1. Argumentos inválidos (InvalidOrder)
          2. Transição inexistente (InvalidTransition)
          3. Regra de negócio (RuleViolation)

        Se todas as verificações passarem, atualiza o estado atomicamente.
        """
        # 1a. Validar 'on'
        _validate_date(on, "on")

        # 1b. 'on' não pode ser anterior ao último evento
        if on < self._last_event_date:
            raise InvalidOrder(
                f"A data '{on}' é anterior ao último evento "
                f"registrado ({self._last_event_date})."
            )

        # 2. Verificar se a transição existe
        key = (self._state, action)
        if key not in _TRANSITIONS:
            raise InvalidTransition(
                f"Ação '{action}' não é permitida no estado {self._state.name}."
            )

        # 3. Verificar regra de negócio (guarda)
        guard = _GUARDS.get(key)
        if guard is not None:
            guard(self, on)  # levanta RuleViolation se inválido

        # --- Atualização atômica do estado ---
        self._state = _TRANSITIONS[key]
        self._last_event_date = on

        if action == "pay":
            self._payment_method = method
            self._payment_date   = on
        elif action == "deliver":
            self._delivery_date = on


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _validate_date(value: object, param_name: str) -> None:
    """
    Aceita apenas datetime.date puro; rejeita datetime.datetime
    (que é subclasse de date) e qualquer outro tipo.
    """
    import datetime as _dt
    if isinstance(value, _dt.datetime) or not isinstance(value, _dt.date):
        raise InvalidOrder(
            f"'{param_name}' deve ser datetime.date (não datetime.datetime)."
        )

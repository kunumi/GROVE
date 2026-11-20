Implemente em Python o ciclo de vida de um pedido de e-commerce seguindo exatamente o contrato abaixo. O comportamento e a interface estão fixados; o desenho interno (como organizar o código, como representar estados e regras, como tratar erros) e a suíte de testes são decisões suas. Projete a implementação para ser fácil de modificar quando surgirem novos estados ou regras, e entregue uma suíte de testes que dê confiança de que nenhuma transição ou combinação de regras ficou sem verificação.

## Entregáveis

Dois arquivos, no seu diretório de trabalho:

- `order_lifecycle.py`: a implementação;
- `test_order_lifecycle.py`: a sua suíte de testes (pytest), que importa de `order_lifecycle`.

Use só a biblioteca padrão do Python 3.12 na implementação.

## Interface obrigatória (módulo `order_lifecycle`)

```python
from datetime import date
from decimal import Decimal
from enum import Enum

class OrderState(Enum):      # exatamente estes membros
    CREATED, PAID, PICKED, SHIPPED, DELIVERED, CANCELLED, REFUNDED

class PaymentMethod(Enum):   # exatamente estes membros
    CREDIT_CARD, PIX, BOLETO

class OrderError(Exception): ...
class InvalidOrder(OrderError): ...        # argumento inválido
class InvalidTransition(OrderError): ...   # a ação não existe no estado atual
class RuleViolation(OrderError): ...       # a ação existe, mas uma regra de negócio a impede

class Order:
    def __init__(self, order_id: str, total: Decimal, created_on: date): ...
    @property
    def state(self) -> OrderState: ...
    def pay(self, method: PaymentMethod, on: date) -> None: ...
    def pick(self, on: date) -> None: ...
    def ship(self, on: date) -> None: ...
    def deliver(self, on: date) -> None: ...
    def cancel(self, on: date) -> None: ...
    def refund(self, on: date) -> None: ...
```

Um pedido novo começa em `CREATED`. Toda ação recebe a data em que acontece (`on`); a implementação não pode consultar o relógio do sistema.

## Transições

| estado atual | ação | novo estado | condição |
|---|---|---|---|
| CREATED | `pay` | PAID | — |
| CREATED | `cancel` | CANCELLED | — (sempre permitido) |
| PAID | `pick` | PICKED | — |
| PAID | `cancel` | CANCELLED | regra C |
| PICKED | `ship` | SHIPPED | — |
| PICKED | `cancel` | CANCELLED | regra C |
| SHIPPED | `deliver` | DELIVERED | — |
| DELIVERED | `refund` | REFUNDED | regra R |
| CANCELLED | `refund` | REFUNDED | regra R2 |

Qualquer combinação de estado e ação fora desta tabela levanta `InvalidTransition`. `REFUNDED` não tem saída.

## Regras de negócio

Nas regras, "dias" é `(on - data_do_evento).days`, contado em dias corridos. "Data do pagamento" é o `on` do `pay`, e "data da entrega" é o `on` do `deliver`.

- **Regra C (cancelar um pedido pago e ainda não enviado):** permitido se os dias desde o pagamento forem no máximo o limite:
  - se o total for maior que 5000,00: limite de 1 dia, qualquer que seja a forma de pagamento;
  - senão, se a forma de pagamento for BOLETO: limite de 2 dias;
  - senão: limite de 7 dias.
- **Regra R (reembolsar um pedido entregue):** permitido só se as três condições valerem:
  - a forma de pagamento é CREDIT_CARD ou PIX;
  - os dias desde a entrega são no máximo 7;
  - o total é no máximo 2000,00.
- **Regra R2 (reembolsar um pedido cancelado):** permitido só se o pedido tiver sido pago antes de ser cancelado. Se a forma de pagamento for BOLETO, exige também pelo menos 1 dia desde o pagamento.

Quando uma regra impede a ação, levante `RuleViolation`.

## Validação de argumentos (`InvalidOrder`)

- `order_id` precisa ser uma `str` não vazia.
- `total` precisa ser um `Decimal` (não `float`, nem `int`), maior que zero e com no máximo duas casas decimais.
- `created_on` e todo `on` precisam ser `datetime.date`; um `datetime.datetime` não é aceito.
- `method` precisa ser um membro de `PaymentMethod`.
- O `on` de uma ação não pode ser anterior à data do último evento do pedido (a criação ou a última transição feita com sucesso).

## Ordem das verificações

Numa chamada de ação, verifique nesta ordem e pare na primeira falha:
1. os argumentos (`InvalidOrder`), inclusive a data anterior ao último evento;
2. se a ação existe no estado atual (`InvalidTransition`);
3. a regra de negócio (`RuleViolation`).

Uma chamada que levanta exceção não altera o pedido: o estado e as datas registradas continuam os mesmos.

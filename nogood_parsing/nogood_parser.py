from collections import Counter
import re
from typing import Any, Dict, List, NamedTuple, Union
from dataclasses import dataclass
from parsy import regex, string, eof, Parser, generate, forward_declaration, seq


@dataclass
class BinaryExpr:
    """
    Represents a binary expression with an operator and two operands.

    Attributes:
        op (str): The operator as a string (e.g., '+', '-', '*', '/', etc.).
        left (Expr): The left operand expression.
        right (Expr): The right operand expression.
    """

    op: str
    left: "Expr"
    right: "Expr"

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryExpr:
    """
    Represents a unary expression with an operator and one operand.

    Attributes:
        op (str): The unary operator (e.g., '-', '!').
        operand (Expr): The operand expression.
    """

    op: str
    operand: "Expr"

    def __repr__(self) -> str:
        return f"{self.op}{self.operand}"


@dataclass
class SetExpr:
    """
    Represents a set expression, typically used with '{expr} in int(...)' operations.

    Attributes:
        expr (Expr): The expression to be evaluated.
        range (List[int]): The list of integer values defining the set.
    """

    expr: "Expr"
    range: List[int]

    def __repr__(self) -> str:
        return f"{self.expr} in int({', '.join(map(str, self.range))})"


@dataclass
class Identifier:
    """
    Represents a variable or constant identifier, possibly with indices.

    Attributes:
        name (str): The name of the identifier.
        indices (List[int]): A list of indices if the identifier is a matrix.
    """

    name: str
    indices: List[int]

    def __repr__(self) -> str:
        if len(self.indices) == 0:
            return f"{self.name}"
        else:
            return f"{self.name}[{', '.join(map(str, self.indices))}]"

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.indices)))


Atom = Union[int, bool, Identifier]
Expr = Union[Atom, BinaryExpr, UnaryExpr, SetExpr]


class Token(NamedTuple):
    kind: str
    value: Union[str, int, bool, Identifier]
    pos: int


class NogoodParser:
    def __init__(self, find_json: List[Dict[str, Any]]):
        self.find_json = find_json

    def tokenize(self, s: str) -> List[Token]:
        token_specification = [
            ("WHITESPACE", r"\s+"),
            ("INT", r"[-+]?\d+"),
            ("BOOL", r"true|false"),
            ("SHIFT", r"shift"),
            ("INT_TYPE", r"int"),
            ("IN", r"in"),
            ("RANGE", r"\.\."),
            ("COMMA", r","),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("NOT", r"!"),
            ("MINUS", r"-"),
            ("PLUS", r"\+"),
            ("MULT", r"\*"),
            ("DIV", r"/"),
            ("MOD", r"%"),
            ("LTE", r"<="),
            ("GTE", r">="),
            ("LT", r"<"),
            ("GT", r">"),
            ("EQ", r"=="),
            ("NEQ", r"!="),
            ("OR", r"\\/"),
            ("IDENTIFIER", r"[a-zA-Z_]\w*"),
        ]
        tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
        tokens = []
        for mo in re.finditer(tok_regex, s):
            kind = mo.lastgroup
            value = mo.group()
            pos = mo.start()
            match kind:
                case "WHITESPACE":
                    pass
                case "INT":
                    value = int(value)
                case "BOOL":
                    value = value == "true"
                case "IDENTIFIER":
                    indices = []
                    for find in self.find_json:
                        find_name = find["name"]
                        if value.startswith(find_name):
                            indices_str = value.replace(f"{find_name}", "")
                            value = find_name
                            if indices_str != "":
                                indices = [int(i) for i in indices_str[1:].split("_")]
                            break
                    value = Identifier(value, indices)
                case None:
                    raise ValueError(f"Unexpected character: {value}")
            tokens.append(Token(kind, value, pos))
        return tokens

    def parse(self, tokens: List[Token]) -> Expr:
        raise NotImplementedError


def get_identifier_counts(tokens: List[Token]) -> Counter[Identifier]:
    identifiers: List[Identifier] = []

    for token in tokens:
        if token.kind == "IDENTIFIER":
            identifiers.append(token.value)  # type: ignore

    return Counter(identifiers)

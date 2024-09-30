"""
Parser module to parse nogood expressions used in constraint satisfaction problems,
specifically EssencePrime models.

This module defines classes and functions to tokenize and parse nogood expressions,
represent them as expression trees, and extract identifier counts for further processing.

Classes:
- BinaryExpr: Represents a binary expression with an operator and two operands.
- UnaryExpr: Represents a unary expression with an operator and one operand.
- SetExpr: Represents a set expression, typically used with 'in' operations.
- Identifier: Represents a variable or constant identifier, possibly with indices.

Functions:
- get_identifier_counts: Traverses expressions and counts identifiers.

Usage:
- Instantiate NogoodParser with a list of variable definitions.
- Use tokenize() to convert a string expression into tokens.
- Use parse() to parse tokens into an expression tree.

Dependencies:
- funcparserlib: Used for tokenization and parsing.
"""

from collections import Counter
from functools import reduce
from typing import Any, Dict, List, Sequence, Tuple, Union
from dataclasses import dataclass
from funcparserlib.lexer import make_tokenizer, Token, TokenSpec
from funcparserlib.parser import tok, many, finished, forward_decl, maybe


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


class NogoodParser:
    """
    Parser class to parse nogood expressions and build expression trees.

    Attributes:
        find_json (List[dict]): List of variable definitions used to resolve identifiers.

    Methods:
        tokenize(s): Tokenizes an input string into tokens.
        parse(tokens): Parses a sequence of tokens into an expression tree.
    """

    def __init__(self, find_json: List[Dict]):
        self.find_json = find_json

    def tokenize(self, s: str) -> List[Token]:
        """
        Tokenize the input string into a list of tokens.

        Args:
            s (str): The input string representing a nogood expression.

        Returns:
            List[Token]: A list of tokens generated from the input string.
        """
        specs = [
            TokenSpec("WS", r"\s+"),
            TokenSpec("INT", r"[+\-]?\d+"),
            TokenSpec("BOOL", r"true|false"),
            TokenSpec("SHIFT", r"shift"),
            TokenSpec("INT_TYPE", r"int"),
            TokenSpec("IN", r"in"),
            TokenSpec("COMMA", r","),
            TokenSpec("RANGE", r"\.\."),
            TokenSpec("NOT", r"!"),
            TokenSpec("LPAREN", r"\("),
            TokenSpec("RPAREN", r"\)"),
            TokenSpec("OR", r"\\/"),
            TokenSpec("ADD", r"\+|-"),
            TokenSpec("MUL", r"\*|/|%"),
            TokenSpec("RELATIONAL", r"<|>|<=|>="),
            TokenSpec("EQUALITY", r"==|!="),
            TokenSpec("ID", r"[a-zA-Z_]\w*"),
        ]
        tokenizer = make_tokenizer(specs)
        return [t for t in tokenizer(s) if t.type != "WS"]

    def parse(self, tokens: Sequence[Token]) -> List[Expr]:
        """
        Parse a sequence of tokens into an expression tree.

        Args:
            tokens (Sequence[Token]): The sequence of tokens to parse.

        Returns:
            List[Expr]: A list of expressions parsed from the tokens.
        """

        def id(s: str) -> Expr:
            """
            Resolve an identifier from a token string.

            Args:
                s (str): The token string representing the identifier.

            Returns:
                Expr: An Identifier expression with name and indices.
            """
            name = ""
            indices = []

            for find in self.find_json:
                find_name = find["name"]
                if s.startswith(find_name):
                    name = find_name
                    indices_str = s.replace(f"{name}", "")
                    if indices_str != "":
                        indices = [int(i) for i in indices_str[1:].split("_")]

            return Identifier(name, indices)

        def to_expr(args: Tuple[Expr, List[Tuple[str, Expr]]]) -> Expr:
            """
            Convert parsed tokens into a binary expression tree.

            This function specifically constructs a chain of BinaryExpr instances
            for binary operations like multiplication, addition, relational, and equality operators.

            Args:
                args (Tuple[Expr, List[Tuple[str, Expr]]]): The initial expression and a list of operator-expression pairs.

            Returns:
                Expr: The resulting binary expression tree.
            """
            first, rest = args
            result = first
            for op, next in rest:
                result = BinaryExpr(op, result, next)
            return result

        def flatten_numeric_list(
            args: Tuple[int | List[int], List[int | List[int]]]
        ) -> List[int]:
            """
            Flatten a nested list of numbers into a 1D list.
            Specifically the form received from the numeric_list parser.

            Args:
                args (Tuple[Union[int, List[int]], List[Union[int, List[int]]]]): The first element and the rest.

            Returns:
                List[int]: A flat list of integers.
            """

            def flatten(l: List[Any]) -> List[Any]:
                """Recursively flatten a nested list."""
                return reduce(
                    lambda x, y: x + flatten(y) if isinstance(y, list) else x + [y],
                    l,
                    [],
                )

            first, rest = args
            return (
                [first] + flatten(rest)
                if isinstance(first, int)
                else first + flatten(rest)
            )

        number = tok("INT") >> int
        numeric = (number + -tok("RANGE") + number) >> (
            lambda t: list(range(t[0], t[1] + 1))
        ) | number
        numeric_list = numeric + many(-tok("COMMA") + numeric) >> flatten_numeric_list
        expr = forward_decl()

        atom = number | tok("BOOL") >> (lambda t: t == "true") | tok("ID") >> id
        parenthesis_expr = -tok("LPAREN") + expr + -tok("RPAREN")
        shift_expr = (
            -tok("SHIFT")
            + -tok("LPAREN")
            + expr
            + -tok("COMMA")
            + number
            + -tok("RPAREN")
        ) >> (lambda t: BinaryExpr("+" if t[1] >= 0 else "-", t[0], abs(t[1])))
        unary_expr = ((tok("NOT") | tok("ADD", "-")) + expr) >> (
            lambda t: UnaryExpr(t[0], t[1])
        )
        primary = atom | parenthesis_expr | shift_expr | unary_expr

        mult_expr = (primary + many(tok("MUL") + primary)) >> to_expr
        add_expr = (mult_expr + many(tok("ADD") + mult_expr)) >> to_expr
        relational_expr = (add_expr + many(tok("RELATIONAL") + add_expr)) >> to_expr
        equality_expr = (
            relational_expr + many(tok("EQUALITY") + relational_expr)
        ) >> to_expr
        set_expr = (
            equality_expr
            + maybe(
                -tok("IN")
                + -tok("INT_TYPE")
                + -tok("LPAREN")
                + numeric_list
                + -tok("RPAREN")
            )
        ) >> (lambda t: SetExpr(t[0], t[1]) if t[1] != None else t[0])

        expr.define(set_expr)

        nogood = (expr + many(-tok("OR") + expr) + -finished) >> (
            lambda t: [t[0]] + t[1]
        )

        return nogood.parse(tokens)


def get_identifier_counts(exprs: List[Expr]) -> Counter[Identifier]:
    """
    Traverse expressions and count occurrences of identifiers.

    Args:
        exprs (List[Expr]): A list of expressions to traverse.

    Returns:
        Counter[Identifier]: A counter mapping identifiers to their occurrence counts.
    """
    identifiers = []
    stack = exprs

    while len(stack) > 0:
        current = stack.pop()
        if isinstance(current, Identifier):
            identifiers.append(current)
        elif isinstance(current, BinaryExpr):
            stack.extend([current.left, current.right])
        elif isinstance(current, UnaryExpr):
            stack.append(current.operand)
        elif isinstance(current, SetExpr):
            stack.append(current.expr)

    return Counter(identifiers)

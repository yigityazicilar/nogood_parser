from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List, NamedTuple, Tuple, Union
from dataclasses import dataclass
import logging, json, gzip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            ("MISMATCH", r"."),
        ]
        tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
        tokens = []
        for mo in re.finditer(tok_regex, s):
            kind = mo.lastgroup
            if kind == None:
                raise ValueError(f"The kind is None at position {mo.start()}")

            value = mo.group()
            pos = mo.start()
            match kind:
                case "WHITESPACE":
                    continue
                case "INT":
                    value = int(value)
                case "BOOL":
                    value = value == "true"
                case "IDENTIFIER":
                    indices = []
                    identifier_correct = False
                    for find in self.find_json:
                        find_name = find["name"]
                        if value.startswith(find_name):
                            indices_str = value.replace(f"{find_name}", "")
                            value = find_name
                            if indices_str != "":
                                indices = [int(i) for i in indices_str[1:].split("_")]
                            value = Identifier(value, indices)
                            identifier_correct = True
                            break
                    if not identifier_correct:
                        raise ValueError(
                            f"Unexpected identifier {value} at position {pos}"
                        )
                case "MISMATCH":
                    raise ValueError(f"Unexpected value {value} as position {pos}")
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


def parse_representation_objects(
    aux_path: Path, find_json_path: Path
) -> Tuple[Dict[int, str], Dict[int, Counter[Identifier]]]:
    """
    Parse representation objects from aux data and build a converted map.

    Parameters:
    - aux_path: Path to the auxiliary data file (compressed JSON).
    - find_json_path: Path to the find_json file containing variable information.

    Returns:
    - A tuple containing:
        - string_representation: A dictionary mapping variable IDs to their string representations.
        - identifier_counts: A dictionary mapping variable IDs to the number of times they are referenced in a single nogood.
    """
    with gzip.open(aux_path, "rt") as f_aux, find_json_path.open("r") as f_find_json:
        aux: Dict[str, Any] = json.load(f_aux)
        find_json: List[Dict[str, Any]] = json.load(f_find_json)
        f_aux.close()
        f_find_json.close()

    parser = NogoodParser(find_json)
    identifier_counts: Dict[int, Counter[Identifier]] = {}
    string_representation: Dict[int, str] = {}
    for key, obj in aux.items():
        if isinstance(obj, dict) and "representation" in obj:
            left_hand_side = obj.get("name", "")
            representation = obj.get("representation", "")

            try:
                parsed_nogood = parser.tokenize(left_hand_side)
            except Exception as e:
                logger.error(f"Failed to parse '{left_hand_side}' with error: {e}")
                raise

            increments = get_identifier_counts(parsed_nogood)

            pos_val, neg_val = "", ""
            pos_op, neg_op = "=", "="

            if representation == "2vals":
                neg_val = obj.get("val1", "")
                pos_val = obj.get("val2", "")
            elif representation == "order":
                pos_val = neg_val = obj.get("value", "")
                pos_op, neg_op = "<=", ">"
            else:
                pos_val = neg_val = obj.get("value", "")
                pos_op, neg_op = "=", "!="

            positive = f"{parsed_nogood}{pos_op}{pos_val}"
            negative = f"{parsed_nogood}{neg_op}{neg_val}"

            try:
                int_key = int(key)
                # string_representation[int_key] = positive
                # string_representation[-int_key] = negative
                identifier_counts[int_key] = increments
            except ValueError:
                logger.warning(f"Invalid key '{key}' in aux data. Skipping.")
                continue

    return string_representation, identifier_counts

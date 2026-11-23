"""Arithmetic calculator with an AST allowlist.

No ``eval`` of user text, no attribute access, no imports and no name lookup
beyond a fixed set of math functions. Expressions are length-limited and
results must be finite and bounded.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

from .base import Tool, ToolError, ToolResult

MAX_EXPRESSION_CHARS = 200
MAX_ABS_RESULT = 1e15
MAX_POWER = 100

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_MAX_FACTORIAL = 170


def _check_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"expected a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("result is not a finite number")
    if abs(value) > MAX_ABS_RESULT:
        raise ToolError(f"value exceeds the {MAX_ABS_RESULT:g} limit")
    return value


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"unsupported literal {node.value!r}")
        return _check_number(node.value)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY:
            raise ToolError(f"operator {op_type.__name__} is not allowed")
        left = _eval(node.left)
        right = _eval(node.right)
        if op_type is ast.Pow:
            if abs(right) > MAX_POWER:
                raise ToolError(f"exponent must be between -{MAX_POWER} and {MAX_POWER}")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ToolError("division by zero")
        return _check_number(_BINARY[op_type](left, right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY:
            raise ToolError(f"unary operator {op_type.__name__} is not allowed")
        return _check_number(_UNARY[op_type](_eval(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError("only the documented math functions may be called")
        if node.keywords:
            raise ToolError("keyword arguments are not supported")
        args = [_eval(arg) for arg in node.args]
        if node.func.id == "factorial":
            if len(args) != 1 or not float(args[0]).is_integer():
                raise ToolError("factorial expects one integer")
            if args[0] > _MAX_FACTORIAL:
                raise ToolError(f"factorial argument must be <= {_MAX_FACTORIAL}")
        if node.func.id == "round" and len(args) == 2:
            args[1] = int(args[1])
        return _check_number(_FUNCTIONS[node.func.id](*args))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    raise ToolError(f"syntax {type(node).__name__} is not allowed")


def evaluate(expression: str) -> float | int:
    expression = (expression or "").strip()
    if not expression:
        raise ToolError("expression is empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    value = _eval(tree)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    return value


class Calculator(Tool):
    name = "calculator"
    description = (
        "Evaluate an arithmetic expression exactly. Supports + - * / // % **, "
        "parentheses, math functions (sqrt, log, exp, sin, cos, tan, factorial, "
        "gcd, abs, round, min, max), and the constants pi, e, tau."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(17 * 23 + 4) / 5'.",
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str, **_: Any) -> ToolResult:
        try:
            value = evaluate(expression)
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            return ToolResult.failure(f"cannot evaluate: {exc}")
        content = f"{expression.strip()} = {value}"
        return ToolResult(ok=True, content=content, data={"expression": expression, "value": value})

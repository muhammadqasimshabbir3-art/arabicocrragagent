"""Casio-style scientific calculator tool for the Andromeda agent.

Provides safe numeric evaluation for arithmetic, logarithms, trigonometry,
complex numbers (iota), factorials, and other scientific functions so the LLM
does not attempt mental math.
"""

from __future__ import annotations

import cmath
import math
import re
from typing import Literal

import sympy as sp
from langchain.tools import tool
from sympy import E, I, N, factorial, log, pi, sqrt

from agent.async_utils import run_in_thread

_ANGLE_MODE = "DEG"


def _set_angle_mode(mode: str) -> None:
    global _ANGLE_MODE
    _ANGLE_MODE = mode.upper()


def _to_radians(value: sp.Expr) -> sp.Expr:
    if _ANGLE_MODE == "DEG":
        return sp.rad(value)
    return value


def _sin(x: sp.Expr) -> sp.Expr:
    return sp.sin(_to_radians(x))


def _cos(x: sp.Expr) -> sp.Expr:
    return sp.cos(_to_radians(x))


def _tan(x: sp.Expr) -> sp.Expr:
    return sp.tan(_to_radians(x))


def _asin(x: sp.Expr) -> sp.Expr:
    result = sp.asin(x)
    return result * 180 / pi if _ANGLE_MODE == "DEG" else result


def _acos(x: sp.Expr) -> sp.Expr:
    result = sp.acos(x)
    return result * 180 / pi if _ANGLE_MODE == "DEG" else result


def _atan(x: sp.Expr) -> sp.Expr:
    result = sp.atan(x)
    return result * 180 / pi if _ANGLE_MODE == "DEG" else result


def _log10(x: sp.Expr) -> sp.Expr:
    """Casio LOG button: base-10 logarithm."""
    return log(x, 10)


def _ln(x: sp.Expr) -> sp.Expr:
    """Casio LN button: natural logarithm."""
    return log(x)


def _normalize_expression(expression: str) -> str:
    """Normalize user/LLM input into a sympy-friendly expression."""
    expr = expression.strip()
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.replace("√", "sqrt")
    expr = re.sub(r"\bpi\b", "PI", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\biota\b", "I", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bj\b", "I", expr)
    expr = re.sub(r"(?<=[\d.])i\b", "*I", expr)
    expr = re.sub(r"\bi(?=[\d.])", "I*", expr)
    expr = re.sub(r"\bi\b", "I", expr)
    expr = re.sub(r"\blog10\s*\(", "log(", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bln\s*\(", "ln(", expr, flags=re.IGNORECASE)
    return expr


def _format_complex(value: complex, output_format: str) -> str:
    if output_format == "polar":
        radius = abs(value)
        angle = cmath.phase(value)
        if _ANGLE_MODE == "DEG":
            angle = math.degrees(angle)
            return f"{radius:.10g}∠{angle:.10g}°"
        return f"{radius:.10g}∠{angle:.10g} rad"

    real = value.real
    imag = value.imag
    if abs(imag) < 1e-12:
        return f"{real:.10g}"
    if abs(real) < 1e-12:
        if abs(imag - 1) < 1e-12:
            return "i"
        if abs(imag + 1) < 1e-12:
            return "-i"
        return f"{imag:.10g}i"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.10g} {sign} {abs(imag):.10g}i"


def _is_invalid_numeric(value: complex) -> bool:
    return not math.isfinite(value.real) or not math.isfinite(value.imag)


def _format_result(result: sp.Expr, output_format: str) -> str:
    numeric = N(result)
    as_complex = complex(numeric.evalf())
    if _is_invalid_numeric(as_complex):
        raise ValueError("result is undefined (NaN or infinity)")

    if numeric.is_real:
        value = as_complex.real
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.10g}"

    return _format_complex(as_complex, output_format)


def evaluate_casio_expression(
    expression: str,
    angle_mode: Literal["DEG", "RAD"] = "DEG",
    output_format: Literal["rectangular", "polar"] = "rectangular",
) -> str:
    """Evaluate a Casio-style scientific calculator expression."""
    raw = (expression or "").strip()
    if not raw:
        return "Calculator error for '': expression is empty."

    lowered = raw.lower()
    # Reject SQL / column refs that the analytics LLM sometimes passes by mistake.
    if (
        re.search(r"\b(select|from|where|join|group\s+by|order\s+by)\b", lowered)
        or re.search(
            r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b",
            raw,
        )  # table.column (not decimals like 2.5)
        or re.search(
            r"\b(line_total|total_amount|unit_price|stock_qty|order_items|products)\b",
            lowered,
        )
    ):
        return (
            f"Calculator error for '{expression}': "
            "pass a pure numeric expression only (e.g. '(15200 - 9800) / 15200 * 100'). "
            "Do not pass SQL, column names, or table.field references — "
            "those belong in query_store_database."
        )

    _set_angle_mode(angle_mode)
    normalized = _normalize_expression(expression)

    local_dict = {
        "PI": pi,
        "E": E,
        "I": I,
        "sqrt": sqrt,
        "log": _log10,
        "ln": _ln,
        "sin": _sin,
        "cos": _cos,
        "tan": _tan,
        "asin": _asin,
        "acos": _acos,
        "atan": _atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "factorial": factorial,
        "abs": sp.Abs,
        "exp": sp.exp,
    }

    try:
        parsed = sp.sympify(
            normalized,
            locals=local_dict,
            convert_xor=True,
            rational=True,
        )
        result = _format_result(parsed, output_format)
        return (
            f"Expression: {expression}\n"
            f"Angle mode: {angle_mode}\n"
            f"Result: {result}"
        )
    except Exception as exc:  # noqa: BLE001 — never crash the agent graph
        return f"Calculator error for '{expression}': {exc}"


def _parse_problem_lines(problems: str) -> list[str]:
    """Extract individual math expressions from a multi-problem input."""
    expressions: list[str] = []
    label_only = re.compile(
        r"^(?:without approximation,?\s*)?"
        r"(?:evaluate\s+exactly|evaluate|what is|solve|exactly)\s*:?\s*$",
        re.IGNORECASE,
    )

    for raw_line in problems.splitlines():
        line = raw_line.strip()
        if not line or label_only.match(line):
            continue

        line = re.sub(
            r"^(?:\d+[\.\):\-]\s*|[-*]\s*|"
            r"(?:without approximation,?\s*)?"
            r"(?:evaluate\s+exactly|evaluate|what is|solve)\s*:?\s*"
            r"|exactly\s*:?\s*)+",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        if line and not label_only.match(line):
            for part in _split_comma_expressions(line):
                if _looks_like_math_expression(part):
                    expressions.append(part)

    return expressions


def _split_comma_expressions(blob: str) -> list[str]:
    """Split comma-separated expressions, respecting parentheses."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in blob:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if current:
        parts.append("".join(current).strip())

    cleaned: list[str] = []
    for part in parts:
        part = re.sub(r"(?i)^and\s+", "", part.strip()).rstrip(".")
        if part:
            cleaned.append(part)
    return cleaned


def _looks_like_math_expression(text: str) -> bool:
    """Return True if text looks like a calculable math expression."""
    candidate = text.strip().rstrip(".")
    if not candidate or len(candidate) < 2:
        return False

    prose_words = re.findall(r"[a-zA-Z]{5,}", candidate)
    math_markers = re.findall(
        r"\b(log|ln|sin|cos|tan|asin|acos|atan|sqrt|factorial|exp|abs)\b|[\d+\-*/^()]",
        candidate,
        flags=re.IGNORECASE,
    )
    if not math_markers or not re.search(r"\d", candidate):
        return False

    if len(prose_words) > 2 and not re.search(
        r"\b(log|ln|sin|cos|tan|sqrt|factorial)\s*\(",
        candidate,
        flags=re.IGNORECASE,
    ):
        return False

    return True


def extract_math_expressions(text: str) -> list[str]:
    """Pull math expressions from natural-language multi-task prompts."""
    working = text

    working = re.sub(
        r"(?i)your first task is to introduce yourself,?\s*then\s*",
        "",
        working,
    )
    working = re.sub(
        r"(?i)i am practicing.*?final answers:\s*",
        "",
        working,
        flags=re.DOTALL,
    )
    working = re.sub(r"(?i)create a pdf.*", "", working, flags=re.DOTALL)
    working = re.sub(r"(?i)(then\s*)?i want.*email.*", "", working, flags=re.DOTALL)
    working = re.sub(r"(?i)send.*email.*", "", working, flags=re.DOTALL)

    expressions: list[str] = []
    for line in working.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in _split_comma_expressions(line):
            cleaned = re.sub(
                r"^(?:\d+[\.\):\-]\s*|[-*]\s*|"
                r"(?:without approximation,?\s*)?"
                r"(?:evaluate\s+exactly|evaluate|what is|solve)\s*:?\s*"
                r"|exactly\s*:?\s*)+",
                "",
                part,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned and _looks_like_math_expression(cleaned):
                expressions.append(cleaned)

    if expressions:
        return expressions

    return [
        line
        for line in _parse_problem_lines(text)
        if _looks_like_math_expression(line)
    ]


def _result_value(expression: str, angle_mode: str) -> str:
    """Return only the numeric result for an expression."""
    output = evaluate_casio_expression(expression, angle_mode=angle_mode)
    if "Result: " in output:
        return output.split("Result: ", 1)[1].strip()
    if "Calculator error" in output:
        return output.split("Calculator error for ", 1)[1]
    return output


def solve_math_batch(
    problems: str,
    angle_mode: Literal["DEG", "RAD"] = "DEG",
) -> str:
    """Evaluate multiple math expressions in one calculator pass."""
    expressions = _parse_problem_lines(problems)
    if not expressions:
        return "No math expressions found. Provide one expression per line."

    lines = [f"Solved {len(expressions)} problem(s) (angle mode: {angle_mode}):", ""]
    for index, expression in enumerate(expressions, start=1):
        value = _result_value(expression, angle_mode)
        lines.append(f"{index}. {expression}")
        lines.append(f"   = {value}")
        lines.append("")

    return "\n".join(lines).strip()


@tool
async def solve_math_batch_tool(
    problems: str,
    angle_mode: Literal["DEG", "RAD"] = "DEG",
) -> str:
    """Solve multiple math problems at once with the Casio calculator.

    ALWAYS use this tool when the user gives more than one math problem to solve.
    Pass each expression on its own line. Prefixes like "Evaluate:", "What is:",
    and numbering are automatically stripped.

    Examples input:
    log(2500) + ln(7)
    sin(73.5) * cos(41.2) + tan(12.7)
    (3 + 4i)**7

    Args:
        problems: Newline-separated list of math expressions to evaluate.
        angle_mode: 'DEG' for degrees (Casio default) or 'RAD' for radians.

    Returns:
        Numbered list of exact calculator results for every problem.
    """
    return await run_in_thread(solve_math_batch, problems, angle_mode)


@tool
async def casio_calculator(
    expression: str,
    angle_mode: Literal["DEG", "RAD"] = "DEG",
    output_format: Literal["rectangular", "polar"] = "rectangular",
) -> str:
    """Evaluate math using a Casio-style scientific calculator.

    ALWAYS use this tool for any arithmetic, algebra, logarithm, trigonometry,
    power, root, factorial, or complex-number (iota) calculation. Never compute
    math yourself.

    Casio conventions:
    - log(x) means base-10 logarithm (LOG button)
    - ln(x) means natural logarithm (LN button)
    - Use i or j for imaginary unit (e.g. 2+3i, sqrt(-1))
    - Trig functions use degrees by default (angle_mode='DEG')
    - Supports sin, cos, tan, asin, acos, atan, sqrt, factorial, exp, abs

    Examples:
    - "2 + 3 * 4"
    - "log(1000)"
    - "ln(e)"
    - "sin(30)"
    - "sqrt(-1)"
    - "2**10"
    - "factorial(5)"
    - "(1+i)**2"

    Args:
        expression: The mathematical expression to evaluate.
        angle_mode: 'DEG' for degrees (Casio default) or 'RAD' for radians.
        output_format: 'rectangular' (a+bi) or 'polar' (r∠θ°) for complex results.

    Returns:
        The computed numeric result as a formatted string.
    """
    return await run_in_thread(
        evaluate_casio_expression, expression, angle_mode, output_format
    )


__all__ = [
    "casio_calculator",
    "solve_math_batch_tool",
    "evaluate_casio_expression",
    "solve_math_batch",
    "extract_math_expressions",
]

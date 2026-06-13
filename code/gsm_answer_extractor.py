"""Deterministic numeric answer extraction shared by GSM evaluations."""

from __future__ import annotations

import re
from fractions import Fraction


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?"
    r"(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?"
)
LATEX_FRAC_PATTERN = re.compile(
    r"(-?)\\frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
CONCLUSION_PATTERN = re.compile(
    r"^(?:so|therefore|thus|hence|consequently|together|finally)\b",
    re.IGNORECASE,
)
ANSWER_PREDICATE_PATTERNS = (
    re.compile(
        r"\b(?:will|would|can|should)\s+"
        r"(?:be|have|eat|make|need|run|spend|get|receive|pay|save|take)\b(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:spends?|costs?|takes?|needs?|earns?|makes?|saves?|receives?|"
        r"gets?|pays?|has|have|had|was|were|is|are)\b(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:total|answer|result|amount|number|difference|combined weight)"
        r"\b[^0-9+\-]*(.+)",
        re.IGNORECASE,
    ),
)


def normalize_numeric_text(text: str) -> str:
    def replace_frac(match: re.Match[str]) -> str:
        leading_sign, numerator, denominator = match.groups()
        if leading_sign and not numerator.startswith("-"):
            numerator = f"-{numerator.lstrip('+')}"
        return f"{numerator}/{denominator}"

    return LATEX_FRAC_PATTERN.sub(replace_frac, text)


def _numbers(text: str | None) -> list[str]:
    if text is None:
        return []
    normalized = normalize_numeric_text(str(text)).replace(",", "")
    return [match.replace(" ", "") for match in NUMBER_PATTERN.findall(normalized)]


def extract_number(text: str | None) -> str | None:
    numbers = _numbers(text)
    return numbers[-1] if numbers else None


def extract_first_number(text: str | None) -> str | None:
    numbers = _numbers(text)
    return numbers[0] if numbers else None


def truncate_generated_continuation(text: str) -> str:
    continuation = re.search(r"\n\s*(?:Question:|Q:)\s+", text)
    return text[: continuation.start()] if continuation else text


def extract_balanced_brace_contents(text: str, command: str) -> list[str]:
    contents: list[str] = []
    command_pattern = re.compile(rf"\\?{re.escape(command)}\s*\{{", re.IGNORECASE)
    for match in command_pattern.finditer(text):
        depth = 1
        start = match.end()
        pos = start
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            contents.append(text[start : pos - 1])
    return contents


def split_answer_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。])\s+|\n+", text)
        if sentence.strip()
    ]


def _extract_last_marked_answer(text: str, pattern: re.Pattern[str]) -> str | None:
    for match in reversed(list(pattern.finditer(text))):
        number = extract_first_number(match.group(1))
        if number is not None:
            return number
    return None


def _extract_equation_result(sentence: str) -> str | None:
    if "=" not in sentence:
        return None
    return extract_first_number(sentence.rsplit("=", 1)[-1])


def extract_conclusion_answer(text: str) -> str | None:
    sentences = split_answer_sentences(text)
    if sentences:
        number = _extract_equation_result(sentences[-1])
        if number is not None:
            return number

    for sentence in reversed(sentences[-2:]):
        if not CONCLUSION_PATTERN.match(sentence):
            continue
        number = _extract_equation_result(sentence)
        if number is not None:
            return number
        for pattern in ANSWER_PREDICATE_PATTERNS:
            match = pattern.search(sentence)
            if match:
                number = extract_first_number(match.group(1))
                if number is not None:
                    return number
        number = extract_number(sentence)
        if number is not None:
            return number
    return None


def extract_answer(output: str | None) -> str | None:
    text = truncate_generated_continuation(str(output or ""))
    marker_patterns = (
        re.compile(r"(?:the\s+)?final answer(?:\s+is)?\s*[:=]?\s*([^\n]+)", re.IGNORECASE),
        re.compile(r"####\s*([^\n]+)"),
    )
    for pattern in marker_patterns:
        number = _extract_last_marked_answer(text, pattern)
        if number is not None:
            return number

    for boxed_content in reversed(extract_balanced_brace_contents(text, "boxed")):
        number = extract_first_number(boxed_content)
        if number is not None:
            return number

    number = extract_conclusion_answer(text)
    if number is not None:
        return number

    number = _extract_last_marked_answer(
        text,
        re.compile(r"(?:the\s+)?answer(?:\s+is)?\s*[:=]\s*([^\n]+)", re.IGNORECASE),
    )
    if number is not None:
        return number
    return extract_number(text)


def normalize_answer(answer: str | None) -> Fraction | None:
    number = extract_answer(answer)
    if number is None:
        return None
    cleaned = number.replace("$", "").replace("%", "").rstrip(".。;:，,")
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(pred_answer: str | None, gold_answer: str, tolerance: float = 1e-6) -> bool:
    pred_value = normalize_answer(pred_answer)
    gold_value = normalize_answer(gold_answer)
    if pred_value is None or gold_value is None:
        return False
    tolerance_value = Fraction(str(tolerance))
    return abs(pred_value - gold_value) <= max(
        tolerance_value,
        tolerance_value * abs(gold_value),
    )

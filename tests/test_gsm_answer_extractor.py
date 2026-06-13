from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from gsm_answer_extractor import extract_answer, is_correct  # noqa: E402


def test_explicit_answer_markers_have_highest_priority() -> None:
    assert extract_answer("Reasoning 12.\n#### 1,234") == "1234"
    assert extract_answer("The final answer is $75.00 over 30 days.") == "75.00"
    assert extract_answer(r"Therefore, \boxed{-\frac{7}{3}}.") == "-7/3"


def test_generated_next_question_is_ignored() -> None:
    output = (
        "300 + 240 = 540 seconds.\n#### 540\n\n"
        "Question: A generated problem?\nAnswer: 100 * 4 = 400\n#### 400"
    )
    assert extract_answer(output) == "540"


def test_last_sentence_equation_uses_right_hand_result() -> None:
    assert extract_answer(
        "Since there are 12 eggs in a dozen, Claire will eat 84 / 12 = 7 dozens of eggs."
    ) == "7"
    assert extract_answer(
        "If they have 24 slices of bread, they can last 24 / 6 = 4 days."
    ) == "4"


def test_conclusion_predicate_ignores_trailing_duration() -> None:
    assert extract_answer("So, Terry spends $75.00 on yogurt over 30 days.") == "75.00"
    assert extract_answer(
        "Therefore, Mel will save 81 kilowatts of electric energy in 30 days."
    ) == "81"


def test_conclusion_predicate_ignores_trailing_input_quantities() -> None:
    assert extract_answer(
        "So, it will take Matthew 4 hours to dig 30 small holes and 15 large holes."
    ) == "4"
    assert extract_answer(
        "So, Tim makes a profit of $60 selling 10 jars of the honey and jam mix."
    ) == "60"


def test_conclusion_predicate_ignores_trailing_target_total() -> None:
    assert extract_answer("So, it will take 12 days for the bamboo to grow to 600 inches.") == "12"
    assert extract_answer("So it will take 9 minutes for the light to blink 459 times.") == "9"


def test_conclusion_equation_keeps_right_hand_total() -> None:
    assert extract_answer("Together, Lily and Amy have 50 + 70 = 120 friends.") == "120"
    assert extract_answer(
        "Therefore, Lloyd earned a total of $50 + $80 = $130 for the first two weeks."
    ) == "130"


def test_numeric_comparison_supports_decimals_and_fractions() -> None:
    assert is_correct("4.00", "4")
    assert is_correct(r"\boxed{\frac{1}{2}}", "0.5")

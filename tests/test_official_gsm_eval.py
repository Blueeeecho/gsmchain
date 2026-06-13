from __future__ import annotations

import sys
import os
from collections import UserDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from eval_official_gsm import (  # noqa: E402
    SUPPORTED_PROMPT_STYLES,
    configure_runtime_env,
    build_fewshot_prompt,
    build_lm_eval_llama_messages,
    build_model_input,
    extract_answer,
    is_correct,
    load_gsm8k_jsonl,
    load_gsmplus_rows,
    parse_model_args,
)


def test_extract_answer_prefers_standard_markers_and_normalizes() -> None:
    assert extract_answer("reasoning\n#### 1,234") == "1234"
    assert extract_answer("The answer is \\boxed{18}.") == "18"
    assert is_correct("18.0", "18")


def test_extract_answer_handles_decimal_final_answer_marker() -> None:
    assert extract_answer("The final answer is 4.33.") == "4.33"
    assert extract_answer("Final answer: $1,234.50.") == "1234.50"


def test_extract_answer_handles_boxed_latex_fractions() -> None:
    assert extract_answer(r"Therefore, \boxed{\frac{1}{2}}.") == "1/2"
    assert extract_answer(r"Therefore, \boxed{-\frac{7}{3}}.") == "-7/3"


def test_extract_answer_falls_back_to_last_number() -> None:
    assert extract_answer("First compute 12. Then after subtracting 5, we get 7.") == "7"


def test_extract_answer_prefers_first_number_in_final_conclusion_sentence() -> None:
    assert extract_answer("Therefore, 84 / 12 = 7 dozens. So, Claire will eat 7 dozens in 4 weeks.") == "7"
    assert extract_answer("Therefore, the candle will be 8 centimeters shorter from 1:00 PM to 5:00 PM.") == "8"
    assert extract_answer("Therefore, Lloyd earned a total of $50 + $80 = $130 for the first two weeks.") == "130"
    assert extract_answer("So, after 3 years, Brenda will have $975 in total.") == "975"


def test_extract_answer_uses_last_repeated_standard_marker() -> None:
    output = "2x + 10 = 110\n#### 2x = 100\nx = 50\n#### 50"
    assert extract_answer(output) == "50"


def test_extract_answer_ignores_generated_next_question() -> None:
    output = (
        "300 + 240 = 540 seconds.\n#### 540\n\n"
        "Question: A new generated problem?\nAnswer: 100 * 4 = 400\n#### 400"
    )
    assert extract_answer(output) == "540"


def test_build_fewshot_prompt_uses_eight_cot_examples() -> None:
    prompt = build_fewshot_prompt("How many apples?")
    assert prompt.count("Question:") == 9
    assert prompt.count("Answer:") == 9
    assert "Let's think step by step." in prompt
    assert prompt.rstrip().endswith("Answer: Let's think step by step.")


def test_build_lm_eval_llama_messages_uses_multiturn_fewshot() -> None:
    messages = build_lm_eval_llama_messages("How many apples?")
    assert len(messages) == 17
    assert [message["role"] for message in messages[:4]] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"].startswith("Given the following problem")
    assert messages[1]["content"].endswith("The final answer is 6")
    assert messages[-1]["role"] == "user"
    assert "Problem: How many apples?" in messages[-1]["content"]


def test_build_model_input_tokenizes_llama_prompt_without_duplicate_bos() -> None:
    class FakeTokenizer:
        bos_token_id = 128000
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert len(messages) == 17
            assert tokenize is True
            assert add_generation_prompt is True
            return UserDict({"input_ids": [self.bos_token_id, 128006, 9125]})

    model_input = build_model_input(
        FakeTokenizer(),
        "How many apples?",
        prompt_style="lm_eval_llama_chat_multiturn",
    )
    assert model_input == [128000, 128006, 9125]
    assert model_input.count(FakeTokenizer.bos_token_id) == 1


def test_only_successful_prompt_styles_are_supported() -> None:
    assert SUPPORTED_PROMPT_STYLES == ("chat", "lm_eval_llama_chat_multiturn")

    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "rendered"

    try:
        build_model_input(FakeTokenizer(), "How many apples?", prompt_style="chat_fewshot")
    except ValueError as exc:
        assert "Unsupported prompt style" in str(exc)
    else:
        raise AssertionError("Obsolete prompt style should be rejected")


def test_default_model_matches_default_qwen_chat_prompt() -> None:
    assert list(parse_model_args([])) == ["Qwen2.5-0.5B-Instruct"]


def test_load_gsm8k_jsonl_reads_local_test_format(tmp_path: Path) -> None:
    data_path = tmp_path / "gsm8k.jsonl"
    data_path.write_text(
        '{"base_id":"gsm8k_test_000001","question":"Q?","answer":"x #### 42","final_answer":"42"}\n',
        encoding="utf-8",
    )
    rows = load_gsm8k_jsonl(data_path)
    assert rows[0].id == "gsm8k_test_000001"
    assert rows[0].split == "original"
    assert rows[0].answer == "42"


def test_load_gsmplus_rows_filters_rephrase_and_distract() -> None:
    raw = [
        {"question": "O", "answer": "#### 1", "perturbation_type": "original question"},
        {"question": "R", "answer": "#### 2", "perturbation_type": "problem understanding"},
        {"question": "D", "answer": "#### 3", "perturbation_type": "distraction insertion"},
    ]
    rephrase = load_gsmplus_rows(raw, split_name="rephrase")
    distract = load_gsmplus_rows(raw, split_name="distract")
    assert [row.question for row in rephrase] == ["R"]
    assert [row.question for row in distract] == ["D"]


def test_configure_runtime_env_prepends_python_env_bin() -> None:
    old_path = os.environ.get("PATH")
    old_cuda_home = os.environ.get("CUDA_HOME")
    try:
        os.environ["PATH"] = "/usr/bin"
        os.environ.pop("CUDA_HOME", None)
        configure_runtime_env("/tmp/myenv/bin/python")
        assert "/tmp/myenv/bin" == os.environ["PATH"].split(":")[0]
        assert os.environ["CUDA_HOME"] == "/tmp/myenv"
    finally:
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path
        if old_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = old_cuda_home

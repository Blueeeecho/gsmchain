from __future__ import annotations

import sys
from collections import UserDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from eval_chaingsm_base_8shot import (  # noqa: E402
    DEFAULT_MODELS,
    LLAMA_PROFILE,
    QWEN_MATH_COMPLETION_PROFILE,
    QWEN_MULTITURN_PROFILE,
    build_llama_messages,
    build_model_input,
    build_qwen_math_completion_prompt,
    build_qwen_messages,
    filter_gpu_memory_candidates,
    select_prompt_profile,
    stop_sequences_for_profile,
)


def test_default_models_are_the_four_requested_qwen_checkpoints() -> None:
    assert [model.name for model in DEFAULT_MODELS] == [
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-Math-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
    ]


def test_select_prompt_profile_routes_all_general_qwen_sizes_identically() -> None:
    for model_name in (
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
    ):
        assert select_prompt_profile(model_name) == QWEN_MULTITURN_PROFILE


def test_select_prompt_profile_routes_qwen_math_to_completion() -> None:
    assert (
        select_prompt_profile("Qwen2.5-Math-1.5B-Instruct")
        == QWEN_MATH_COMPLETION_PROFILE
    )


def test_select_prompt_profile_still_supports_llama_when_selected_explicitly() -> None:
    assert select_prompt_profile("Llama-3.2-1B-Instruct") == LLAMA_PROFILE


def test_qwen_messages_use_eight_multiturn_examples() -> None:
    messages = build_qwen_messages("How many apples?")
    assert len(messages) == 18
    assert messages[0] == {
        "role": "system",
        "content": (
            "As an expert problem solver, solve step by step the following "
            "mathematical questions."
        ),
    }
    assert [message["role"] for message in messages[1:5]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[-1] == {
        "role": "user",
        "content": "Q: How many apples?\nA: Let's think step by step.",
    }


def test_llama_messages_use_multiturn_fewshot() -> None:
    messages = build_llama_messages("How many apples?")
    assert len(messages) == 17
    assert [message["role"] for message in messages[:4]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[-1]["role"] == "user"
    assert "Problem: How many apples?" in messages[-1]["content"]


def test_qwen_model_input_is_rendered_chat_string() -> None:
    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert len(messages) == 18
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered-qwen-prompt"

    assert (
        build_model_input(FakeTokenizer(), "Q?", QWEN_MULTITURN_PROFILE)
        == "rendered-qwen-prompt"
    )


def test_qwen_math_completion_does_not_use_chat_template() -> None:
    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            raise AssertionError("Math completion profile must bypass chat template")

    prompt = build_model_input(FakeTokenizer(), "How many apples?", QWEN_MATH_COMPLETION_PROFILE)
    assert prompt == build_qwen_math_completion_prompt("How many apples?")
    assert prompt.count("\nQ:") == 8
    assert prompt.count("\nA:") == 9
    assert prompt.startswith("Q:")
    assert prompt.rstrip().endswith("A: Let's think step by step.")
    assert "You are a helpful assistant." not in prompt
    assert "<|im_start|>" not in prompt


def test_llama_model_input_is_single_bos_token_ids() -> None:
    class FakeTokenizer:
        bos_token_id = 128000
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert len(messages) == 17
            assert tokenize is True
            assert add_generation_prompt is True
            return UserDict({"input_ids": [128000, 128006, 9125]})

    model_input = build_model_input(FakeTokenizer(), "Q?", LLAMA_PROFILE)
    assert model_input == [128000, 128006, 9125]
    assert model_input.count(128000) == 1


def test_llama_stop_sequences_cover_chat_and_generated_next_question() -> None:
    stops = stop_sequences_for_profile(LLAMA_PROFILE)
    assert "<|eot_id|>" in stops
    assert "<|start_header_id|>user<|end_header_id|>" in stops
    assert "Q:" in stops
    assert stop_sequences_for_profile(QWEN_MULTITURN_PROFILE) == ["<|im_end|>"]


def test_qwen_math_stop_sequences_prevent_generating_another_question() -> None:
    assert stop_sequences_for_profile(QWEN_MATH_COMPLETION_PROFILE) == [
        "\nQ:",
        "\nQuestion:",
        "<|im_end|>",
    ]


def test_gpu_memory_candidates_are_filtered_below_current_free_ratio() -> None:
    assert filter_gpu_memory_candidates(
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25],
        available_ratio=0.681,
        safety_margin=0.03,
    ) == [0.6, 0.5, 0.4, 0.3, 0.25]


def test_gpu_memory_candidates_keep_a_low_fallback() -> None:
    assert filter_gpu_memory_candidates(
        [0.8, 0.7, 0.6],
        available_ratio=0.22,
        safety_margin=0.03,
        min_candidate=0.1,
    ) == [0.19]

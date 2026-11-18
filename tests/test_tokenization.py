"""Test Day 01: tokenization, special token, chat template, context window, biaya.

Semua test berjalan offline dan tidak membutuhkan API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "day01_tokenization"))

import bpe  # noqa: E402
import chat_template  # noqa: E402
import context_window  # noqa: E402
import special_tokens  # noqa: E402
import token_cost  # noqa: E402
import tokenizer_comparison  # noqa: E402

CORPUS = "the cat sat on the mat the cat ran away"


# --------------------------------------------------------------- BPE
def test_bpe_training_reduces_token_count():
    model = bpe.BPE.train(CORPUS, num_merges=10)
    assert model.vocab_size > len(set(CORPUS))
    assert len(model.merges) == 10
    assert len(model.encode("the cat")) < len("the cat")


def test_bpe_roundtrip():
    model = bpe.BPE.train(CORPUS, num_merges=10)
    for text in ["the cat", "cat sat on mat", "the cat ran away"]:
        assert model.decode(model.encode(text)) == text


def test_bpe_merges_are_ordered_pairs_of_symbols():
    model = bpe.BPE.train(CORPUS, num_merges=5)
    for left, right in model.merges:
        assert isinstance(left, str) and isinstance(right, str)


def test_merge_pair_is_non_overlapping():
    assert bpe.merge_pair(("a", "a", "a"), ("a", "a")) == ("aa", "a")
    assert bpe.merge_pair(("a", "b", "c"), ("b", "c")) == ("a", "bc")


# ------------------------------------------------------- tokenizer compare
def test_char_and_word_tokenizers():
    assert tokenizer_comparison.char_tokens("abc") == ["a", "b", "c"]
    assert tokenizer_comparison.word_tokens("a b  c") == ["a", "b", "c"]


def test_compression_ratio_is_chars_per_token():
    assert tokenizer_comparison.compression_ratio("abcdef", 3) == pytest.approx(2.0)
    assert tokenizer_comparison.compression_ratio("", 0) == 0.0


# --------------------------------------------------------- special tokens
def test_special_token_is_encoded_as_single_id():
    tok = special_tokens.ToyTokenizer(["<|im_end|>"])
    ids = tok.encode("halo<|im_end|>dunia")
    assert len(ids) == 3
    assert ids[1] == tok.stoi["<|im_end|>"]


def test_special_tokens_can_be_disabled():
    tok = special_tokens.ToyTokenizer(["<|im_end|>"])
    ids = tok.encode("a<|im_end|>b", allow_special=False)
    assert all(i != tok.stoi["<|im_end|>"] for i in ids)


def test_build_labels_masks_prompt_and_padding():
    labels = special_tokens.build_labels([5, 6, 7, 0, 0], prompt_len=2, pad_id=0)
    assert labels == [-100, -100, 7, -100, -100]
    assert special_tokens.masked_token_count(labels) == 1


# --------------------------------------------------------- chat template
def test_chatml_contains_roles_and_markers():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    text = chat_template.render_chatml(messages)
    assert text.startswith("<|bos|>")
    assert "<|im_start|>system\nsys<|im_end|>" in text
    assert "<|im_start|>user\nhi<|im_end|>" in text


def test_empty_system_prompt_is_skipped():
    messages = [{"role": "system", "content": "  "}, {"role": "user", "content": "hi"}]
    text = chat_template.render_chatml(messages)
    assert "<|im_start|>system" not in text


def test_plain_rendering_has_no_markup():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert chat_template.render_plain(messages) == "a b"


# --------------------------------------------------------- context window
def test_fit_to_budget_respects_every_strategy():
    messages = [
        {"role": "system", "content": "sys " * 20},
        {"role": "user", "content": "important 4271"},
        {"role": "assistant", "content": "ok " * 60},
        {"role": "user", "content": "latest question " * 30},
    ]
    for strategy in ["drop_oldest", "keep_system", "head_tail", "truncate_content"]:
        result = context_window.fit_to_budget(messages, budget=120, strategy=strategy)
        assert result["tokens"] <= 120
        assert result["fits"] is True


def test_keep_system_preserves_system_message():
    messages = [
        {"role": "system", "content": "keep me"},
        {"role": "user", "content": "x " * 200},
        {"role": "user", "content": "y"},
    ]
    result = context_window.fit_to_budget(messages, budget=50, strategy="keep_system")
    assert result["kept"][0]["role"] == "system"
    assert result["kept"][0]["content"] == "keep me"


def test_truncate_content_keeps_message_count():
    messages = [{"role": "user", "content": "word " * 200}]
    result = context_window.fit_to_budget(messages, budget=50, strategy="truncate_content")
    assert len(result["kept"]) == 1
    assert result["tokens"] <= 50


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        context_window.fit_to_budget([], 10, "nope")


def test_attention_cost_is_quadratic():
    assert context_window.attention_cost(10) == 55
    assert context_window.attention_cost(100) / context_window.attention_cost(50) > 3.5


# ------------------------------------------------------------ token cost
def test_cost_is_linear_in_tokens():
    one = token_cost.cost_usd(1000, 1000, "qwen-plus")
    two = token_cost.cost_usd(2000, 2000, "qwen-plus")
    assert two["total"] == pytest.approx(2 * one["total"])


def test_output_tokens_are_more_expensive():
    assert token_cost.PRICES["qwen-plus"][1] > token_cost.PRICES["qwen-plus"][0]
    result = token_cost.cost_usd(0, 1_000_000, "qwen-plus")
    assert result["total"] == pytest.approx(1.20)


def test_cached_prompt_is_cheaper():
    normal = token_cost.cost_usd(10_000, 100, "qwen-plus")["total"]
    cached = token_cost.cost_with_cache(10_000, 9_000, 100, "qwen-plus")
    assert cached < normal


def test_cache_validation():
    with pytest.raises(ValueError):
        token_cost.cost_with_cache(100, 200, 10, "qwen-plus")


def test_monthly_cost_scales_with_requests():
    daily = token_cost.monthly_cost(1, 1000, 500, "qwen-turbo")
    assert daily == pytest.approx(30 * token_cost.cost_usd(1000, 500, "qwen-turbo")["total"])

# ------------------------------------------------------- edge case (review)
def test_bpe_rejects_unseen_characters():
    model = bpe.BPE.train("abc ab", num_merges=2)
    with pytest.raises(ValueError, match="tidak ada di vocabulary"):
        model.encode("abz")


def test_bpe_lenient_mode_drops_unknown():
    model = bpe.BPE.train("abc ab", num_merges=2)
    ids = model.encode("abz", strict=False)
    assert len(ids) >= 2
    assert model.decode(ids) == "ab"


def test_bpe_validates_training_arguments():
    with pytest.raises(ValueError):
        bpe.BPE.train("abc", -1)
    with pytest.raises(ValueError):
        bpe.BPE.train("abc", 1.5)
    with pytest.raises(ValueError):
        bpe.BPE.train("   ", 3)


def test_special_tokens_use_longest_match():
    tok = special_tokens.ToyTokenizer(["<|a|>", "<|ab|>"])
    assert tok.encode("<|ab|>") == [tok.stoi["<|ab|>"]]


def test_generation_prompt_adds_assistant_header():
    messages = [{"role": "user", "content": "hi"}]
    with_header = chat_template.render_chatml(messages, add_generation_prompt=True)
    without = chat_template.render_chatml(messages)
    assert with_header.endswith("<|im_start|>assistant\n")
    assert not without.endswith("<|im_start|>assistant\n")


def test_token_count_of_empty_text_is_zero():
    assert chat_template.token_count("   ") == 0
    assert chat_template.token_count("abcd") == 1


def test_llama2_template_uses_system_once():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    text = chat_template.render_llama2(messages)
    assert text.startswith("<s>")
    assert text.count("[INST]") == 2
    assert "[INST] SYS u1 [/INST]" in text
    assert "[INST] u2 [/INST]" in text


def test_budget_is_strict_even_for_tiny_values():
    messages = [
        {"role": "system", "content": "s" * 200},
        {"role": "user", "content": "u" * 200},
    ]
    for budget in [1, 2, 3, 4, 5, 10, 25]:
        result = context_window.fit_to_budget(messages, budget, "keep_system")
        assert result["fits"] is True
        assert result["tokens"] <= budget


def test_reserve_tokens_reduces_available_space():
    messages = [{"role": "user", "content": "x " * 50}]
    result = context_window.fit_to_budget(messages, 100, "keep_system", reserve_tokens=40)
    assert result["tokens"] <= 60
    assert result["reserve_tokens"] == 40


def test_drop_oldest_and_keep_system_differ_on_system_prompt():
    messages = [
        {"role": "system", "content": "system " * 30},
        {"role": "user", "content": "u " * 30},
    ]
    dropped = context_window.fit_to_budget(messages, 50, "drop_oldest")
    kept = context_window.fit_to_budget(messages, 50, "keep_system")
    assert kept["kept"][0]["role"] == "system"
    assert all(m["role"] != "system" for m in dropped["kept"])


def test_truncate_content_preserves_short_messages():
    messages = [
        {"role": "user", "content": "needle 4271"},
        {"role": "assistant", "content": "filler " * 100},
    ]
    result = context_window.fit_to_budget(messages, 30, "truncate_content")
    assert len(result["kept"]) == 2
    assert "4271" in " ".join(m["content"] for m in result["kept"])


def test_invalid_context_arguments():
    with pytest.raises(ValueError):
        context_window.fit_to_budget([], 0, "keep_system")
    with pytest.raises(ValueError):
        context_window.fit_to_budget([], 10, "keep_system", reserve_tokens=10)
    with pytest.raises(ValueError):
        context_window.attention_cost(-1)


def test_chars_per_token_helper():
    assert tokenizer_comparison.chars_per_token(10, 4) == 2.5
    assert tokenizer_comparison.chars_per_token(0, 0) == 0.0


def test_cost_rejects_invalid_input():
    with pytest.raises(ValueError):
        token_cost.cost_usd(-1, 10, "qwen-plus")
    with pytest.raises(ValueError):
        token_cost.cost_usd(1, 1, "model-tidak-ada")
    with pytest.raises(TypeError):
        token_cost.cost_usd(1.5, 1, "qwen-plus")

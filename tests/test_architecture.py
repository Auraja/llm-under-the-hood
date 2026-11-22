"""Test Day 03: loss, konsep training, LoRA, quantization, MoE, RAG."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "day03_model_architecture"))

import adaptation_methods  # noqa: E402
import lora_estimator  # noqa: E402
import loss_perplexity  # noqa: E402
import model_families  # noqa: E402
import quantization_estimator  # noqa: E402
import rag_vs_finetuning  # noqa: E402
import training_concepts  # noqa: E402


# ---------------------------------------------------------------- loss
def test_cross_entropy_bounds():
    assert loss_perplexity.cross_entropy(1.0) == pytest.approx(0.0)
    assert loss_perplexity.cross_entropy(0.5) == pytest.approx(math.log(2))
    with pytest.raises(ValueError):
        loss_perplexity.cross_entropy(0.0)


def test_perplexity_is_inverse_probability():
    assert loss_perplexity.perplexity(0.0) == pytest.approx(1.0)
    ppl = loss_perplexity.perplexity(math.log(10))
    assert ppl == pytest.approx(10.0)


def test_uniform_baseline_matches_vocab():
    assert loss_perplexity.perplexity(math.log(50_000)) == pytest.approx(50_000)


def test_temperature_increases_entropy_and_loss():
    logits = np.array([3.0, 1.0, 0.0, -1.0])
    cold = loss_perplexity.softmax(logits, 0.5)
    hot = loss_perplexity.softmax(logits, 2.0)
    assert (hot > cold).sum() >= 2
    assert loss_perplexity.cross_entropy(float(hot[0])) > loss_perplexity.cross_entropy(float(cold[0]))


def test_bits_per_token_relation():
    assert loss_perplexity.bits_per_token(math.log(2)) == pytest.approx(1.0)


# ------------------------------------------------------- training concepts
def test_shift_targets_pairs_consecutive_tokens():
    pairs = training_concepts.shift_targets(["a", "b", "c"])
    assert pairs == [("a", "b"), ("b", "c")]


def test_sgd_step_reduces_loss():
    target = 3.0
    w = 0.0
    grad = training_concepts.gradient(w, target)
    new_w = training_concepts.sgd_step(w, grad, 0.1)
    assert new_w > w
    assert training_concepts.mse_loss(new_w, target) < training_concepts.mse_loss(w, target)


def test_training_memory_components():
    mem = training_concepts.training_memory_bytes(1_000, bytes_per_weight=2, use_adam=True)
    assert mem["total"] == mem["weights"] + mem["gradients"] + mem["optimizer"] + mem["master_weights"]
    no_adam = training_concepts.training_memory_bytes(1_000, use_adam=False)
    assert no_adam["total"] < mem["total"]
    assert no_adam["optimizer"] == 0


def test_cosine_schedule_warmup_and_decay():
    peak = 1e-3
    assert training_concepts.cosine_lr(0, 1000, 100, peak) == pytest.approx(peak / 100)
    assert training_concepts.cosine_lr(99, 1000, 100, peak) == pytest.approx(peak)
    assert training_concepts.cosine_lr(999, 1000, 100, peak) < peak * 0.01


def test_dpo_prefers_chosen_when_margin_positive():
    good = training_concepts.dpo_loss(-8.0, -25.0, -12.0, -14.0)
    bad = training_concepts.dpo_loss(-25.0, -8.0, -12.0, -14.0)
    assert good["loss"] < bad["loss"]
    assert good["preference_prob"] > 0.5 > bad["preference_prob"]
    assert 0 < good["preference_prob"] < 1


# ------------------------------------------------------------------ LoRA
def test_lora_parameter_formula():
    d_in, d_out, r = 4096, 4096, 8
    assert lora_estimator.lora_params_module(d_in, d_out, r) == r * (d_in + d_out)
    total = lora_estimator.lora_params_total(8, ("q", "v"), 2)
    assert total == 2 * 2 * lora_estimator.lora_params_module(4096, 4096, 8)


def test_lora_params_scale_with_rank():
    low = lora_estimator.lora_params_total(8, ("q", "v"), 32)
    high = lora_estimator.lora_params_total(16, ("q", "v"), 32)
    assert high == pytest.approx(2 * low)


def test_zero_b_adapter_is_identity():
    rng = np.random.default_rng(0)
    d_in, d_out, r = 16, 12, 4
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = np.zeros((d_out, r))
    x = rng.normal(size=(3, d_in))
    base = x @ W.T
    with_adapter = base + (8.0 / r) * (x @ A.T @ B.T)
    assert np.allclose(base, with_adapter)


def test_merge_matches_unmerged_output():
    rng = np.random.default_rng(2)
    d_in, d_out, r, alpha = 24, 16, 4, 8.0
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(scale=0.1, size=(r, d_in))
    B = rng.normal(scale=0.1, size=(d_out, r))
    x = rng.normal(size=(5, d_in))

    unmerged = x @ W.T + (alpha / r) * (x @ A.T @ B.T)
    merged = lora_estimator.merge(W, A, B, alpha, r)
    assert np.allclose(unmerged, x @ merged.T)


def test_lora_memory_is_smaller_than_full_finetuning():
    n = lora_estimator.lora_params_total(16, tuple(lora_estimator.MODULES), 32)
    adapter = lora_estimator.lora_memory(n)["total"]
    full = 7_000_000_000 * (2 + 2 + 8 + 4)
    assert adapter < full / 100


# --------------------------------------------------------- quantization
def test_more_bits_means_less_error():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (128, 128))
    err8 = quantization_estimator.quant_error(
        W, quantization_estimator.quantize_symmetric(W, 8))["relative_mse"]
    err4 = quantization_estimator.quant_error(
        W, quantization_estimator.quantize_symmetric(W, 4))["relative_mse"]
    assert err8 < err4


def test_grouping_helps_with_outliers():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (64, 512))
    W[0, 0] = 2.0  # outlier
    per_tensor = quantization_estimator.quantize_symmetric(W, 4)
    grouped = quantization_estimator.quantize_symmetric(W, 4, group_size=64)
    e_tensor = quantization_estimator.quant_error(W, per_tensor)["relative_mse"]
    e_group = quantization_estimator.quant_error(W, grouped)["relative_mse"]
    assert e_group < e_tensor


def test_quantized_values_stay_within_range():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(16, 16))
    out = quantization_estimator.quantize_symmetric(W, 4)
    qmax = 2 ** 3 - 1
    scale = np.abs(W).max() / qmax
    assert np.abs(out).max() <= np.abs(W).max() + 1e-9
    assert out.shape == W.shape


def test_weight_memory_ordering_and_group_overhead():
    n = 1_000_000
    assert quantization_estimator.weight_memory_gb(n, 4) < quantization_estimator.weight_memory_gb(n, 8)
    assert quantization_estimator.weight_memory_gb(n, 8) < quantization_estimator.weight_memory_gb(n, 16)
    grouped = quantization_estimator.weight_memory_gb(n, 4, group_size=128)
    plain = quantization_estimator.weight_memory_gb(n, 4)
    assert grouped > plain


# ---------------------------------------------------------------- MoE
def test_dense_active_equals_total():
    cfg = model_families.DENSE["LLaMA-3 8B"]
    summary = model_families.summarize_dense(cfg)
    assert summary["active"] == summary["total"]
    assert summary["total"] / 1e9 == pytest.approx(8.0, abs=0.5)


def test_moe_active_is_smaller_than_total():
    cfg = model_families.MOE["Mixtral 8x7B"]
    summary = model_families.summarize_moe(cfg)
    assert summary["active"] < summary["total"]
    assert summary["total"] / 1e9 == pytest.approx(46.7, abs=1.5)
    assert summary["active"] / 1e9 == pytest.approx(12.9, abs=1.0)


def test_moe_active_scales_with_top_k():
    cfg = model_families.MOE["Mixtral 8x7B"]
    one = model_families.moe_layer_params({**cfg, "top_k": 1})["active"]
    two = model_families.moe_layer_params({**cfg, "top_k": 2})["active"]
    assert two > one


def test_attention_params_mha_formula():
    assert model_families.attention_params(64, 8, 8) == 4 * 64 * 64
    assert model_families.attention_params(64, 8, 2) < 4 * 64 * 64


# ------------------------------------------------------- adaptation methods
def test_lora_much_smaller_than_full_finetuning():
    lora = adaptation_methods.lora_params(4096, 11008, 32, 8)
    assert lora < adaptation_methods.BASE_PARAMS * 0.01


def test_memory_estimate_ordering():
    full = adaptation_methods.memory_estimate(adaptation_methods.BASE_PARAMS, train_base=True)["total"]
    lora = adaptation_methods.memory_estimate(adaptation_methods.lora_params(4096, 11008, 32, 8))["total"]
    assert lora < full


def test_prefix_and_prompt_params_are_small():
    prefix = adaptation_methods.prefix_params(32, 32, 128, 20)
    prompt = adaptation_methods.prompt_params(4096, 100)
    assert prefix < 10_000_000
    assert prompt == 4096 * 100


# ------------------------------------------------------------------ RAG
def test_llm_cost_is_linear():
    a = rag_vs_finetuning.llm_cost(1000, 100)
    b = rag_vs_finetuning.llm_cost(2000, 200)
    assert b == pytest.approx(2 * a)


def test_rag_costs_more_per_request_than_baseline():
    scenario = rag_vs_finetuning.SCENARIOS["FAQ produk statis"]
    base = rag_vs_finetuning.baseline_cost(scenario["question_tokens"], scenario["answer_tokens"])
    rag = rag_vs_finetuning.rag_cost(scenario["doc_tokens"], scenario["question_tokens"],
                                     scenario["answer_tokens"])
    assert rag > base


def test_break_even_requests_math():
    assert rag_vs_finetuning.break_even_requests(100.0, 0.01) == pytest.approx(10_000)
    assert rag_vs_finetuning.break_even_requests(100.0, 0) == float("inf")


def test_finetune_cost_positive_and_includes_human():
    cost = rag_vs_finetuning.finetune_cost(1_000_000)
    assert cost["total"] == pytest.approx(cost["compute"] + cost["human"] + cost["eval"])
    assert cost["human"] > cost["compute"]


def test_latency_increases_with_prompt_tokens():
    small = rag_vs_finetuning.latency_ms(100, 50)
    large = rag_vs_finetuning.latency_ms(10_000, 50, retrieval_ms=100)
    assert large["total"] > small["total"]
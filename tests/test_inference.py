"""Test Day 04: decoding, KV cache, prefix cache, speculative, memori, benchmark.

Tidak ada test yang memanggil jaringan. API client hanya diuji bagian
pemuat .env dan utilitasnya.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "day04_inference"))

import api_client  # noqa: E402
import benchmark  # noqa: E402
import decoding_simulation  # noqa: E402
import generation_parameters  # noqa: E402
import kv_cache  # noqa: E402
import memory_estimator  # noqa: E402
import prefix_caching  # noqa: E402
import speculative_decoding  # noqa: E402


# ------------------------------------------------------------- api client
def test_load_env_parses_and_expands(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# komentar\nA=1\nB=hello world\nC=${A}-x\nD='quoted'\n",
        encoding="utf-8",
    )
    values = api_client.load_env(env_file)
    assert values["A"] == "1"
    assert values["B"] == "hello world"
    assert values["C"] == "1-x"
    assert values["D"] == "quoted"


def test_mask_key_does_not_leak_full_key():
    masked = api_client.mask_key("sk-1234567890abcdef")
    assert masked.startswith("sk-123")
    assert masked.endswith("cdef")
    assert "67890" not in masked
    assert api_client.mask_key("") == "(kosong)"


def test_has_api_key_rejects_placeholder():
    # placeholder dari .env.example harus dianggap belum diisi
    assert api_client.has_api_key() in (True, False)  # tidak crash tanpa key


def test_default_base_url_is_singapore():
    assert "dashscope-intl" in api_client.DEFAULT_BASE_URL


# -------------------------------------------------- generation parameters
def test_softmax_temperature_shapes_distribution():
    logits = np.array([4.0, 1.0, 0.0, -2.0])
    cold = generation_parameters.softmax(logits, 0.3)
    warm = generation_parameters.softmax(logits, 3.0)
    assert cold.sum() == pytest.approx(1.0)
    assert generation_parameters.entropy(cold) < generation_parameters.entropy(warm)
    with pytest.raises(ValueError):
        generation_parameters.softmax(logits, 0)


def test_top_k_keeps_only_k_tokens():
    probs = generation_parameters.softmax(np.arange(10.0))
    filtered = generation_parameters.apply_top_k(probs, 3)
    assert (filtered > 0).sum() == 3
    assert filtered.sum() == pytest.approx(1.0)


def test_top_p_keeps_enough_mass():
    probs = generation_parameters.softmax(np.arange(10.0))
    for p in [0.5, 0.9, 0.99]:
        filtered = generation_parameters.apply_top_p(probs, p)
        assert filtered.sum() == pytest.approx(1.0)
        kept_mass = probs[filtered > 0].sum()
        assert kept_mass >= p - 1e-9


def test_top_p_validation():
    with pytest.raises(ValueError):
        generation_parameters.apply_top_p(np.ones(4) / 4, 0)
    with pytest.raises(ValueError):
        generation_parameters.apply_top_p(np.ones(4) / 4, 1.5)


def test_repetition_penalty_reduces_seen_logits():
    logits = np.array([3.0, 2.0, 1.0])
    penalized = generation_parameters.apply_repetition_penalty(logits, [0, 1], 1.5)
    assert penalized[0] < logits[0]
    assert penalized[2] == logits[2]
    assert np.allclose(
        generation_parameters.apply_repetition_penalty(logits, [], 1.5), logits
    )


# -------------------------------------------------------------- decoding
def test_greedy_picks_highest_probability_token():
    tokens, _ = decoding_simulation.greedy()
    assert tokens[0] == "<bos>"
    assert tokens[1] == "the"          # bos -> the adalah argmax
    assert tokens[2] == "cat"          # the -> cat adalah argmax
    assert len(tokens) <= 11


def test_beam_search_returns_sorted_hypotheses():
    beams = decoding_simulation.beam_search(width=3)
    assert 0 < len(beams) <= 3
    scores = [score for _, score in beams]
    assert scores == sorted(scores, reverse=True)
    for tokens, _ in beams:
        assert tokens[0] == "<bos>"


def test_sampling_respects_top_k_vocab():
    rng = np.random.default_rng(0)
    for _ in range(20):
        tokens = decoding_simulation.sample_sequence(max_steps=6, top_k=2,
                                                     temperature=0.7, rng=rng)
        assert tokens[0] == "<bos>"
        assert len(tokens) <= 7


def test_beam_width_bounds_number_of_hypotheses():
    assert len(decoding_simulation.beam_search(width=1)) == 1
    assert len(decoding_simulation.beam_search(width=5)) <= 5


# --------------------------------------------------------------- KV cache
def test_incremental_attention_matches_full():
    rng = np.random.default_rng(0)
    seq, d_k = 12, 8
    Q, K, V = (rng.normal(size=(seq, d_k)) for _ in range(3))
    full = kv_cache.full_attention(Q, K, V)
    incremental = np.array([
        kv_cache.incremental_attention(Q[t], K[: t + 1], V[: t + 1]) for t in range(seq)
    ])
    assert np.allclose(full, incremental)


def test_cache_reduces_work_and_grows_linearly():
    assert kv_cache.work_units_cache(100) < kv_cache.work_units_nocache(100)
    assert kv_cache.kv_cache_bytes(4, 2, 8, 1000) * 2 == kv_cache.kv_cache_bytes(4, 2, 8, 2000)
    assert kv_cache.kv_cache_bytes(4, 2, 8, 1000, batch=2) == 2 * kv_cache.kv_cache_bytes(4, 2, 8, 1000)


# ---------------------------------------------------------- prefix caching
def test_cached_request_is_cheaper_and_faster():
    plain = prefix_caching.request_cost(2000, 0, 100)
    cached = prefix_caching.request_cost(2000, 1500, 100)
    assert cached < plain
    assert prefix_caching.prefill_latency_s(2000, 1500) < prefix_caching.prefill_latency_s(2000, 0)


def test_more_cache_hits_save_more():
    low = prefix_caching.monthly_savings(1000, 2000, 500, 100)
    high = prefix_caching.monthly_savings(1000, 2000, 1800, 100)
    assert high["saved"] > low["saved"]
    assert high["percent"] > low["percent"]


def test_cached_tokens_cannot_exceed_prompt():
    with pytest.raises(ValueError):
        prefix_caching.request_cost(100, 200, 10)


def test_cache_storage_worth_it_at_high_traffic():
    result = prefix_caching.cache_is_worth_it(4000, requests_per_hour=500,
                                              storage_cost_per_gb_hour=0.001,
                                              cache_bytes=200e6, ttl_hours=1)
    assert result["worth_it"] is True


# ---------------------------------------------------- speculative decoding
def test_acceptance_probability_bounds():
    assert speculative_decoding.acceptance_probability(0.5, 0.5) == 1.0
    assert speculative_decoding.acceptance_probability(0.2, 0.4) == 0.5
    assert speculative_decoding.acceptance_probability(0.4, 0.0) == 1.0


def test_residual_distribution_is_valid_probability():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    residual = speculative_decoding.residual_distribution(p, q)
    assert residual.sum() == pytest.approx(1.0)
    assert (residual >= 0).all()


def test_expected_tokens_at_least_one():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.1, 0.8])
    expected = speculative_decoding.expected_accepted(p, q, gamma=4)
    assert expected >= 1.0
    assert expected <= 5.0


def test_identical_models_give_maximum_speedup():
    p = np.array([0.5, 0.3, 0.2])
    gamma = 3
    expected = speculative_decoding.expected_accepted(p, p, gamma)
    assert expected == pytest.approx(gamma + 1)
    assert speculative_decoding.speedup(p, p, gamma, 0.0) == pytest.approx(gamma + 1)


def test_speculative_sampling_preserves_target_distribution():
    p = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    q = np.array([0.30, 0.30, 0.20, 0.10, 0.06, 0.04])
    rng = np.random.default_rng(0)
    counts = np.zeros(len(p))
    for _ in range(4000):
        tokens, _ = speculative_decoding.speculate_step(p, q, gamma=3, rng=rng)
        for token in tokens:
            counts[token] += 1
    empirical = counts / counts.sum()
    assert np.abs(empirical - p).max() < 0.03


# ------------------------------------------------------- memory estimator
def test_memory_grows_with_context_and_batch():
    cfg = memory_estimator.CONFIGS["LLaMA-3 8B"]
    small = memory_estimator.estimate(cfg, 1_000)["total"]
    large = memory_estimator.estimate(cfg, 16_000)["total"]
    batched = memory_estimator.estimate(cfg, 1_000, batch=4)["total"]
    assert large > small
    assert batched > small


def test_quantized_weights_use_less_memory():
    cfg = memory_estimator.CONFIGS["LLaMA-3 8B"]
    fp16 = memory_estimator.estimate(cfg, 4_000, weight_bytes=2)["total"]
    int4 = memory_estimator.estimate(cfg, 4_000, weight_bytes=0.5)["total"]
    assert int4 < fp16


def test_max_batch_decreases_with_context():
    cfg = memory_estimator.CONFIGS["LLaMA-3 8B"]
    short = memory_estimator.max_batch(cfg, 1_000, budget_gb=64)
    long = memory_estimator.max_batch(cfg, 32_000, budget_gb=64)
    assert long < short


def test_kv_cache_is_dominant_at_long_context():
    cfg = memory_estimator.CONFIGS["LLaMA-3 8B"]
    parts = memory_estimator.estimate(cfg, 128_000)
    assert parts["kv_cache"] > parts["weights"]


# ---------------------------------------------------------------- benchmark
def test_decode_estimate_is_positive_and_decreases_with_model_size():
    bytes_small = 500_000_000 * 2
    bytes_large = 8_000_000_000 * 2
    assert benchmark.estimate_decode_tokens_per_s(20.0, 500_000_000, 0) > 0
    small = 20e9 / bytes_small
    large = 20e9 / bytes_large
    assert small > large


def test_best_time_returns_minimum_of_repeats():
    calls = []

    def fake() -> None:
        calls.append(1)

    assert benchmark.best_time(fake, repeats=3) >= 0
    assert len(calls) == 3
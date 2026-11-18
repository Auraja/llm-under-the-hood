"""Test Day 02: attention, posisi, normalization, FFN, decoder block, parameter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "day02_transformer"))

import attention  # noqa: E402
import attention_variants  # noqa: E402
import decoder_block  # noqa: E402
import embeddings  # noqa: E402
import feed_forward  # noqa: E402
import normalization  # noqa: E402
import parameter_estimator  # noqa: E402
import positional_encoding  # noqa: E402
import rope  # noqa: E402


# ------------------------------------------------------------- softmax
def test_softmax_sums_to_one_and_is_stable():
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -999.0]])
    probs = attention.softmax(x)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()


def test_causal_mask_is_lower_triangular():
    mask = attention.causal_mask(4)
    assert mask.dtype == bool
    assert mask[0, 1:].sum() == 0
    assert mask[3].all()


def test_attention_weights_respect_causality():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(5, 8)) for _ in range(3))
    _, weights = attention.scaled_dot_product_attention(q, k, v, attention.causal_mask(5))
    assert np.allclose(weights.sum(axis=-1), 1.0)
    for i in range(5):
        assert weights[i, i + 1:].sum() == 0


def test_future_tokens_do_not_change_past_outputs():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4, 8))
    k = rng.normal(size=(4, 8))
    v = rng.normal(size=(4, 8))
    mask = attention.causal_mask(4)
    out1, _ = attention.scaled_dot_product_attention(q, k, v, mask)

    k2, v2 = k.copy(), v.copy()
    k2[3] += 5.0
    v2[3] -= 5.0
    out2, _ = attention.scaled_dot_product_attention(q, k2, v2, mask)
    assert np.allclose(out1[:3], out2[:3], atol=1e-12)
    assert not np.allclose(out1[3], out2[3])


def test_multi_head_attention_shape_and_params():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 16))
    w = rng.normal(scale=0.1, size=(16, 16))
    out = attention.multi_head_attention(x, w, w, w, w, n_heads=4)
    assert out.shape == x.shape
    assert attention.attention_params(16) == 4 * 16 * 16
    assert attention.attention_params(16, bias=True) == 4 * (16 * 16 + 16)


def test_single_head_equals_plain_attention():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 8))
    w = rng.normal(scale=0.1, size=(8, 8))
    out = attention.multi_head_attention(x, w, w, w, w, n_heads=1)
    q = x @ w
    expected, _ = attention.scaled_dot_product_attention(q, q, q, attention.causal_mask(4))
    assert np.allclose(out, expected @ w)


# -------------------------------------------------------------- RoPE
def test_rope_preserves_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 8))
    rotated = rope.apply_rope(x, np.array([7]))
    assert np.allclose(np.linalg.norm(x), np.linalg.norm(rotated))


def test_rope_dot_product_depends_on_relative_distance():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(1, 8))
    k = rng.normal(size=(1, 8))

    def value(m: int, n: int) -> float:
        qm = rope.apply_rope(q, np.array([m]))[0]
        kn = rope.apply_rope(k, np.array([n]))[0]
        return float(qm @ kn)

    assert value(4, 1) == pytest.approx(value(40, 37), abs=1e-9)
    assert value(10, 10) != pytest.approx(value(10, 5), abs=1e-6)


def test_rope_batch_heads_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 8))
    out = rope.apply_rope(x, np.arange(5))
    assert out.shape == x.shape


# ------------------------------------------------ positional encoding
def test_sinusoidal_encoding_properties():
    pe = positional_encoding.sinusoidal_encoding(64, 16)
    assert pe.shape == (64, 16)
    assert pe.min() >= -1 and pe.max() <= 1
    assert np.allclose(pe[0, 0::2], 0.0)   # sin(0)
    assert np.allclose(pe[0, 1::2], 1.0)   # cos(0)


def test_similarity_decays_with_distance():
    pe = positional_encoding.sinusoidal_encoding(256, 32)
    near = positional_encoding.relative_similarity(pe, 1)
    far = positional_encoding.relative_similarity(pe, 64)
    assert near > far


# -------------------------------------------------------- normalization
def test_layer_norm_normalizes_each_row():
    x = np.random.default_rng(0).normal(loc=3, scale=2, size=(4, 8))
    y = normalization.layer_norm(x, np.ones(8), np.zeros(8))
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-6)
    assert np.allclose(y.var(axis=-1), 1, atol=1e-4)


def test_rms_norm_sets_unit_rms():
    x = np.random.default_rng(0).normal(loc=3, scale=2, size=(4, 8))
    z = normalization.rms_norm(x, np.ones(8))
    assert np.allclose(np.sqrt((z**2).mean(axis=-1)), 1.0, atol=1e-5)


def test_rmsnorm_has_half_the_parameters():
    assert normalization.norm_params(64, "rmsnorm") == 64
    assert normalization.norm_params(64, "layernorm") == 128


# ------------------------------------------------------------ feed forward
def test_ffn_shapes_and_params():
    x = np.random.default_rng(0).normal(size=(3, 16))
    p = feed_forward.make_ffn(16, 64)
    out = feed_forward.ffn_forward(x, p)
    assert out.shape == (3, 16)
    assert feed_forward.ffn_params(16, 64) == 2 * 16 * 64 + 64 + 16


def test_swiglu_formula_and_params():
    assert feed_forward.swiglu_params(16, 43) == 3 * 16 * 43
    assert feed_forward.silu(np.array([0.0]))[0] == pytest.approx(0.0)
    assert feed_forward.gelu(np.array([0.0]))[0] == pytest.approx(0.0)


def test_ffn_does_not_mix_positions():
    x = np.random.default_rng(0).normal(size=(4, 16))
    p = feed_forward.make_ffn(16, 64)
    y1 = feed_forward.ffn_forward(x, p)
    x2 = x.copy()
    x2[0] += 10
    y2 = feed_forward.ffn_forward(x2, p)
    assert np.allclose(y1[1:], y2[1:])


# ---------------------------------------------------------- decoder block
def test_decoder_block_shape_and_params():
    block = decoder_block.DecoderBlock(32, 4, 88)
    x = np.random.default_rng(0).normal(size=(5, 32))
    out = block.forward(x)
    assert out.shape == x.shape
    counts = block.num_params()
    assert counts["total"] == counts["attention"] + counts["ffn"] + counts["norm"]
    assert counts["ffn"] > counts["attention"]


def test_stack_preserves_causality():
    blocks = decoder_block.build_stack(3, 16, 2, 48)
    x = np.random.default_rng(0).normal(size=(6, 16))
    out1 = x
    for block in blocks:
        out1 = block.forward(out1)

    x2 = x.copy()
    x2[-1] += 5.0
    out2 = x2
    for block in blocks:
        out2 = block.forward(out2)

    assert np.allclose(out1[:-1], out2[:-1], atol=1e-10)
    assert not np.allclose(out1[-1], out2[-1])


# ------------------------------------------------------ parameter estimator
def test_parameter_estimator_matches_published_sizes():
    for name, cfg in parameter_estimator.CONFIGS.items():
        total = parameter_estimator.estimate(cfg)["total"] / 1e9
        assert abs(total - cfg["published"]) / cfg["published"] < 0.02, name


def test_gqa_reduces_attention_parameters():
    cfg = parameter_estimator.CONFIGS["LLaMA-2 70B"]
    mha = parameter_estimator.estimate({**cfg, "n_kv_heads": cfg["n_heads"]})
    gqa = parameter_estimator.estimate({**cfg, "n_kv_heads": 8})
    assert gqa["attention"] < mha["attention"]
    assert gqa["total"] < mha["total"]


def test_embeddings_lookup_equals_one_hot():
    table = embeddings.make_embedding(50, 8, seed=0)
    ids = [1, 7, 42]
    assert np.allclose(embeddings.embed_lookup(ids, table),
                       embeddings.embed_onehot(ids, table))


def test_weight_tying_saves_parameters():
    tied = embeddings.parameter_count(1000, 64, tied=True)["total"]
    untied = embeddings.parameter_count(1000, 64, tied=False)["total"]
    assert tied * 2 == untied


# ------------------------------------------------------- attention variants
def test_gqa_kv_expansion_matches_query_heads():
    rng = np.random.default_rng(0)
    k = rng.normal(size=(2, 3, 4))
    expanded = attention_variants.expand_kv(k, groups=4)
    assert expanded.shape == (8, 3, 4)
    assert np.allclose(expanded[0], expanded[1])


def test_kv_cache_sizes_ordering():
    cfg = {"d_model": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8}
    gqa = attention_variants.kv_cache_bytes(cfg, 1000)
    mha = attention_variants.kv_cache_bytes({**cfg, "n_kv_heads": 32}, 1000)
    mqa = attention_variants.kv_cache_bytes({**cfg, "n_kv_heads": 1}, 1000)
    assert mqa < gqa < mha
    assert mha / mqa == pytest.approx(32)
    assert attention_variants.kv_cache_bytes(cfg, 2000) == pytest.approx(2 * gqa)


def test_projection_params_shrink_with_fewer_kv_heads():
    full = attention_variants.projection_params(4096, 32, 32)
    grouped = attention_variants.projection_params(4096, 32, 8)
    assert grouped < full
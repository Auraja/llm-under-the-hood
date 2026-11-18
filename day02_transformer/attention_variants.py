"""Varian attention: MHA, MQA, GQA - dan berapa KV cache yang dihemat.

- MHA (Multi-Head Attention): setiap query head punya key/value head sendiri.
  Kualitas terbaik, KV cache terbesar.
- MQA (Multi-Query Attention): semua query head berbagi SATU key/value head.
  KV cache sangat kecil, kualitas sedikit turun, sering dipakai untuk inference
  cepat.
- GQA (Grouped-Query Attention): query head dibagi ke beberapa grup; setiap grup
  berbagi satu KV head. Ini kompromi yang dipakai LLaMA-2 70B, LLaMA-3, Qwen2.5,
  Mistral.

KV cache per token = 2 (K dan V) * n_layers * n_kv_heads * head_dim * bytes.
Jadi n_kv_heads adalah tombol utama untuk menghemat memori konteks panjang.

Jalankan: python day02_transformer/attention_variants.py
"""

from __future__ import annotations

import numpy as np


def projection_params(d_model: int, n_heads: int, n_kv_heads: int) -> int:
    """Parameter proyeksi Q, K, V, O tanpa bias (head_dim = d_model / n_heads)."""
    head_dim = d_model // n_heads
    q = d_model * d_model
    k = d_model * (n_kv_heads * head_dim)
    v = d_model * (n_kv_heads * head_dim)
    o = d_model * d_model
    return q + k + v + o


def kv_cache_bytes(config: dict, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    head_dim = config["d_model"] // config["n_heads"]
    return 2 * config["n_layers"] * config["n_kv_heads"] * head_dim * seq_len * batch * dtype_bytes


def expand_kv(k: np.ndarray, groups: int) -> np.ndarray:
    """Ulangi setiap KV head sebanyak ``groups`` agar cocok dengan query head.

    (n_kv, seq, head_dim) -> (n_kv * groups, seq, head_dim)
    """
    return np.repeat(k, groups, axis=0)


def attention_with_kv_layout(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, groups: int, causal: bool = True
) -> np.ndarray:
    """Attention dengan jumlah query head = n_kv_heads * groups."""
    k = expand_kv(k, groups)
    v = expand_kv(v, groups)
    head_dim = q.shape[-1]
    seq = q.shape[1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(head_dim)
    if causal:
        scores = np.where(np.tril(np.ones((seq, seq), dtype=bool)), scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


CONFIGS = {
    "MHA  (LLaMA-2 7B)": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32),
    "GQA  (LLaMA-2 70B)": dict(d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8),
    "GQA  (Mistral 7B)": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8),
    "MQA  (PaLM-style)": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=1),
}


def main() -> None:
    print("=== 1. Parameter proyeksi per varian ===")
    print(f"{'varian':<22} {'q_heads':>8} {'kv_heads':>9} {'params':>12} {'vs MHA':>8}")
    base = projection_params(4096, 32, 32)
    for name, cfg in CONFIGS.items():
        if cfg["d_model"] != 4096:
            continue
        p = projection_params(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"])
        print(f"{name:<22} {cfg['n_heads']:>8} {cfg['n_kv_heads']:>9} {p:>12,} {p / base:>7.2f}x")

    print("\n=== 2. KV cache per 1.000 token (fp16) ===")
    print(f"{'varian':<22} {'seq=1k':>10} {'seq=32k':>12} {'seq=128k':>13}")
    for name, cfg in CONFIGS.items():
        row = [f"{kv_cache_bytes(cfg, n) / 1024 / 1024:>9.1f}M" for n in (1_000, 32_000, 128_000)]
        print(f"{name:<22} {row[0]:>10} {row[1]:>12} {row[2]:>13}")
    print("\ncatatan: angka di atas hanya untuk 1 sequence, 1 batch.")
    print("batch 8 x 32k token pada MHA langsung menjadi ~64 GB hanya untuk cache.")

    print("\n=== 3. Hubungan jumlah head ===")
    cfg = CONFIGS["GQA  (Mistral 7B)"]
    print(f"d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, n_kv_heads={cfg['n_kv_heads']}")
    print(f"groups = n_heads / n_kv_heads = {cfg['n_heads'] // cfg['n_kv_heads']} query head per KV head")
    print(f"KV cache MHA  : {kv_cache_bytes({**cfg, 'n_kv_heads': 32}, 8192) / 1024 / 1024:>7.1f} MB")
    print(f"KV cache GQA-8: {kv_cache_bytes(cfg, 8192) / 1024 / 1024:>7.1f} MB")
    print(f"KV cache MQA  : {kv_cache_bytes({**cfg, 'n_kv_heads': 1}, 8192) / 1024 / 1024:>7.1f} MB")

    print("\n=== 4. Semua varian dihitung dengan rumus attention yang sama ===")
    rng = np.random.default_rng(0)
    seq, head_dim, n_heads = 5, 4, 8
    q = rng.normal(size=(n_heads, seq, head_dim))
    for name, n_kv, groups in [("MHA", 8, 1), ("GQA-4", 4, 2), ("MQA", 1, 8)]:
        k = rng.normal(size=(n_kv, seq, head_dim))
        v = rng.normal(size=(n_kv, seq, head_dim))
        out = attention_with_kv_layout(q, k, v, groups)
        n_params = 2 * seq * n_kv * head_dim  # cache untuk K dan V
        print(f"{name:<6} kv_heads={n_kv} -> output {out.shape}, "
              f"cache {n_params} nilai")

    print("\n=== 5. Kapan memilih apa? ===")
    print("MQA : throughput maksimum, kualitas turun paling banyak.")
    print("GQA : default modern; 8 KV head biasanya cukup, kualitas hampir MHA.")
    print("MHA : dipakai kalau memori bukan masalah atau untuk riset presisi.")
    print("Semua varian di atas memakai jumlah parameter QK^T yang sama;")
    print("yang berubah hanyalah ukuran K/V dan cache-nya.")


if __name__ == "__main__":
    main()
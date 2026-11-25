"""KV cache: kenapa decoding autoregressive tidak menghitung ulang semuanya.

Tanpa cache: untuk menghasilkan token ke-t, model memproses ulang seluruh
prefix (t token) -> total kerja ~1+2+...+T = O(T^2).

Dengan cache: key dan value dari token lama disimpan. Untuk token baru, cukup
hitung q/k/v token itu, lalu attention terhadap cache -> total kerja O(T).

File ini membuktikan dua hal:
1. Hasil incremental attention identik dengan full attention (secara numerik).
2. Penghematan kerja tumbuh kuadratik terhadap panjang output.

Catatan: cache TIDAK mengubah hasil. Cache hanya menghindari perhitungan ulang,
karena key/value token lama tidak pernah berubah (causal mask).

Jalankan: python day04_inference/kv_cache.py
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def full_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Attention causal untuk seluruh sequence sekaligus."""
    seq, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.tril(np.ones((seq, seq), dtype=bool))
    scores = np.where(mask, scores, -np.inf)
    return softmax(scores) @ V


def incremental_attention(q_last: np.ndarray, K_cache: np.ndarray,
                          V_cache: np.ndarray) -> np.ndarray:
    """Attention untuk SATU query baru terhadap cache K/V (termasuk dirinya)."""
    scores = K_cache @ q_last / np.sqrt(q_last.shape[-1])
    weights = softmax(scores)  # tanpa mask: semua isi cache sudah masa lalu
    return weights @ V_cache


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int,
                   seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * dtype_bytes


def work_units_nocache(n_tokens: int) -> int:
    """Jumlah pasangan query-key yang dihitung saat generate T token dari nol."""
    return n_tokens * (n_tokens + 1) // 2


def work_units_cache(n_tokens: int, prompt_len: int = 1) -> int:
    """Prefill prompt + satu query baru per token."""
    prefill = prompt_len * (prompt_len + 1) // 2
    return prefill + max(n_tokens - prompt_len, 0)


def main() -> None:
    rng = np.random.default_rng(0)
    seq, d_k = 40, 16
    Q = rng.normal(size=(seq, d_k))
    K = rng.normal(size=(seq, d_k))
    V = rng.normal(size=(seq, d_k))

    print("=== 1. Incremental == full ===")
    full = full_attention(Q, K, V)
    mismatches = []
    # bangun output satu per satu memakai cache
    cache_out = []
    for t in range(seq):
        cache_out.append(incremental_attention(Q[t], K[: t + 1], V[: t + 1]))
    cache_out = np.array(cache_out)
    print(f"bentuk full    : {full.shape}")
    print(f"bentuk cached  : {cache_out.shape}")
    print(f"selisih maks   : {np.abs(full - cache_out).max():.2e}")
    print("cache tidak mengubah hasil, hanya cara menghitungnya.")

    print("\n=== 2. Kerja yang dihemat ===")
    print(f"{'output token':>12} {'tanpa cache':>13} {'dengan cache':>14} {'rasio':>8}")
    for n in [10, 50, 100, 500, 1000, 4000]:
        no_cache = work_units_nocache(n)
        with_cache = work_units_cache(n, prompt_len=10)
        print(f"{n:>12,} {no_cache:>13,} {with_cache:>14,} {no_cache / with_cache:>7.1f}x")
    print("\nuntuk output 4.000 token, cache menghemat ~3 orde besaran kerja attention.")

    print("\n=== 3. Memori cache ===")
    print("konfigurasi: Qwen2.5-7B (28 layer, 4 KV head, head_dim 128, fp16)")
    layers, n_kv, head_dim = 28, 4, 128
    print(f"{'context':>9} {'batch 1':>10} {'batch 8':>10} {'batch 32':>10}")
    for ctx in [1_000, 8_000, 32_000, 128_000]:
        row = [kv_cache_bytes(layers, n_kv, head_dim, ctx, b) / 1e9 for b in (1, 8, 32)]
        print(f"{ctx:>9,} {row[0]:>9.2f}G {row[1]:>9.2f}G {row[2]:>9.2f}G")

    print("\n=== 4. Cache tumbuh linear, attention tumbuh kuadratik ===")
    print("KV cache per token tetap (kecuali model mengubah arsitektur).")
    print("Karena itu konteks panjang menghabiskan memori, bukan compute saja.")
    print("PagedAttention/vLLM memecah cache menjadi blok supaya tidak perlu")
    print("alokasi maksimum di depan dan mengurangi fragmentasi.")

    print("\n=== 5. Efek pada latensi ===")
    print("TTFT (time to first token) = prefill, sebanding panjang prompt.")
    print("TPOT (time per output token) = 1 forward + baca cache, hampir konstan,")
    print("tetapi naik perlahan saat cache makin besar (bandwidth memori).")
    for ctx in [1_000, 32_000]:
        cache_gb = kv_cache_bytes(layers, n_kv, head_dim, ctx) / 1e9
        bandwidth = 50  # GB/s efektif (CPU)
        print(f"  ctx={ctx:>6,}: baca ~{cache_gb:.2f} GB/token -> "
              f"~{cache_gb / bandwidth * 1000:.2f} ms hanya untuk cache")
    print("\nini menjelaskan kenapa CPU inference lambat pada konteks panjang:")
    print("setiap token harus membaca ulang seluruh cache dari RAM.")


if __name__ == "__main__":
    main()
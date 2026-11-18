"""Scaled dot-product attention + causal mask, dari nol dengan NumPy.

Rumus:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Tiga komponen yang wajib dipahami:
1. Q K^T menghasilkan skor relevansi antar semua pasangan token.
2. Dibagi sqrt(d_k) supaya skor tidak membengkak saat dimensi besar (kalau
   tidak, softmax menjadi terlalu tajam dan gradien hampir nol).
3. Causal mask mengisi posisi masa depan dengan -inf sebelum softmax, sehingga
   bobotnya tepat 0. Ini yang membuat model bisa dipakai untuk next-token
   prediction tanpa "membaca jawaban".

Jalankan: python day02_transformer/attention.py
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax yang stabil secara numerik (kurangi nilai maksimum)."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    """True = boleh diperhatikan. Token i hanya boleh melihat j <= i."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """q: (..., seq_q, d_k), k/v: (..., seq_k, d_k) -> (out, weights)."""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if allowed is not None:
        # -inf (bukan angka besar negatif) membuat exp(-inf) = 0 persis
        scores = np.where(allowed, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(seq, d_model) -> (n_heads, seq, head_dim)."""
    seq, d_model = x.shape
    head_dim = d_model // n_heads
    return x.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)


def multi_head_attention(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    n_heads: int,
    causal: bool = True,
) -> np.ndarray:
    """Satu layer multi-head self-attention."""
    seq = x.shape[0]
    q = split_heads(x @ wq, n_heads)
    k = split_heads(x @ wk, n_heads)
    v = split_heads(x @ wv, n_heads)

    allowed = causal_mask(seq)[None, :, :] if causal else None
    out, _ = scaled_dot_product_attention(q, k, v, allowed)

    # gabungkan kembali head: (heads, seq, head_dim) -> (seq, d_model)
    merged = out.transpose(1, 0, 2).reshape(seq, -1)
    return merged @ wo


def attention_params(d_model: int, bias: bool = False) -> int:
    b = d_model if bias else 0
    return 4 * (d_model * d_model + b)


def main() -> None:
    rng = np.random.default_rng(0)
    seq, d_k = 4, 8
    q, k, v = (rng.normal(size=(seq, d_k)) for _ in range(3))

    print("=== 1. Skor, softmax, dan bobot ===")
    scores = q @ k.T / np.sqrt(d_k)
    scores_full = q @ k.T / np.sqrt(d_k)
    w_no_mask = softmax(scores_full)
    print("skor mentah (QK^T / sqrt(d)):")
    print(np.round(scores, 2))
    print(f"\nsetiap baris bobot berjumlah: {np.round(w_no_mask.sum(axis=1), 6)}")
    print("weight[0]:", np.round(w_no_mask[0], 3), "-> semua token boleh dilihat")

    print("\n=== 2. Causal mask ===")
    mask = causal_mask(seq)
    print("mask (True = boleh):")
    print(mask.astype(int))
    masked_scores = np.where(mask, scores, -np.inf)
    print("\nskor setelah mask:")
    print(np.round(masked_scores, 2))
    w_causal = softmax(masked_scores)
    print("\nbobot setelah mask (baris 0 hanya melihat token 0):")
    print(np.round(w_causal, 3))
    print(f"bobot masa depan baris 0: {w_causal[0, 1:].sum():.1f} (harus 0)")
    print(f"jumlah tiap baris       : {np.round(w_causal.sum(axis=1), 6)}")

    print("\n=== 3. Kausalitas diuji dengan mengubah token terakhir ===")
    k2, v2 = k.copy(), v.copy()
    k2[3] = rng.normal(size=d_k) * 5   # ubah key token terakhir drastis
    v2[3] = rng.normal(size=d_k) * 5   # ubah juga value-nya
    out1, _ = scaled_dot_product_attention(q, k, v, mask)
    out2, _ = scaled_dot_product_attention(q, k2, v2, mask)
    print(f"perubahan output token 0-2: {np.abs(out1[:3] - out2[:3]).max():.2e}")
    print(f"perubahan output token 3  : {np.abs(out1[3] - out2[3]).max():.2e}")
    print("output posisi lama tidak terpengaruh token baru -> tidak ada kebocoran masa depan.")

    print("\n=== 4. Kenapa dibagi sqrt(d_k)? ===")
    large = 256
    q_big = rng.normal(size=(1, large))
    k_big = rng.normal(size=(1, large))
    raw = float((q_big @ k_big.T).item())
    print(f"d_k={large}: dot mentah = {raw:+.1f}, setelah skala = {raw / np.sqrt(large):+.1f}")
    print("tanpa skala, softmax atas nilai puluhan membuat distribusi nyaris one-hot;")
    print("gradien menjadi sangat kecil dan training tidak stabil.")

    print("\n=== 5. Multi-head attention ===")
    d_model, n_heads = 32, 4
    x = rng.normal(size=(seq, d_model))
    wq, wk, wv, wo = (rng.normal(scale=0.1, size=(d_model, d_model)) for _ in range(4))
    out = multi_head_attention(x, wq, wk, wv, wo, n_heads)
    print(f"input      : {x.shape}")
    print(f"per head   : {d_model // n_heads} dimensi")
    print(f"output     : {out.shape}")
    print(f"parameter  : {attention_params(d_model):,} (4 proyeksi, tanpa bias)")
    print(f"per head   : {attention_params(d_model) / n_heads:,.0f} parameter dibagi rata")

    print("\n=== 6. Cocokkan dengan PyTorch ===")
    try:
        import torch
        import torch.nn.functional as F

        tq, tk, tv = (torch.tensor(a, dtype=torch.float64) for a in (q, k, v))
        ref = F.scaled_dot_product_attention(tq, tk, tv, is_causal=True).numpy()
        ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(seq))
        print(f"selisih maksimum NumPy vs PyTorch: {np.abs(ref - ours).max():.2e}")
    except ImportError:
        print("PyTorch tidak terpasang, lewati perbandingan.")

    print("\n=== 7. Kompleksitas ===")
    for n in [128, 1024, 8192]:
        print(f"seq={n:>5}: QK^T = {n * n:>12,} elemen skor "
              f"({n * n * 4 / 1024 / 1024:.1f} MB di fp32, per head)")


if __name__ == "__main__":
    main()
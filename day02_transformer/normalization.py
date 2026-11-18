"""Normalization: LayerNorm vs RMSNorm, dan mengapa pre-norm.

LayerNorm (Ba et al., 2016):
    y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

RMSNorm (Zhang & Sennrich, 2019), dipakai LLaMA/Qwen/Mistral:
    y = gamma * x / sqrt(mean(x^2) + eps)

Perbedaan:
- RMSNorm tidak mengurangi mean dan tidak punya beta -> lebih murah.
- Keduanya menormalkan per token (bukan per batch), jadi tidak bergantung pada
  token lain dalam batch. Ini penting untuk inference autoregressive.
- Pre-norm (norm sebelum sublayer) membuat residual stream tetap bersih dan
  training deep transformer jauh lebih stabil. Post-norm (GPT-2 asli) lebih
  rentan pada kedalaman besar.

Jalankan: python day02_transformer/normalization.py
"""

from __future__ import annotations

import numpy as np


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return gamma * x / rms


def norm_params(d_model: int, kind: str) -> int:
    if kind == "layernorm":
        return 2 * d_model  # gamma + beta
    if kind == "rmsnorm":
        return d_model      # gamma saja
    raise ValueError(kind)


def main() -> None:
    rng = np.random.default_rng(0)
    d_model = 8
    x = rng.normal(loc=5.0, scale=2.0, size=(3, d_model))  # sengaja punya mean != 0

    gamma = np.ones(d_model)
    beta = np.zeros(d_model)

    print("=== 1. LayerNorm ===")
    y = layer_norm(x, gamma, beta)
    print("input  baris 0:", np.round(x[0], 3))
    print("output baris 0:", np.round(y[0], 3))
    print(f"mean output    : {np.round(y.mean(axis=-1), 7)}")
    print(f"var output     : {np.round(y.var(axis=-1), 7)}")
    print("setiap baris dipaksa mean=0, var=1, lalu diskalakan gamma dan digeser beta.")

    print("\n=== 2. RMSNorm ===")
    z = rms_norm(x, gamma)
    print("output baris 0:", np.round(z[0], 3))
    print(f"root mean square: {np.round(np.sqrt((z**2).mean(axis=-1)), 6)} (harus ~1)")
    print(f"mean output     : {np.round(z.mean(axis=-1), 4)} (tidak dipaksa 0)")
    print("RMSNorm hanya mengatur magnitudo, arah tetap.")
    print(f"selisih dengan LayerNorm: {np.abs(z - y).max():.3f}")

    print("\n=== 3. Biaya parameter (per layer, d_model=4096) ===")
    d = 4096
    print(f"LayerNorm: {norm_params(d, 'layernorm'):>6,} parameter per norm")
    print(f"RMSNorm  : {norm_params(d, 'rmsnorm'):>6,} parameter per norm")
    print(f"model 32 layer, 2 norm per layer:")
    print(f"  LayerNorm total: {32 * 2 * norm_params(d, 'layernorm'):,}")
    print(f"  RMSNorm total  : {32 * 2 * norm_params(d, 'rmsnorm'):,}")
    print("bedanya kecil dibanding miliaran parameter, tetapi RMSNorm juga lebih")
    print("sedikit operasi (tanpa mean dan tanpa beta) sehingga lebih ramah memori.")

    print("\n=== 4. Pre-norm vs post-norm: pertumbuhan residual ===")
    d, n_layers = 64, 24
    stream = rng.normal(size=(4, d))
    gamma4, beta4 = np.ones(d), np.zeros(d)
    print(f"{'layer':>5} {'pre-norm |x|':>14} {'post-norm |x|':>15}")

    pre = stream.copy()
    post = stream.copy()
    for layer in range(1, n_layers + 1):
        # sublayer mainan: fungsi acak yang menjaga skala
        sublayer = lambda h: np.tanh(h @ rng.normal(0, 0.1, (d, d)))
        pre = pre + sublayer(layer_norm(pre, gamma4, beta4))
        post = layer_norm(post + sublayer(post), gamma4, beta4)
        if layer in (1, 4, 8, 16, 24):
            print(f"{layer:>5} {np.linalg.norm(pre):>14.2f} {np.linalg.norm(post):>15.2f}")

    print("\npre-norm: norma residual tumbuh perlahan dan tetap terkontrol.")
    print("post-norm: setiap layer memaksa skala, tetapi gradien harus melewati")
    print("banyak normalisasi -> lebih sulit dioptimasi pada model dalam.")
    print("karena itu GPT-3 ke atas, LLaMA, dan Qwen memakai pre-norm.")

    print("\n=== 5. Mengapa norm per-token, bukan per-batch? ===")
    print("BatchNorm butuh statistik seluruh batch, sehingga saat inference")
    print("satu token saja hasilnya berbeda dari saat training.")
    print("LayerNorm/RMSNorm hanya melihat satu vektor token -> konsisten,")
    print("dan itulah yang dibutuhkan KV cache + decoding satu per satu.")


if __name__ == "__main__":
    main()
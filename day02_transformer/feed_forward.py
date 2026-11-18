"""Feed-forward network (FFN) di dalam decoder block.

Dua fakta yang sering mengejutkan pemula:
1. Sebagian besar parameter transformer ada di FFN, bukan di attention.
   Untuk LLaMA-7B, FFN ~= 2/3 dari total parameter per layer.
2. FFN bekerja per-token secara independen (tidak ada interaksi antar posisi).
   Semua pencampuran informasi antar token terjadi di attention.

Varian:
- Klasik (GPT-2): Linear(d -> 4d) -> GELU -> Linear(4d -> d)
- SwiGLU (LLaMA/Qwen/Mistral): Linear(d -> h) -> SiLU, Linear(d -> h), lalu
  dikalikan (gating), Linear(h -> d). Biasanya h = 8/3 * d ~ 2.67d supaya
  jumlah parameternya setara dengan FFN 4d.

Jalankan: python day02_transformer/feed_forward.py
"""

from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray, approximate: bool = True) -> np.ndarray:
    """GELU: gerbang halus, dipakai GPT-2/BERT."""
    if approximate:  # tanh approximation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    from math import erf, sqrt

    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: x * sigmoid(x). Dipakai di SwiGLU."""
    return x / (1 + np.exp(-x))


def make_ffn(d_model: int, d_ff: int, seed: int = 0) -> dict:
    """Inisialisasi bobot FFN klasik dua lapis."""
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(d_model)
    return {
        "W1": rng.normal(0, scale, (d_model, d_ff)),
        "b1": np.zeros(d_ff),
        "W2": rng.normal(0, scale, (d_ff, d_model)),
        "b2": np.zeros(d_model),
    }


def ffn_forward(x: np.ndarray, p: dict, activation=gelu) -> np.ndarray:
    return activation(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def make_swiglu(d_model: int, d_ff: int, seed: int = 0) -> dict:
    """W_gate dan W_up memproyeksikan ke d_ff; W_down kembali ke d_model."""
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(d_model)
    return {
        "W_gate": rng.normal(0, scale, (d_model, d_ff)),
        "W_up": rng.normal(0, scale, (d_model, d_ff)),
        "W_down": rng.normal(0, scale, (d_ff, d_model)),
    }


def swiglu_forward(x: np.ndarray, p: dict) -> np.ndarray:
    gate = silu(x @ p["W_gate"])
    up = x @ p["W_up"]
    return (gate * up) @ p["W_down"]


def ffn_params(d_model: int, d_ff: int, bias: bool = True) -> int:
    return d_model * d_ff + d_ff * d_model + (d_ff + d_model if bias else 0)


def swiglu_params(d_model: int, d_ff: int) -> int:
    return 3 * d_model * d_ff


def main() -> None:
    d_model = 64
    x = np.random.default_rng(1).normal(size=(5, d_model))

    print("=== 1. FFN klasik ===")
    p = make_ffn(d_model, 4 * d_model)
    y = ffn_forward(x, p)
    print(f"input      : {x.shape}")
    print(f"hidden     : {p['W1'].shape}  (ekspansi 4x)")
    print(f"output     : {y.shape}")
    print(f"parameter  : {ffn_params(d_model, 4 * d_model):,}")
    print(f"attention  : {4 * d_model * d_model:,} (MHA, tanpa bias)")
    print(f"rasio FFN/attention: {ffn_params(d_model, 4 * d_model) / (4 * d_model * d_model):.2f}x")

    print("\n=== 2. Activation function ===")
    z = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    print(f"{'x':>6} {'relu':>8} {'gelu':>8} {'silu':>8}")
    for value in z:
        print(f"{value:>6.1f} {relu(np.array([value]))[0]:>8.3f} "
              f"{gelu(np.array([value]))[0]:>8.3f} {silu(np.array([value]))[0]:>8.3f}")
    print("ReLU memotong negatif; GELU/SiLU membiarkan sedikit sinyal negatif lewat.")
    print("SwiGLU memakai SiLU sebagai gerbang, bukan sebagai aktivasi akhir.")

    print("\n=== 3. SwiGLU dan aturan 8/3 ===")
    d_ff = int(8 / 3 * d_model)
    d_ff = (d_ff + 7) // 8 * 8  # dibulatkan ke kelipatan 8 untuk efisiensi GPU
    p_sw = make_swiglu(d_model, d_ff)
    y_sw = swiglu_forward(x, p_sw)
    print(f"d_model={d_model}, d_ff SwiGLU={d_ff} ({d_ff / d_model:.2f}x)")
    print(f"parameter SwiGLU : {swiglu_params(d_model, d_ff):,}")
    print(f"parameter FFN 4x : {ffn_params(d_model, 4 * d_model):,}")
    print("dengan 8/3, jumlah parameter SwiGLU mendekati FFN 4x tetapi kualitasnya lebih baik.")

    print("\n=== 4. FFN tidak mencampur posisi ===")
    x2 = x.copy()
    x2[0] += 100.0  # ubah satu token saja
    y1 = ffn_forward(x, make_ffn(d_model, 4 * d_model))
    y2 = ffn_forward(x2, make_ffn(d_model, 4 * d_model))
    print(f"token 0 berubah -> selisih baris lain: {np.abs(y1[1:] - y2[1:]).max():.2e}")
    print("FFN hanya bekerja per token; tidak ada informasi yang menyebar antar posisi.")

    print("\n=== 5. Skala model nyata ===")
    for name, d, ff in [("GPT-2 small", 768, 3072), ("LLaMA-7B", 4096, 11008), ("LLaMA-70B", 8192, 28672)]:
        ffn_p = 3 * d * ff  # asumsi SwiGLU
        attn_p = 4 * d * d
        print(f"{name:<12} d={d:>5,} ffn={ff:>6,}: FFN {ffn_p / 1e6:>7.1f}M | "
              f"attn {attn_p / 1e6:>6.1f}M | FFN porsi {ffn_p / (ffn_p + attn_p):.0%}")

    print("\n=== 6. FLOPs per token ===")
    print("FFN  : ~2 * 3 * d * d_ff = 6 * d * d_ff FLOPs (multiply + add)")
    print(f"untuk d=4096, d_ff=11008: {6 * 4096 * 11008 / 1e9:.2f} GFLOP per token")
    print("itulah sebabnya kecepatan inference sangat ditentukan oleh FFN.")


if __name__ == "__main__":
    main()
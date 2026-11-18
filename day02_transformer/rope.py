"""Rotary Position Embedding (RoPE), dipakai LLaMA, Qwen, Mistral, Gemma.

Ide: jangan tambahkan vektor posisi ke embedding. Putar pasangan dimensi dari
query dan key sebesar sudut yang bergantung pada posisi.

Untuk pasangan dimensi (2i, 2i+1) pada posisi m:
    [q'_2i  ]   [cos(m*t_i)  -sin(m*t_i)] [q_2i  ]
    [q'_2i+1] = [sin(m*t_i)   cos(m*t_i)] [q_2i+1]

dengan t_i = base^(-2i/d) dan base biasanya 10.000 (bisa 500.000 untuk
long context).

Kenapa penting: hasil dot product q'_m . k'_n hanya bergantung pada (m - n).
Artinya attention melihat jarak relatif, dan rotasi ini tidak menambah parameter.

Jalankan: python day02_transformer/rope.py
"""

from __future__ import annotations

import numpy as np


def rope_angles(dim: int, positions: np.ndarray, base: float = 10_000.0) -> np.ndarray:
    """Sudut rotasi untuk setiap posisi dan setiap pasangan dimensi.

    Output: (n_positions, dim/2)
    """
    if dim % 2 != 0:
        raise ValueError("dim harus genap")
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))  # (dim/2,)
    return positions[:, None] * inv_freq[None, :]


def rotate_half(x: np.ndarray) -> np.ndarray:
    """[x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = -x2
    out[..., 1::2] = x1
    return out


def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10_000.0) -> np.ndarray:
    """Putar vektor x (shape: (seq, dim) atau (heads, seq, dim))."""
    dim = x.shape[-1]
    angles = rope_angles(dim, positions, base)
    cos = np.cos(angles)
    sin = np.sin(angles)
    if x.ndim == 3:  # (heads, seq, dim) -> broadcast per head
        cos = cos[None, :, :]
        sin = sin[None, :, :]
    return x * np.repeat(cos, 2, axis=-1) + rotate_half(x) * np.repeat(sin, 2, axis=-1)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def main() -> None:
    dim = 8
    rng = np.random.default_rng(0)
    q = rng.normal(size=dim)
    k = rng.normal(size=dim)

    print("=== 1. Efek rotasi terhadap norma ===")
    q_pos = apply_rope(q[None, :], np.array([3]))[0]
    print(f"norma sebelum: {np.linalg.norm(q):.4f}")
    print(f"norma sesudah: {np.linalg.norm(q_pos):.4f}  (rotasi tidak mengubah panjang)")
    print(f"angle[3]     : {np.round(rope_angles(dim, np.array([3]))[0], 4)}")

    print("\n=== 2. Dot product hanya bergantung pada jarak relatif ===")
    print(f"{'m':>4} {'n':>4} {'m-n':>5} {'q_m . k_n':>12}")
    distances = {}
    for m, n in [(0, 0), (5, 5), (5, 4), (10, 9), (50, 49), (10, 6), (30, 26), (100, 96)]:
        qm = apply_rope(q[None, :], np.array([m]))[0]
        kn = apply_rope(k[None, :], np.array([n]))[0]
        value = dot(qm, kn)
        print(f"{m:>4} {n:>4} {m - n:>5} {value:>12.6f}")
        distances.setdefault(m - n, value)
    print("\njarak sama -> nilai sama (sampai presisi float).")
    print("inilah properti yang membuat RoPE 'relatif'.")

    print("\n=== 3. Bandingkan dengan absolute positional encoding ===")
    from positional_encoding import sinusoidal_encoding  # noqa: E402

    pe = sinusoidal_encoding(256, dim)
    print(f"{'jarak':>6} {'RoPE@30':>10} {'sin@10':>10} {'sin@60':>10}")
    for d in [0, 1, 2, 8, 16, 32]:
        q_rope = apply_rope(q[None, :], np.array([30]))[0]
        k_rope = apply_rope(k[None, :], np.array([30 + d]))[0]
        # absolute PE: posisi hanya ditambahkan ke q/k
        sin_10 = dot(pe[10] + q, pe[10 + d] + k)
        sin_60 = dot(pe[60] + q, pe[60 + d] + k)
        print(f"{d:>6} {dot(q_rope, k_rope):>10.4f} {sin_10:>10.4f} {sin_60:>10.4f}")
    print("\nRoPE: jarak yang sama memberi nilai yang sama di posisi mana pun.")
    print("sinusoidal: nilai untuk jarak 8 berbeda di posisi 10 dan 60, karena")
    print("posisi hanya ditambahkan dan tidak diputar secara relatif.")
    print("RoPE diterapkan pada q/k di setiap layer, sehingga sinyal posisi")
    print("tidak 'hilang' di residual stream seperti pada absolute PE.")

    print("\n=== 4. Kenapa base bisa 10.000 atau 500.000? ===")
    positions = np.arange(0, 4096)
    for base in [10_000.0, 500_000.0]:
        inv_freq = 1.0 / (base ** (np.arange(0, 64, 2) / 64))
        # periode = waktu (dalam token) untuk satu putaran penuh
        period = 2 * np.pi / inv_freq
        print(f"base={base:>9.0f}: periode dimensi tercepat = {period[0]:.2f} token, "
              f"terlambat = {period[-1]:,.0f} token")
    print("\nbase lebih besar -> rotasi lebih lambat -> model bisa membedakan")
    print("posisi yang jauh (long context), dengan sedikit penurunan resolusi lokal.")

    print("\n=== 5. RoPE tidak menambah parameter ===")
    print("tidak ada matriks yang dipelajari: cos/sin dihitung dari posisi.")
    print("biayanya hanya sedikit operasi trigonometri + perkalian elemen.")


if __name__ == "__main__":
    main()
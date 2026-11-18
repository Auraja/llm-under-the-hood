"""Positional encoding sinusoidal (Vaswani et al., 2017).

Masalah yang diselesaikan:
Attention murni bersifat *permutation equivariant*: kalau urutan token diacak,
output-nya ikut teracak dengan cara yang sama. Model tidak tahu "kucing makan
ikan" berbeda dari "ikan makan kucing". Karena itu posisi harus disuntikkan.

Rumus:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Sifat penting: untuk pasangan dimensi (2i, 2i+1), encoding posisi p+k dapat
ditulis sebagai rotasi linear dari encoding posisi p. Karena itu dot product
antara dua encoding hanya bergantung pada jarak relatifnya - walaupun secara
implementasi ini masih *absolute* encoding yang ditambahkan ke embedding.

Jalankan: python day02_transformer/positional_encoding.py
"""

from __future__ import annotations

import numpy as np


def sinusoidal_encoding(max_len: int, d_model: int, base: float = 10_000.0) -> np.ndarray:
    """Matriks PE berukuran (max_len, d_model)."""
    if d_model % 2 != 0:
        raise ValueError("d_model harus genap")
    positions = np.arange(max_len)[:, None]                    # (max_len, 1)
    dims = np.arange(0, d_model, 2)[None, :]                   # (1, d/2)
    freqs = 1.0 / (base ** (dims / d_model))                   # (1, d/2)
    angles = positions * freqs                                  # (max_len, d/2)

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def relative_similarity(pe: np.ndarray, distance: int, n_pairs: int = 200) -> float:
    """Rata-rata dot product untuk pasangan posisi yang berjarak ``distance``."""
    max_start = len(pe) - distance
    starts = np.arange(min(n_pairs, max_start))
    dots = np.sum(pe[starts] * pe[starts + distance], axis=1)
    return float(dots.mean())


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def main() -> None:
    d_model, max_len = 16, 128
    pe = sinusoidal_encoding(max_len, d_model)

    print("=== 1. Bentuk dan contoh nilai ===")
    print(f"bentuk PE: {pe.shape}")
    print(f"posisi 0 : {np.round(pe[0], 3)}")
    print(f"posisi 1 : {np.round(pe[1], 3)}")
    print(f"posisi 2 : {np.round(pe[2], 3)}")
    print("\ndimensi awal berubah cepat (frekuensi tinggi),")
    print("dimensi akhir berubah lambat (frekuensi rendah) -> semacam 'jam' biner.")

    print("\n=== 2. Rentang nilai selalu [-1, 1] ===")
    print(f"min={pe.min():.3f} max={pe.max():.3f} mean={pe.mean():+.3f}")

    print("\n=== 3. Dot product bergantung pada jarak, bukan posisi absolut ===")
    print(f"{'jarak':>6} {'rata-rata dot':>15}")
    for distance in [0, 1, 2, 4, 8, 16, 32, 64]:
        print(f"{distance:>6} {relative_similarity(pe, distance):>15.3f}")
    print("\njarak kecil -> paling mirip; makin jauh -> dot produk turun.")
    print("polanya tidak monoton sempurna karena komponen cos punya rata-rata")
    print("positif, tetapi isyarat 'dekat vs jauh' tetap terbaca oleh model.")
    print("ini memberi model isyarat 'dekat vs jauh' tanpa parameter tambahan.")

    print("\n=== 4. Bukti bahwa ia menyandi posisi relatif ===")
    d = 8
    pe8 = sinusoidal_encoding(max_len, d)
    p, k = 10, 5
    print(f"dot(PE[{p}], PE[{p}])     = {dot(pe8[p], pe8[p]):.4f}")
    print(f"dot(PE[{p + k}], PE[{p}]) = {dot(pe8[p + k], pe8[p]):.4f}")
    print(f"dot(PE[{p + 20}], PE[{p + 15}]) = {dot(pe8[p + 20], pe8[p + 15]):.4f}")
    print("jarak 5 pada posisi berbeda memberi nilai yang sama -> relatif.")

    print("\n=== 5. PE ditambahkan, bukan digabung ===")
    rng = np.random.default_rng(0)
    token_vec = rng.normal(size=d_model)
    combined = token_vec + pe[3]
    print(f"norma token        : {np.linalg.norm(token_vec):.3f}")
    print(f"norma PE           : {np.linalg.norm(pe[3]):.3f}")
    print(f"norma gabungan     : {np.linalg.norm(combined):.3f}")
    print("penjumlahan tidak menaikkan dimensi, tetapi mencampur dua sinyal.")
    print("karena itu posisi 'terbaca' dari arah vektor hasil, bukan dari selisih eksplisit.")

    print("\n=== 6. Keterbatasan ===")
    print("- PE absolut tidak otomatis ekstrapolasi ke panjang > max_len training.")
    print("- Efeknya melemah setelah melewati banyak layer + LayerNorm.")
    print("- Model modern (LLaMA, Qwen, Mistral) memakai RoPE: posisi diputar")
    print("  langsung di dalam attention, sehingga sifat relatifnya lebih kuat.")


if __name__ == "__main__":
    main()
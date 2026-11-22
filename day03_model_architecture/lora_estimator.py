"""LoRA: estimasi parameter, memori, dan pembuktian bahwa adapter bisa di-merge.

Rumus LoRA untuk satu matriks bobot W (out_features x in_features):
    W' = W + (alpha / r) * B @ A
    A: (r, in_features), B: (out_features, r)
Parameter tambahan = r * (in_features + out_features), jauh lebih kecil dari
in_features * out_features.

Aspek penting:
- ``r`` (rank) mengatur kapasitas adapter. r=8..64 biasanya cukup.
- ``alpha`` mengatur skala; scale = alpha/r. B diinisialisasi nol sehingga
  adapter awalnya tidak mengubah output model.
- Setelah training, W' bisa dihitung dan disimpan sebagai satu matriks biasa.
  Tidak ada biaya inference tambahan setelah merge.
- Kalau tidak di-merge, LoRA menambah satu matmul kecil per modul (overhead
  kecil tetapi nyata pada batch besar).

Jalankan: python day03_model_architecture/lora_estimator.py
"""

from __future__ import annotations

import numpy as np

# (d_in, d_out) untuk model 7B
MODULES = {
    "q": (4096, 4096),
    "k": (4096, 4096),
    "v": (4096, 4096),
    "o": (4096, 4096),
    "gate": (4096, 11008),
    "up": (4096, 11008),
    "down": (11008, 4096),
}


def lora_params_module(d_in: int, d_out: int, rank: int) -> int:
    return rank * (d_in + d_out)


def lora_params_total(rank: int, targets: tuple[str, ...], n_layers: int) -> int:
    per_layer = sum(lora_params_module(*MODULES[t], rank) for t in targets)
    return per_layer * n_layers


def lora_memory(n_params: int, dtype_bytes: int = 2, optimizer: str = "adam") -> dict:
    """Bobot + gradient + state optimizer (fp32) untuk parameter adapter saja."""
    weights = n_params * dtype_bytes
    grads = n_params * dtype_bytes
    optim = n_params * 8 if optimizer == "adam" else 0
    return {"weights": weights, "grads": grads, "optimizer": optim,
            "total": weights + grads + optim}


def merge(W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, rank: int) -> np.ndarray:
    return W + (alpha / rank) * (B @ A)


def main() -> None:
    print("=== 1. Parameter per modul (r=8) ===")
    print(f"{'modul':<12} {'bentuk':>14} {'penuh':>12} {'LoRA':>9} {'rasio':>8}")
    for name, (d_in, d_out) in MODULES.items():
        full = d_in * d_out
        low = lora_params_module(d_in, d_out, 8)
        print(f"{name + '_proj':<12} {f'{d_out}x{d_in}':>14} {full:>12,} {low:>9,} {low / full:>7.2%}")
    print("\nproyeksi persegi (q/k/v/o) punya rasio LoRA lebih besar daripada")
    print("proyeksi FFN, karena rank 8 dibanding d=4096 sudah cukup signifikan.")

    print("\n=== 2. Total untuk 7B (32 layer) ===")
    print(f"{'target':<34} {'r=8':>10} {'r=16':>10} {'r=64':>11}")
    for targets, label in [
        (("q", "v"), "q, v"),
        (("q", "k", "v", "o"), "q, k, v, o (attention)"),
        (("q", "k", "v", "o", "gate", "up", "down"), "semua proyeksi"),
    ]:
        row = [f"{lora_params_total(r, targets, 32) / 1e6:>9.2f}M" for r in (8, 16, 64)]
        print(f"{label:<34} {row[0]:>10} {row[1]:>10} {row[2]:>11}")
    total_7b = 7_000_000_000
    print(f"\nsemua proyeksi r=16: {lora_params_total(16, tuple(MODULES), 32) / total_7b:.2%} dari parameter 7B")

    print("\n=== 3. Memori adapter (r=16, semua proyeksi) ===")
    n = lora_params_total(16, tuple(MODULES), 32)
    mem = lora_memory(n)
    print(f"parameter adapter : {n:,}")
    for key, value in mem.items():
        print(f"  {key:<9} {value / 1e6:>8.1f} MB")
    full_ft = total_7b * (2 + 2 + 8 + 4)  # bobot+grad+adam+master fp32
    print(f"full fine-tuning  : {full_ft / 1e9:>8.1f} GB (state training saja)")
    print(f"rasio             : {full_ft / mem['total']:,.0f}x lebih hemat")

    print("\n=== 4. Merge: output identik tanpa overhead ===")
    rng = np.random.default_rng(0)
    d_in, d_out, rank, alpha = 64, 48, 4, 8.0
    W = rng.normal(0, 0.1, (d_out, d_in))
    A = rng.normal(0, 0.02, (rank, d_in))
    B = np.zeros((d_out, rank))  # inisialisasi nol: adapter mulai sebagai identitas
    x = rng.normal(size=(5, d_in))

    base_out = x @ W.T
    lora_out = x @ W.T + (alpha / rank) * (x @ A.T @ B.T)
    print(f"B=0 -> selisih adapter vs base: {np.abs(base_out - lora_out).max():.2e}")

    B = rng.normal(0, 0.02, (d_out, rank))  # setelah training
    with_adapter = x @ W.T + (alpha / rank) * (x @ A.T @ B.T)
    merged = merge(W, A, B, alpha, rank)
    merged_out = x @ merged.T
    print(f"setelah training, tanpa merge : {with_adapter.shape}, dua matmul")
    print(f"setelah merge                 : {merged_out.shape}, satu matmul")
    print(f"selisih maksimum              : {np.abs(with_adapter - merged_out).max():.2e}")
    print("secara matematis identik; merge hanya mengubah urutan operasi.")

    print("\n=== 5. Memilih rank ===")
    for rank in [1, 4, 8, 16, 32, 64, 128]:
        n = lora_params_total(rank, tuple(MODULES), 32)
        print(f"r={rank:>3}: {n / 1e6:>6.2f}M parameter, "
              f"{(n * 2) / 1e9:.3f} GB bobot adapter (fp16)")
    print("\nr kecil  -> underfit pada tugas kompleks.")
    print("r besar  -> lebih dekat ke full fine-tuning, tetapi memori dan overfit naik.")
    print("mulai dari r=16 pada semua proyeksi; turunkan kalau data sedikit.")

    print("\n=== 6. Target module mana? ===")
    print("q,v saja           : paling murah, sering cukup untuk gaya/format.")
    print("q,k,v,o            : standar, keseimbangan.")
    print("+ gate,up,down     : kapasitas terbesar, paling dekat ke full FT.")
    print("MLP-only           : kadang lebih baik untuk pengetahuan baru.")


if __name__ == "__main__":
    main()
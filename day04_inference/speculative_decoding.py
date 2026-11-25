"""Speculative decoding: model kecil menebak, model besar memverifikasi.

Alur satu langkah:
1. Draft model (kecil/cepat) menghasilkan gamma token.
2. Target model (besar) menghitung probabilitas untuk semua token itu SEKALIGUS
   (satu forward pass, karena verifikasi bersifat paralel).
3. Token diterima dengan probabilitas min(1, p_target / p_draft).
   Kalau ditolak, sample dari distribusi residual max(0, p_target - p_draft).
4. Kalau semua diterima, target model memberi satu token bonus.

Sifat penting: hasil akhirnya SECARA DISTRIBUSI sama dengan sampling dari target
model. Speculative decoding bukan aproksimasi, asalkan aturan penerimaan benar.

Speedup = token per langkah / biaya per langkah
        ~ (1 + alpha + alpha^2 + ...) / (1 + c * gamma)
dengan alpha = tingkat penerimaan rata-rata dan c = biaya draft relatif target.

Jalankan: python day04_inference/speculative_decoding.py
"""

from __future__ import annotations

import numpy as np


def acceptance_probability(p: float, q: float) -> float:
    """Peluang menerima token yang diusulkan draft dengan prob q."""
    if q <= 0:
        return 1.0
    return min(1.0, p / q)


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """max(0, p - q) yang dinormalisasi; dipakai saat token ditolak."""
    residual = np.maximum(p - q, 0.0)
    total = residual.sum()
    if total <= 0:
        return p.copy()
    return residual / total


def speculate_step(p: np.ndarray, q: np.ndarray, gamma: int,
                   rng: np.random.Generator) -> tuple[list[int], bool]:
    """Satu langkah spekulasi (draft i.i.d. dari q, cukup untuk analisis)."""
    accepted: list[int] = []
    for _ in range(gamma):
        token = int(rng.choice(len(q), p=q))
        if rng.random() < acceptance_probability(p[token], q[token]):
            accepted.append(token)
        else:
            # tolak: ambil dari residual lalu berhenti
            residual = residual_distribution(p, q)
            accepted.append(int(rng.choice(len(p), p=residual)))
            return accepted, False
    # semua diterima -> token bonus dari target
    accepted.append(int(rng.choice(len(p), p=p)))
    return accepted, True


def expected_accepted(p: np.ndarray, q: np.ndarray, gamma: int) -> float:
    """Ekspektasi token yang dihasilkan per langkah.

    Selalu ada minimal 1 token (hasil residual saat ditolak), plus peluang
    token ke-i diterima sebesar alpha^i, ditambah 1 token bonus kalau semua
    proposal diterima (alpha^gamma).
    """
    alpha = float(np.minimum(p, q).sum())
    return 1.0 + sum(alpha**i for i in range(1, gamma + 1))


def speedup(p: np.ndarray, q: np.ndarray, gamma: int, cost_ratio: float) -> float:
    return expected_accepted(p, q, gamma) / (1 + cost_ratio * gamma)


def main() -> None:
    rng = np.random.default_rng(0)

    # target model (besar) dan draft model (kecil) atas 6 token
    p = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    q = np.array([0.30, 0.30, 0.20, 0.10, 0.06, 0.04])

    print("=== 1. Dua distribusi ===")
    print(f"target p: {np.round(p, 2)}")
    print(f"draft  q: {np.round(q, 2)}")
    print(f"total overlap min(p,q) = alpha = {np.minimum(p, q).sum():.3f}")
    print("alpha adalah peluang rata-rata sebuah proposal diterima.")

    print("\n=== 2. Simulasi 20.000 langkah (gamma=4) ===")
    gamma, steps = 4, 20_000
    histogram = np.zeros(len(p))
    accepted_counts = []
    bonus = 0
    for _ in range(steps):
        tokens, all_accepted = speculate_step(p, q, gamma, rng)
        accepted_counts.append(len(tokens))
        bonus += int(all_accepted)
        for token in tokens:
            histogram[token] += 1
    histogram /= histogram.sum()

    print(f"rata-rata token per langkah : {np.mean(accepted_counts):.2f} "
          f"(teori {expected_accepted(p, q, gamma):.2f})")
    print(f"semua proposal diterima     : {bonus / steps:.1%}")
    print(f"{'token':>6} {'target p':>10} {'hasil simulasi':>15} {'selisih':>9}")
    for i in range(len(p)):
        print(f"{i:>6} {p[i]:>10.3f} {histogram[i]:>15.3f} {abs(histogram[i] - p[i]):>9.4f}")
    print("\ndistribusi output tetap mengikuti target p -> bukan aproksimasi.")

    print("\n=== 3. Gamma vs speedup ===")
    print("c = biaya draft / biaya target (0.05 = draft 20x lebih murah)")
    print(f"{'gamma':>6} " + " ".join(f"{f'c={c:.2f}':>8}" for c in [0.02, 0.05, 0.1, 0.2]))
    for g in [1, 2, 4, 6, 8]:
        row = " ".join(f"{speedup(p, q, g, c):>8.2f}x" for c in [0.02, 0.05, 0.1, 0.2])
        print(f"{g:>6} {row}")

    print("\n=== 4. Efek kualitas draft model ===")
    drafts = {
        "sangat mirip": np.array([0.39, 0.25, 0.15, 0.10, 0.07, 0.04]),
        "mirip": np.array([0.30, 0.30, 0.20, 0.10, 0.06, 0.04]),
        "lemah": np.array([0.20, 0.20, 0.20, 0.20, 0.10, 0.10]),
    }
    for name, draft in drafts.items():
        alpha = float(np.minimum(p, draft).sum())
        print(f"{name:<12} alpha={alpha:.3f} -> token/langkah={expected_accepted(p, draft, 4):.2f}, "
              f"speedup(c=0.05)={speedup(p, draft, 4, 0.05):.2f}x")
    print("\ndraft model yang buruk bisa membuat decoding lebih lambat dari baseline.")

    print("\n=== 5. Kapan dipakai ===")
    print("- Ada model kecil dari keluarga yang sama (draft dan target sepakat).")
    print("- Beban GPU belum jenuh; verifikasi paralel memanfaatkan idle compute.")
    print("- Output panjang dan mudah diprediksi (kode, ringkasan format).")
    print("Tidak cocok kalau draft sering salah atau batch sudah penuh.")

    print("\n=== 6. Alternatif ===")
    print("Medusa / EAGLE: beberapa 'head' tambahan memprediksi token masa depan.")
    print("Lookahead decoding: tanpa model draft terpisah.")
    print("Semua memakai ide yang sama: usulkan banyak token, verifikasi sekaligus.")


if __name__ == "__main__":
    main()
"""Cross-entropy loss dan perplexity, dari nol.

Cross-entropy untuk satu token:
    loss = -log(p_target)

Perplexity:
    ppl = exp(loss)

Interpretasi: perplexity adalah jumlah pilihan yang "efektif" sedang
dipertimbangkan model. ppl=1 berarti model yakin 100%. ppl=50.000 berarti model
sama tidak tahunya dengan menebak acak dari 50.000 token.

Fakta penting:
- Loss dan perplexity hanya bermakna untuk tokenizer yang sama. PPL 10 pada
  tokenizer A tidak bisa dibandingkan dengan PPL 10 pada tokenizer B.
- Nilai loss rata-rata (mean over tokens) dipakai untuk training; nilai
  perplexity dipakai untuk komunikasi.
- `bits per token = loss / ln(2)` menghubungkan loss dengan kompresi.

Jalankan: python day03_model_architecture/loss_perplexity.py
"""

from __future__ import annotations

import math

import numpy as np


def cross_entropy(prob_target: float) -> float:
    if not 0 < prob_target <= 1:
        raise ValueError("probabilitas harus di (0, 1]")
    return -math.log(prob_target)


def perplexity(loss: float) -> float:
    return math.exp(loss)


def mean_loss(probs: list[float]) -> float:
    return sum(cross_entropy(p) for p in probs) / len(probs)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature harus > 0")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()
    exp = np.exp(z)
    return exp / exp.sum()


def bits_per_token(loss: float) -> float:
    return loss / math.log(2)


def main() -> None:
    print("=== 1. Loss untuk satu prediksi ===")
    print(f"{'p(target)':>10} {'loss':>8} {'ppl':>10} {'bits':>8}")
    for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
        loss = cross_entropy(p)
        print(f"{p:>10.2f} {loss:>8.3f} {perplexity(loss):>10.2f} {bits_per_token(loss):>8.3f}")
    print("\nppl satu token = 1/p. Model yang yakin (p=0.99) punya ppl 1.01.")

    print("\n=== 2. Baseline acak ===")
    for vocab in [32_000, 50_257, 128_256]:
        uniform_loss = math.log(vocab)
        print(f"vocab {vocab:>7,}: loss={uniform_loss:.3f}, ppl={perplexity(uniform_loss):,.0f}")
    print("setiap model harus jauh di bawah angka ini agar dianggap belajar sesuatu.")

    print("\n=== 3. Satu sequence ===")
    probs = [0.8, 0.6, 0.95, 0.4, 0.7, 0.3]
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    print(f"{'token':>8} {'p':>6} {'-log p':>8}")
    for tok, p in zip(tokens, probs):
        print(f"{tok:>8} {p:>6.2f} {cross_entropy(p):>8.3f}")
    avg = mean_loss(probs)
    print(f"\nmean loss  : {avg:.4f}")
    print(f"perplexity : {perplexity(avg):.2f}")
    print(f"bits/token : {bits_per_token(avg):.3f}")
    print(f"geometric mean probabilitas: {math.exp(-avg):.4f} (selalu <= arithmetic mean)")

    print("\n=== 4. Efek temperature pada loss ===")
    logits = np.array([3.0, 1.5, 0.5, -1.0])
    target_index = 0
    print(f"{'T':>5} {'p(target)':>10} {'loss':>8} {'ppl':>8}")
    for temperature in [0.5, 1.0, 2.0, 4.0]:
        p = softmax(logits, temperature)
        loss = cross_entropy(float(p[target_index]))
        print(f"{temperature:>5.1f} {p[target_index]:>10.3f} {loss:>8.3f} {perplexity(loss):>8.3f}")
    print("\nT < 1 mempertajam distribusi -> loss turun, tapi belum tentu lebih benar.")
    print("T > 1 meratakan distribusi -> loss naik. Training memakai T=1.")

    print("\n=== 5. Hubungan dengan kompresi ===")
    text = "the quick brown fox jumps over the lazy dog"
    n_tokens = 10
    loss = 1.2
    total_bits = bits_per_token(loss) * n_tokens
    print(f"loss {loss} -> {bits_per_token(loss):.3f} bit/token")
    print(f"{n_tokens} token -> {total_bits:.1f} bit = {total_bits / 8:.2f} byte")
    print("model bahasa yang baik adalah kompresor yang baik;")
    print("perplexity rendah berarti lebih sedikit bit untuk menyandi teks.")

    print("\n=== 6. Jebakan saat membandingkan PPL ===")
    print("- Tokenizer beda -> jumlah token beda -> PPL tidak sebanding.")
    print("- Evaluasi pada teks yang bocor ke data training -> PPL terlalu bagus.")
    print("- PPL rendah tidak menjamin jawaban benar; ia hanya mengukur prediksi token.")
    print("- Chat model dilatih dengan masking prompt; PPL dihitung pada token jawaban.")


if __name__ == "__main__":
    main()
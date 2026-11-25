"""Simulasi decoding: greedy, beam search, top-k, top-p.

Model mainan di sini adalah tabel logits per konteks (bukan neural network),
supaya fokusnya ke algoritma pemilihan token.

Yang perlu dilihat:
- Greedy memilih argmax; cepat, deterministik, tapi bisa terjebak.
- Beam search menyimpan beberapa hipotesis dan memilih total log-prob terbaik;
  sering menghasilkan kalimat "aman" dan berulang.
- Sampling (top-k/top-p + temperature) menghasilkan variasi.
- Beam search memaksimalkan probabilitas sequence, BUKAN kualitas jawaban.

Jalankan: python day04_inference/decoding_simulation.py
"""

from __future__ import annotations

import math

import numpy as np

VOCAB = ["<bos>", "the", "cat", "sat", "on", "mat", "<eos>"]
IDX = {t: i for i, t in enumerate(VOCAB)}

# logits untuk token berikutnya, dikondisikan pada token terakhir
TABLE: dict[str, dict[str, float]] = {
    "<bos>": {"the": 2.4, "cat": 1.1},
    "the": {"cat": 2.2, "mat": 1.5},
    "cat": {"sat": 2.6, "on": 1.2},
    "sat": {"on": 2.8, "mat": 1.1},
    "on": {"mat": 2.7, "the": 1.8},
    "mat": {"<eos>": 2.9, "on": 1.0},
    "<eos>": {},
}


def logits_for(last_token: str) -> np.ndarray:
    """Vektor logits penuh; token yang tidak mungkin diberi -inf."""
    logits = np.full(len(VOCAB), -np.inf)
    for token, value in TABLE.get(last_token, {}).items():
        logits[IDX[token]] = value
    return logits


def softmax(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    z = x[finite] - x[finite].max()
    e = np.exp(z)
    out[finite] = e / e.sum()
    return out


def greedy(max_steps: int = 10) -> tuple[list[str], float]:
    tokens = ["<bos>"]
    score = 0.0
    for _ in range(max_steps):
        logits = logits_for(tokens[-1])
        if not np.isfinite(logits).any():
            break
        idx = int(np.argmax(logits))
        probs = softmax(logits)
        score += math.log(probs[idx])
        tokens.append(VOCAB[idx])
        if VOCAB[idx] == "<eos>":
            break
    return tokens, score


def beam_search(width: int = 3, max_steps: int = 8,
                length_penalty: float = 0.0) -> list[tuple[list[str], float]]:
    """Kembalikan ``width`` hipotesis teratas (skor = total log-prob)."""
    beams: list[tuple[list[str], float]] = [([ "<bos>"], 0.0)]
    finished: list[tuple[list[str], float]] = []

    for _ in range(max_steps):
        candidates: list[tuple[list[str], float]] = []
        for tokens, score in beams:
            if tokens[-1] == "<eos>":
                finished.append((tokens, score))
                continue
            logits = logits_for(tokens[-1])
            probs = softmax(logits)
            for idx in np.argsort(probs)[::-1][:width]:
                if probs[idx] == 0:
                    continue
                new_tokens = tokens + [VOCAB[idx]]
                new_score = score + math.log(float(probs[idx]))
                if VOCAB[idx] == "<eos>":
                    finished.append((new_tokens, new_score))
                else:
                    candidates.append((new_tokens, new_score))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:width]

    def normalized(beam: tuple[list[str], float]) -> float:
        tokens, score = beam
        return score / (len(tokens) ** length_penalty if length_penalty else 1)

    return sorted(finished + beams, key=normalized, reverse=True)[:width]


def sample_sequence(max_steps: int = 8, temperature: float = 1.0,
                    top_k: int = 0, top_p: float = 1.0,
                    rng: np.random.Generator | None = None) -> list[str]:
    rng = rng or np.random.default_rng(0)
    tokens = ["<bos>"]
    for _ in range(max_steps):
        logits = logits_for(tokens[-1]) / temperature
        probs = softmax(logits)

        if top_k and 0 < top_k < len(probs):
            keep = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask / mask.sum()
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            cutoff = int(np.searchsorted(cumulative, top_p) + 1)
            mask = np.zeros_like(probs)
            mask[order[:cutoff]] = probs[order[:cutoff]]
            probs = mask / mask.sum()

        idx = int(rng.choice(len(VOCAB), p=probs))
        if not np.isfinite(logits[idx]):
            break
        tokens.append(VOCAB[idx])
        if VOCAB[idx] == "<eos>":
            break
    return tokens


def sentence(tokens: list[str]) -> str:
    return " ".join(t for t in tokens if t not in ("<bos>", "<eos>"))


def main() -> None:
    print("=== 1. Greedy ===")
    tokens, score = greedy()
    print(f"tokens : {tokens}")
    print(f"teks   : {sentence(tokens)!r}")
    print(f"logprob total: {score:.3f} (rata-rata {score / (len(tokens) - 1):.3f}/token)")

    print("\n=== 2. Beam search (width=3) ===")
    for rank, (tokens, score) in enumerate(beam_search(width=3), 1):
        print(f"beam {rank}: {sentence(tokens)!r:<22} logprob={score:.3f} "
              f"rata2={score / max(len(tokens) - 1, 1):.3f}")
    print("\nbeam search mencari total log-prob terbaik. Perhatikan bahwa")
    print("kalimat dengan rata-rata per token lebih baik bisa kalah karena")
    print("logprob total selalu turun seiring panjang sequence.")

    print("\n=== 3. Length normalization ===")
    print(f"{'penalty':>8} {'beam teratas':<24} {'skor ternormalisasi':>20}")
    for penalty in [0.0, 0.7, 1.0]:
        best = beam_search(width=3, length_penalty=penalty)[0]
        norm = best[1] / (len(best[0]) ** penalty if penalty else 1)
        print(f"{penalty:>8.1f} {sentence(best[0])!r:<24} {norm:>20.3f}")
    print("penalty=1.0 membagi dengan panjang -> sequence panjang tidak otomatis kalah.")

    print("\n=== 4. Sampling 5x per setelan ===")
    rng = np.random.default_rng(7)
    for label, temperature, k, p in [
        ("T=0.5", 0.5, 0, 1.0),
        ("T=1.0", 1.0, 0, 1.0),
        ("T=1.0 top_k=2", 1.0, 2, 1.0),
        ("T=1.0 top_p=0.8", 1.0, 0, 0.8),
        ("T=2.0", 2.0, 0, 1.0),
    ]:
        outputs = [sentence(sample_sequence(temperature=temperature, top_k=k,
                                            top_p=p, rng=rng)) for _ in range(5)]
        print(f"{label:<18} {len(set(outputs))} unik: {outputs}")

    print("\n=== 5. Kenapa hasil greedy/beam bisa 'membosankan'? ===")
    print("model dilatih memaksimalkan probabilitas token berikutnya,")
    print("bukan kualitas paragraf. Beam search menambah bias ke arah")
    print("sequence yang paling mungkin secara statistik, yang sering berarti")
    print("pengulangan dan frasa aman. Sampling mengembalikan keragaman,")
    print("dengan risiko keluar dari fakta kalau temperature terlalu tinggi.")

    print("\n=== 6. Biaya komputasi ===")
    seq_len = 12
    comparisons = {"greedy": 1, "beam width 3": 3, "beam width 8": 8}
    print(f"model calls untuk {seq_len} token:")
    for name, multiplier in comparisons.items():
        print(f"  {name:<14}: {seq_len * multiplier:>3} forward pass")
    print("sampling setara greedy (1 forward per token), hanya berbeda cara memilih.")


if __name__ == "__main__":
    main()
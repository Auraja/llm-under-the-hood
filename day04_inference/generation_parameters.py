"""Parameter generasi: temperature, top-k, top-p, repetition penalty, stop.

Semua parameter ini bekerja pada SATU hal yang sama: distribusi probabilitas
token berikutnya. Tidak ada yang mengubah bobot model.

- temperature T   : logits dibagi T. T<1 mempertajam, T>1 meratakan.
- top_k           : simpan k token teratas, sisanya dibuang.
- top_p (nucleus) : simpan token tersering sampai total probabilitas >= p.
- repetition_penalty : turunkan logit token yang sudah muncul.
- max_tokens / stop : memotong panjang output, bukan distribusi.

Bagian 1-4 murni simulasi lokal (tanpa API). Bagian 5 opsional memanggil API.

Jalankan: python day04_inference/generation_parameters.py
"""

from __future__ import annotations

import numpy as np

VOCAB = ["the", "cat", "sat", "on", "mat", "and", "a", "dog", "ran", "away"]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature harus > 0")
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= len(probs):
        return probs.copy()
    keep = np.argsort(probs)[-k:]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def apply_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    if not 0 < p <= 1:
        raise ValueError("top_p harus di (0, 1]")
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cumulative, p) + 1)
    keep = order[:cutoff]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def apply_repetition_penalty(logits: np.ndarray, seen: list[int], penalty: float) -> np.ndarray:
    """Penalty sederhana: logit positif dibagi, negatif dikali."""
    if penalty <= 1.0 or not seen:
        return logits
    out = logits.copy()
    for idx in set(seen):
        if out[idx] > 0:
            out[idx] /= penalty
        else:
            out[idx] *= penalty
    return out


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def show_distribution(label: str, probs: np.ndarray, top: int = 5) -> None:
    order = np.argsort(probs)[::-1][:top]
    print(f"{label:<28} entropy={entropy(probs):.2f} "
          + " ".join(f"{VOCAB[i]}={probs[i]:.2f}" for i in order))


def main() -> None:
    logits = np.array([3.0, 2.4, 1.8, 1.2, 0.9, 0.4, 0.2, -0.3, -1.0, -2.0])

    print("=== 1. Temperature ===")
    for temperature in [0.2, 0.7, 1.0, 1.5, 2.5]:
        show_distribution(f"T={temperature}", softmax(logits, temperature))
    print("\nT rendah: hampir deterministik (entropy kecil).")
    print("T tinggi: ekor distribusi ikut hidup, output lebih beragam.")

    print("\n=== 2. Top-k ===")
    base = softmax(logits, 1.0)
    for k in [1, 3, 5, 10]:
        filtered = apply_top_k(base, k)
        show_distribution(f"top_k={k}", filtered)
    print("\ntop_k=1 identik dengan greedy decoding.")

    print("\n=== 3. Top-p (nucleus) ===")
    for p in [0.5, 0.8, 0.95, 1.0]:
        filtered = apply_top_p(base, p)
        kept = int((filtered > 0).sum())
        show_distribution(f"top_p={p} ({kept} token)", filtered)
    print("\ntop_p beradaptasi: saat distribusi tajam, sedikit token yang disimpan;")
    print("saat distribusi datar, lebih banyak token ikut dipertimbangkan.")

    print("\n=== 4. Repetition penalty ===")
    seen = [0, 1, 2]  # 'the', 'cat', 'sat' sudah muncul
    penalized = apply_repetition_penalty(logits, seen, penalty=1.5)
    show_distribution("tanpa penalty", softmax(logits, 1.0))
    show_distribution("penalty=1.5", softmax(penalized, 1.0))
    print("\npenalty terlalu besar -> model menghindari kata umum dan jadi tidak wajar.")

    print("\n=== 5. Simulasi 5 sample per setelan ===")
    rng = np.random.default_rng(0)
    settings = [
        ("greedy", 0.0, 1, 1.0),
        ("T=0.7, top_p=0.9", 0.7, 0, 0.9),
        ("T=1.0, top_k=5", 1.0, 5, 1.0),
        ("T=1.6, top_k=5", 1.6, 5, 1.0),
        ("T=1.0, tanpa filter", 1.0, 0, 1.0),
    ]
    print(f"{'setelan':<22} {'variasi output (5 sample)':<40} {'token unik':>10}")
    for label, temperature, k, p in settings:
        probs = softmax(logits, temperature) if temperature > 0 else None
        if probs is None:
            outputs = [VOCAB[int(np.argmax(logits))]] * 5
        else:
            probs = apply_top_k(probs, k)
            probs = apply_top_p(probs, p)
            outputs = [VOCAB[sample(probs, rng)] for _ in range(5)]
        print(f"{label:<22} {', '.join(outputs):<40} {len(set(outputs)):>10}")

    print("\n=== 6. max_tokens dan stop sequence ===")
    print("keduanya memotong OUTPUT, bukan mengubah distribusi.")
    print("finish_reason='length' berarti output terpotong oleh max_tokens;")
    print("jawaban bisa terlihat selesai padahal belum (penting untuk parsing JSON).")

    print("\n=== 7. Rekomendasi awal ===")
    print("data extraction / kode : T=0.0-0.3, top_p=1")
    print("chat umum              : T=0.7, top_p=0.9")
    print("brainstorm             : T=1.0-1.2, top_p=0.95")
    print("jangan set T dan top_p ekstrem bersamaan; efeknya saling menutupi.")

    print("\n=== 8. Uji ke API (opsional) ===")
    try:
        from api_client import chat, has_api_key
    except ImportError:
        print("api_client tidak ditemukan; lewati.")
        return
    if not has_api_key():
        print("API key tidak ada -> bagian ini dilewati (simulasi di atas sudah cukup).")
        return

    prompt = "Lanjutkan dengan tepat 8 kata: Di dalam lab, model bahasa"
    for temperature in [0.0, 0.9]:
        try:
            r = chat([{"role": "user", "content": prompt}],
                     max_tokens=24, temperature=temperature)
            print(f"T={temperature}: {r['content'].strip()!r} "
                  f"({r['completion_tokens']} token, {r['latency_ms']:.0f} ms)")
        except RuntimeError as exc:
            print(f"T={temperature}: gagal - {exc}")


if __name__ == "__main__":
    main()
"""Token embedding: bagaimana id token menjadi vektor.

Tiga hal yang sering disalahpahami:
1. Embedding bukan "arti kata" yang diberikan manusia; nilainya dipelajari dari
   data dan hanya bermakna secara relatif terhadap vektor lain.
2. Operasi lookup embedding identik dengan one-hot matrix multiply. Secara
   matematis sama, secara komputasi lookup jauh lebih murah.
3. Weight tying: banyak model memakai matriks embedding yang sama untuk lapisan
   output (unembedding). Ini menghemat parameter sebesar vocab_size * d_model.

Jalankan: python day02_transformer/embeddings.py
"""

from __future__ import annotations

import numpy as np


def make_embedding(vocab_size: int, d_model: int, seed: int = 0) -> np.ndarray:
    """Matriks embedding (vocab_size, d_model).

    Di model nyata nilainya diinisialisasi kecil (mis. normal std=0.02) supaya
    logits awal tidak terlalu besar.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, size=(vocab_size, d_model))


def embed_lookup(token_ids: list[int], table: np.ndarray) -> np.ndarray:
    """Lookup: ambil baris ke-id. Ini yang dilakukan GPU (gather)."""
    return table[np.asarray(token_ids)]


def embed_onehot(token_ids: list[int], table: np.ndarray) -> np.ndarray:
    """Cara ' textbook': one-hot (n, vocab) @ (vocab, d) -> (n, d).

    Hasilnya identik dengan lookup, tapi memori one-hot membengkak:
    n * vocab angka, padahal yang dibutuhkan hanya n * d.
    """
    n, vocab = len(token_ids), table.shape[0]
    onehot = np.zeros((n, vocab), dtype=np.float64)
    onehot[np.arange(n), token_ids] = 1.0
    return onehot @ table


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nearest_tokens(query: np.ndarray, table: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Cari token dengan arah vektor paling mirip (cosine)."""
    norms = np.linalg.norm(table, axis=1) * np.linalg.norm(query) + 1e-12
    sims = (table @ query) / norms
    order = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in order]


def parameter_count(vocab_size: int, d_model: int, tied: bool) -> dict:
    emb = vocab_size * d_model
    out = 0 if tied else vocab_size * d_model
    return {"embedding": emb, "unembedding": out, "total": emb + out}


def main() -> None:
    vocab_size, d_model = 1000, 64
    table = make_embedding(vocab_size, d_model)

    print("=== 1. Lookup == one-hot matmul ===")
    ids = [42, 7, 999]
    lookup = embed_lookup(ids, table)
    onehot = embed_onehot(ids, table)
    print(f"bentuk output lookup : {lookup.shape}")
    print(f"selisih maksimum     : {np.abs(lookup - onehot).max():.2e}")
    print(f"memori one-hot       : {len(ids) * vocab_size * 8 / 1024:.1f} KB")
    print(f"memori lookup        : {lookup.nbytes / 1024:.1f} KB")

    print("\n=== 2. Embedding hanya bermakna relatif ===")
    q = table[42]
    print("tetangga token 42 (cosine similarity):")
    for idx, sim in nearest_tokens(q, table, k=5):
        marker = "  <- dirinya sendiri" if idx == 42 else ""
        print(f"  token {idx:>4}: {sim:+.3f}{marker}")
    print("catatan: dengan embedding acak, kemiripan tertinggi hampir selalu dirinya")
    print("sendiri (sim=1.000). Kemiripan semantik baru muncul setelah training.")

    print("\n=== 3. Arah vs magnitudo ===")
    a, b = table[10], table[11]
    print(f"cosine(a, 2a)      : {cosine_similarity(a, 2 * a):.3f}  (arah sama)")
    print(f"cosine(a, -a)      : {cosine_similarity(a, -a):.3f}  (arah berlawanan)")
    print(f"cosine(a, b)       : {cosine_similarity(a, b):.3f}  (acak)")
    print("LayerNorm memakai arah, bukan magnitudo; karena itu cosine lebih informatif.")

    print("\n=== 4. Weight tying ===")
    for name, tied in [("tied", True), ("untied", False)]:
        c = parameter_count(vocab_size, d_model, tied)
        print(f"{name:<7} embedding={c['embedding']:>8,}  unembedding={c['unembedding']:>8,}  "
              f"total={c['total']:>8,}")
    print(f"penghematan tying: {vocab_size * d_model:,} parameter "
          f"({vocab_size * d_model * 2 / 1024:.1f} KB di fp16)")

    print("\n=== 5. Skala di model nyata ===")
    for name, v, d, tied in [
        ("GPT-2 small", 50_257, 768, True),
        ("LLaMA-7B", 32_000, 4_096, False),
        ("Qwen2.5-7B", 151_936, 3_584, False),
    ]:
        c = parameter_count(v, d, tied)
        print(f"{name:<12} vocab={v:>7,} d={d:>5,} -> embedding {c['embedding'] / 1e6:>7.1f}M "
              f"({c['embedding'] / 1e9:.2f}B param)")
    print("untuk vocab besar, tabel embedding saja bisa >1 miliar parameter.")


if __name__ == "__main__":
    main()
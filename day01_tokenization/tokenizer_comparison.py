"""Bandingkan strategi tokenisasi: karakter, kata, BPE, dan tiktoken.

Poin pembelajaran:
- Token bukan karakter dan bukan kata; token adalah potongan teks yang
  frekuensinya dipelajari dari data.
- Jumlah token menentukan biaya API dan seberapa cepat context window penuh.
- Tokenizer yang berbeda memotong teks yang sama dengan cara berbeda, sehingga
  "panjang prompt" hanya bermakna kalau tokenizer-nya disebutkan.

Jalankan: python day01_tokenization/tokenizer_comparison.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from bpe import BPE  # noqa: E402

SENTENCES = [
    "Halo dunia",
    "Tokenisasi bukan sekadar memecah teks pada spasi.",
    "LLM mempelajari pola dari data, bukan aturan tata bahasa.",
    "def attention(q, k, v): return softmax(q @ k.T) @ v",
    "Mempertanggungjawabkan",
    "2026-11-17T09:00:00Z",
    "    kode menjorok dengan     spasi aneh   ",
]


def char_tokens(text: str) -> list[str]:
    """Baseline paling naif: satu karakter = satu token.

    Spasi ikut dihitung. Ini penting: tokenizer nyata juga menyimpan spasi
    (biasanya digabung ke token berikutnya, mis. ``" the"``).
    """
    return list(text)


def word_tokens(text: str) -> list[str]:
    """Baseline kata: split spasi. Kelemahannya: kosakata tak terbatas."""
    return text.split()


def get_tiktoken_encoding(name: str = "cl100k_base"):
    """Ambil encoding tiktoken. Mengembalikan None kalau file belum ter-cache
    dan mesin sedang offline, supaya script tetap jalan."""
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as exc:
        print(f"[info] tiktoken tidak tersedia: {type(exc).__name__}: {exc}\n")
        return None


def chars_per_token(n_chars: int, n_tokens: int) -> float:
    """Karakter per token dari angka mentah. 0 kalau tidak ada karakter."""
    if n_chars <= 0:
        return 0.0
    return n_chars / max(n_tokens, 1)


def compression_ratio(text: str, n_tokens: int) -> float:
    """Karakter per token. Semakin besar, semakin efisien tokenizer-nya."""
    return chars_per_token(len(text), n_tokens)


def main() -> None:
    with open(os.path.join(ROOT, "data", "sample_corpus.txt"), encoding="utf-8") as handle:
        corpus = handle.read()
    bpe = BPE.train(corpus, num_merges=40)
    enc = get_tiktoken_encoding()

    print("=== Perbandingan tokenizer ===")
    print(f"BPE lokal: {bpe.vocab_size} token vocabulary, 40 merge rules")
    if enc is not None:
        print(f"tiktoken : {enc.n_vocab:,} token vocabulary (cl100k_base)")
    print("catatan  : BPE lokal dilatih pada corpus huruf kecil, jadi karakter")
    print("           di luar vocabulary (huruf kapital, emoji) dibuang.")
    print("           tiktoken memakai byte-level fallback sehingga tidak ada")
    print("           teks yang hilang, hanya dipecah lebih banyak token.")
    print()

    header = f"{'teks':<44} {'char':>5} {'word':>5} {'bpe':>5} {'tiktok':>7}"
    print(header)
    print("-" * len(header))

    totals = {"char": 0, "word": 0, "bpe": 0, "tiktoken": 0}
    for text in SENTENCES:
        n_char = len(char_tokens(text))
        n_word = len(word_tokens(text))
        n_bpe = len(bpe.encode(text, strict=False))
        n_tk = len(enc.encode(text)) if enc is not None else 0

        totals["char"] += n_char
        totals["word"] += n_word
        totals["bpe"] += n_bpe
        totals["tiktoken"] += n_tk

        label = text if len(text) <= 42 else text[:39] + "..."
        tk_cell = f"{n_tk:>7}" if enc is not None else f"{'n/a':>7}"
        print(f"{label:<44} {n_char:>5} {n_word:>5} {n_bpe:>5} {tk_cell}")

    print("-" * len(header))
    tk_total = f"{totals['tiktoken']:>7}" if enc is not None else f"{'n/a':>7}"
    print(f"{'TOTAL':<44} {totals['char']:>5} {totals['word']:>5} "
          f"{totals['bpe']:>5} {tk_total}")

    # denominator harus karakter yang sama dengan yang di-tokenize
    total_chars = sum(len(text) for text in SENTENCES)
    print(f"\nRasio kompresi BPE      : {chars_per_token(total_chars, totals['bpe']):.2f} char/token")
    if enc is not None:
        print(f"Rasio kompresi tiktoken : {chars_per_token(total_chars, totals['tiktoken']):.2f} char/token")

    # Kasus yang paling menjelaskan perbedaan: kata panjang & kode
    print("\n=== Di mana tokenizer berbeda? ===")
    for text in ["Mempertanggungjawabkan", "def attention(q, k, v): return softmax(q @ k.T) @ v"]:
        print(f"\nteks: {text}")
        if enc is not None:
            ids = enc.encode(text)
            print(f"  tiktoken ({len(ids)}): {[enc.decode([i]) for i in ids]}")
        bpe_ids = bpe.encode(text, strict=False)
        print(f"  BPE lokal ({len(bpe_ids)}): {[bpe.itos[i] for i in bpe_ids]}")

    # Kalimat yang sama isinya, dibandingkan dalam dua bahasa. Tokenizer
    # berbahasa Inggris hampir selalu lebih boros untuk teks Indonesia.
    id_text = "Model bahasa mempelajari pola dari data."
    en_text = "The language model learns patterns from data."
    print("\n=== Harga teks Indonesia di tokenizer berbahasa Inggris ===")
    print(f"indonesia : {id_text!r} ({len(id_text)} karakter)")
    print(f"inggris   : {en_text!r} ({len(en_text)} karakter)")
    print(f"BPE lokal : {len(bpe.encode(id_text, strict=False))} token indonesia, "
          f"{len(bpe.encode(en_text, strict=False))} token inggris "
          f"(huruf kapital dibuang, corpus latihan huruf kecil)")
    if enc is not None:
        id_ids = enc.encode(id_text)
        en_ids = enc.encode(en_text)
        print(f"tiktoken  : {len(id_ids)} token indonesia "
              f"({len(id_text) / len(id_ids):.2f} char/token), "
              f"{len(en_ids)} token inggris ({len(en_text) / len(en_ids):.2f} char/token)")
        print(f"potongan indonesia: {[enc.decode([i]) for i in id_ids]}")


if __name__ == "__main__":
    main()
"""Byte-Pair Encoding (BPE) from scratch.

BPE adalah algoritma kompresi yang dipakai hampir semua tokenizer LLM modern.
Idenya sederhana:

1. Mulai dari level karakter (setiap karakter = satu token).
2. Hitung pasangan simbol yang paling sering muncul bersebelahan.
3. Gabungkan pasangan itu menjadi satu simbol baru (merge rule).
4. Ulangi langkah 2-3 sebanyak ``num_merges`` kali.

Hasilnya adalah daftar *merge rules* berurutan. Untuk meng-encode teks baru,
kita menerapkan merge rules yang sama dengan urutan yang sama. Aturan yang
dipelajari lebih awal punya prioritas lebih tinggi.

Catatan: implementasi ini murni Python + Counter, tanpa dependensi eksternal,
supaya mekanismenya terlihat jelas. Dua penyederhanaan yang disengaja:

- split berdasarkan spasi (tokenizer produksi memakai regex pre-tokenization,
  sehingga "token." dan " token" bisa menjadi unit berbeda),
- tidak ada byte-level fallback: karakter baru yang tidak pernah dilihat saat
  training akan ditolak (``strict=True``) atau dibuang (``strict=False``).
"""

import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END_OF_WORD = "</w>"  # penanda batas kata, seperti pada BPE level kata


def word_frequencies(text: str) -> Counter:
    """Hitung frekuensi setiap kata. Split sederhana berdasarkan spasi."""
    return Counter(text.split())


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Gabung semua kemunculan ``pair`` yang tidak saling tumpang tindih.

    Contoh: ('l','o','w') + ('l','o') -> ('lo','w')
    """
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPE:
    """Tokenizer BPE minimal: hasil training + encode + decode."""

    def __init__(self, merges: list[tuple[str, str]], alphabet: list[str]):
        self.merges = merges
        # vocabulary = simbol dasar (karakter) + hasil setiap merge
        tokens = list(alphabet) + [a + b for a, b in merges]
        self.itos = tokens
        self.stoi = {tok: i for i, tok in enumerate(tokens)}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, text: str, num_merges: int, verbose: bool = False) -> "BPE":
        """Latih merge rules dari sebuah corpus teks."""
        if not isinstance(num_merges, int) or num_merges < 0:
            raise ValueError("num_merges harus bilangan bulat >= 0")
        if not text.strip():
            raise ValueError("corpus kosong; tidak ada token untuk dilatih")

        words: Counter = Counter()
        for word, freq in word_frequencies(text).items():
            words[tuple(word) + (END_OF_WORD,)] += freq

        # urutan alfabet harus deterministik agar hasil reproducible
        alphabet = sorted({s for w in words for s in w})

        merges: list[tuple[str, str]] = []
        for step in range(num_merges):
            pairs: Counter = Counter()
            for symbols, freq in words.items():
                for a, b in zip(symbols, symbols[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            # tie-break pakai pasangan itu sendiri supaya deterministik
            best, freq = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            merges.append(best)
            words = Counter({merge_pair(w, best): c for w, c in words.items()})
            if verbose:
                print(f"  merge {step + 1:2d}: {best[0]!r} + {best[1]!r} "
                      f"(freq={freq}) -> {best[0] + best[1]!r}")

        return cls(merges, alphabet)

    # ----------------------------------------------------------------- encode
    def encode(self, text: str, strict: bool = True) -> list[int]:
        """Encode teks menjadi id.

        ``strict=True`` melempar ValueError kalau ada karakter yang tidak
        pernah dilihat saat training. Ini lebih baik daripada membuang karakter
        diam-diam: teks yang hilang tanpa peringatan adalah bug yang sulit
        dilacak. Tokenizer produksi menanganinya dengan byte-level fallback.
        """
        ids: list[int] = []
        unknown: set[str] = set()
        for word in text.split():
            symbols = tuple(word) + (END_OF_WORD,)
            for pair in self.merges:  # urutan merge = prioritas
                symbols = merge_pair(symbols, pair)
            for symbol in symbols:
                if symbol in self.stoi:
                    ids.append(self.stoi[symbol])
                else:
                    unknown.add(symbol)
        if unknown and strict:
            # ascii() supaya pesan tetap bisa dicetak di console non-UTF8
            sample = ascii("".join(sorted(unknown))[:10])
            raise ValueError(
                f"karakter tidak ada di vocabulary: {sample}. "
                "Latih ulang dengan corpus yang lebih kaya atau pakai strict=False."
            )
        return ids

    # ----------------------------------------------------------------- decode
    def decode(self, ids: list[int]) -> str:
        text = "".join(self.itos[i] for i in ids)
        # setiap END_OF_WORD berarti satu spasi di antara kata
        return text.replace(END_OF_WORD, " ").strip()

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def main() -> None:
    with open(os.path.join(ROOT, "data", "sample_corpus.txt"), encoding="utf-8") as handle:
        corpus = handle.read()

    print("=== BPE from scratch ===")
    print(f"karakter corpus : {len(corpus):,}")
    print(f"kata unik       : {len(word_frequencies(corpus)):,}")

    model = BPE.train(corpus, num_merges=40, verbose=True)
    print(f"\nvocabulary size : {model.vocab_size}")

    sample = "token bukan kata"
    ids = model.encode(sample)
    print(f"\nteks      : {sample!r}")
    print(f"token     : {[model.itos[i] for i in ids]}")
    print(f"ids       : {ids}")
    print(f"roundtrip : {model.decode(ids)!r}  (sama: {model.decode(ids) == sample})")

    # rasio kompresi: karakter -> token
    n_char = len(sample)
    n_tok = len(ids)
    print(f"\nkompresi  : {n_char} karakter -> {n_tok} token "
          f"({n_char / n_tok:.2f} karakter/token)")

    print("\n=== Karakter yang tidak dikenal ===")
    unseen = "token \U0001F600"  # emoji tidak ada di corpus
    try:
        model.encode(unseen)
    except ValueError as exc:
        print(f"strict=True  -> ValueError: {exc}")
    lenient = model.encode(unseen, strict=False)
    print(f"strict=False -> {len(lenient)} id, karakter asing dibuang diam-diam")
    print("produksi memakai byte-level fallback supaya tidak ada teks yang hilang.")

    # bandingkan dengan tokenizer produksi kalau tersedia (butuh unduhan sekali)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tk = enc.encode(sample)
        print(f"tiktoken  : {len(tk)} token -> {[enc.decode([t]) for t in tk]}")
    except Exception as exc:  # offline / file encoding belum ter-cache
        print(f"tiktoken  : dilewati ({type(exc).__name__})")


if __name__ == "__main__":
    main()
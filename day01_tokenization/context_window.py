"""Context window: batas token yang bisa dibaca model dalam satu waktu.

Fakta penting:
- Context window adalah batas keras. Melebihi batas -> API error atau prompt
  dipotong diam-diam (tergantung provider).
- Biaya attention tumbuh kuadratik terhadap panjang sequence, sedangkan biaya
  KV cache tumbuh linear. Menambah konteks 2x berarti attention 4x lebih mahal.
- Kalau pesan tidak muat, strategi pemotongan menentukan informasi apa yang
  hilang. "Lost in the middle": model cenderung lebih baik mengingat awal dan
  akhir konteks daripada bagian tengah.

Akuntansi token di file ini:
    tokens(pesan) = estimate_tokens(isi) + MESSAGE_OVERHEAD
    budget efektif = budget - reserve_tokens

``reserve_tokens`` dipakai untuk menyisakan ruang bagi output. Tanpa reserve,
prompt yang "pas" akan langsung kehabisan tempat untuk jawaban.

Jalankan: python day01_tokenization/context_window.py
"""

from __future__ import annotations

MESSAGE_OVERHEAD = 4  # token untuk penanda role + newline + separator


def estimate_tokens(text: str) -> int:
    """Heuristik murah: ~4 karakter per token untuk teks Latin.

    String kosong = 0 token. Untuk angka pasti pakai tokenizer model; heuristik
    ini hanya untuk memperlihatkan mekanisme budget.
    """
    if not text.strip():
        return 0
    return max(1, round(len(text) / 4))


def message_tokens(message: dict) -> int:
    return estimate_tokens(message["content"]) + MESSAGE_OVERHEAD


def total_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def _truncate_contents(kept: list[dict], available: int) -> list[dict]:
    """Paksa ``kept`` muat dengan memotong isi terpanjang lebih dulu.

    Ini jaring pengaman untuk semua strategi: apa pun yang terjadi, hasil akhir
    tidak boleh melebihi ``available``. Kalau semua isi sudah kosong tetapi
    overhead per pesan masih terlalu besar, pesan dibuang satu per satu
    (system duluan dipertahankan selama mungkin).
    """
    while kept and total_tokens(kept) > available:
        longest = max(kept, key=lambda m: len(m["content"]))
        excess_tokens = total_tokens(kept) - available
        cut_chars = max(4, excess_tokens * 4)
        content = longest["content"]
        longest["content"] = (
            content[:-cut_chars].rstrip() if len(content) > cut_chars else ""
        )

        if not any(m["content"] for m in kept):
            # tidak ada teks tersisa yang bisa dipotong -> buang pesan
            index = next(
                (i for i, m in enumerate(kept) if m["role"] != "system"),
                0,  # kalau hanya system, buang system-nya juga
            )
            kept.pop(index)
    return kept


def fit_to_budget(
    messages: list[dict],
    budget: int,
    strategy: str,
    reserve_tokens: int = 0,
) -> dict:
    """Potong percakapan agar muat dalam ``budget`` token (termasuk reserve).

    Strategi:
    - ``drop_oldest``     : buang pesan paling lama, termasuk system prompt.
    - ``keep_system``     : selalu pertahankan system prompt, buang yang lama.
    - ``head_tail``       : pertahankan awal percakapan + pesan terbaru.
    - ``truncate_content``: pertahankan semua pesan, potong isi yang terpanjang.

    Semua strategi berakhir dengan pemotongan isi jika masih melebihi budget,
    sehingga ``fits`` selalu True (kecuali ``budget`` negatif).
    """
    if budget <= 0:
        raise ValueError("budget harus > 0")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens tidak boleh negatif")
    if reserve_tokens >= budget:
        raise ValueError("reserve_tokens harus lebih kecil dari budget")

    available = budget - reserve_tokens
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]

    if strategy == "drop_oldest":
        kept = [dict(m) for m in messages]
        while len(kept) > 1 and total_tokens(kept) > available:
            kept.pop(0)

    elif strategy == "keep_system":
        kept = [dict(m) for m in system + rest]
        while len(kept) > len(system) and total_tokens(kept) > available:
            kept.pop(len(system))

    elif strategy == "head_tail":
        if total_tokens(system + rest) <= available:
            kept = [dict(m) for m in system + rest]
        else:
            head = [dict(m) for m in rest[:2]]
            kept = [dict(m) for m in system] + head
            tail: list[dict] = []
            for msg in reversed(rest[2:]):
                candidate = system + head + [msg] + list(reversed(tail))
                if total_tokens(candidate) > available:
                    break
                tail.append(dict(msg))
            kept = [dict(m) for m in system] + head + list(reversed(tail))

    elif strategy == "truncate_content":
        kept = [dict(m) for m in messages]

    else:
        raise ValueError(f"strategi tidak dikenal: {strategy}")

    kept = _truncate_contents(kept, available)

    dropped = len(messages) - len(kept)
    used = total_tokens(kept)
    return {
        "kept": kept,
        "dropped_messages": dropped,
        "tokens": used,
        "utilization": used / available,
        "fits": used <= available,
        "reserve_tokens": reserve_tokens,
    }


def attention_cost(n_tokens: int) -> int:
    """Jumlah pasangan token yang dihitung attention (causal: n*(n+1)/2)."""
    if n_tokens < 0:
        raise ValueError("n_tokens tidak boleh negatif")
    return n_tokens * (n_tokens + 1) // 2


def main() -> None:
    conversation = [
        {"role": "system", "content": "Kamu adalah tutor LLM. Jawab ringkas dan akurat."},
        {"role": "user", "content": "Ingat angka 4271. Angka ini penting untuk pertanyaan terakhir."},
        {"role": "assistant", "content": "Baik, saya ingat angka 4271."},
    ]
    for i in range(12):
        conversation.append({"role": "user", "content": f"Pertanyaan latihan nomor {i + 1} tentang attention dan tokenizer. " * 3})
        conversation.append({"role": "assistant", "content": f"Jawaban latihan nomor {i + 1}: perhatikan detail nomor {i + 1} agar konsisten. " * 3})
    conversation.append({"role": "user", "content": "Berapa angka yang saya minta diingat di awal?"})

    budget = 220
    reserve = 40  # sisakan ruang untuk jawaban model
    print("=== 1. Apakah percakapan muat? ===")
    print(f"jumlah pesan    : {len(conversation)}")
    print(f"total token     : {total_tokens(conversation)}")
    print(f"context budget  : {budget}")
    print(f"reserve output  : {reserve}")
    print(f"tersedia prompt : {budget - reserve}")
    print(f"status          : {'MUAT' if total_tokens(conversation) <= budget - reserve else 'OVERFLOW'}")

    print("\n=== 2. Perbandingan strategi pemotongan ===")
    header = f"{'strategi':<18} {'pesan':>6} {'dibuang':>8} {'token':>6} {'util':>6} {'ingat 4271?':>12}"
    print(header)
    print("-" * len(header))
    for strategy in ["drop_oldest", "keep_system", "head_tail", "truncate_content"]:
        result = fit_to_budget(conversation, budget, strategy, reserve_tokens=reserve)
        text = " ".join(m["content"] for m in result["kept"])
        remembers = "YA" if "4271" in text else "TIDAK"
        print(f"{strategy:<18} {len(result['kept']):>6} {result['dropped_messages']:>8} "
              f"{result['tokens']:>6} {result['utilization']:>5.0%} {remembers:>12}")

    print("\n=== 3. Isi setelah pemotongan (keep_system) ===")
    kept = fit_to_budget(conversation, budget, "keep_system", reserve_tokens=reserve)["kept"]
    print(f"pesan pertama yang tersisa: [{kept[0]['role']}] {kept[0]['content'][:60]!r}...")
    print(f"pesan terakhir            : [{kept[-1]['role']}] {kept[-1]['content'][:60]!r}...")

    print("\n=== 4. Budget ekstrem tidak boleh crash ===")
    for tiny_budget in [1, 5, MESSAGE_OVERHEAD, 12]:
        result = fit_to_budget(conversation, tiny_budget, "keep_system")
        print(f"budget={tiny_budget:>2}: pesan={len(result['kept'])}, "
              f"token={result['tokens']}, fits={result['fits']}")

    print("\n=== 5. Biaya attention tumbuh kuadratik ===")
    print(f"{'context':>8} {'pasangan attention':>20} {'relatif 1k':>12}")
    base = attention_cost(1000)
    for n in [1_000, 4_000, 8_000, 32_000, 128_000]:
        print(f"{n:>8} {attention_cost(n):>20,} {attention_cost(n) / base:>11.0f}x")
    print("\n4x konteks = ~16x pekerjaan attention. 128x konteks = ~16.000x pekerjaan.")

    print("\n=== 6. 'Lost in the middle' (ilustrasi posisi) ===")
    print("riset menunjukkan akurasi retrieval informasi berbentuk U: tinggi di awal,")
    print("turun di tengah, tinggi lagi di akhir. Karena itu instruksi penting")
    print("sebaiknya diletakkan di awal (system prompt) DAN diulang di akhir prompt.")


if __name__ == "__main__":
    main()
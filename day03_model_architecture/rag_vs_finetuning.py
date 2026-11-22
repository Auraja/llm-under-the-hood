"""RAG vs fine-tuning: kapan menambah konteks, kapan mengubah bobot.

Dua cara memasukkan pengetahuan ke sistem LLM:

1. RAG (Retrieval-Augmented Generation)
   - Dokumen diambil saat request, dimasukkan ke prompt.
   - Bobot model tidak berubah.
   - Biaya per request = token dokumen tambahan + retrieval.
   - Data bisa berubah dalam hitungan detik.

2. Fine-tuning (SFT / LoRA)
   - Bobot diubah dengan data contoh.
   - Biaya sekali di depan (training), lalu request lebih murah (prompt pendek).
   - Pengetahuan baru tidak selalu terserap andal; lebih cocok untuk gaya,
     format, dan pola jawaban.

File ini menghitung titik impas biaya secara eksplisit agar keputusan tidak
hanya berdasarkan intuisi.

Jalankan: python day03_model_architecture/rag_vs_finetuning.py
"""

from __future__ import annotations

# ilustrasi USD per 1 juta token
PRICE_IN = 0.40
PRICE_OUT = 1.20
TRAIN_PRICE_PER_MTOK = 3.00  # ilustrasi biaya training per 1M token


def llm_cost(prompt_tokens: int, completion_tokens: int,
             price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> float:
    return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out


def rag_cost(doc_tokens: int, question_tokens: int, answer_tokens: int,
             retrieval_tokens: int = 120) -> float:
    """Biaya satu request RAG: prompt lebih panjang karena dokumen."""
    prompt = doc_tokens + retrieval_tokens + question_tokens
    return llm_cost(prompt, answer_tokens)


def baseline_cost(question_tokens: int, answer_tokens: int) -> float:
    return llm_cost(question_tokens, answer_tokens)


def finetune_cost(train_tokens: int, epochs: int = 3,
                  price_per_mtok: float = TRAIN_PRICE_PER_MTOK,
                  human_hours: float = 16, hourly_rate: float = 15) -> dict:
    compute = train_tokens * epochs / 1e6 * price_per_mtok
    human = human_hours * hourly_rate
    eval_cost = compute * 0.2  # evaluasi & iterasi
    return {"compute": compute, "human": human, "eval": eval_cost,
            "total": compute + human + eval_cost}


def break_even_requests(fixed_cost: float, extra_per_request: float) -> float:
    if extra_per_request <= 0:
        return float("inf")
    return fixed_cost / extra_per_request


def latency_ms(prompt_tokens: int, output_tokens: int, retrieval_ms: float = 0,
               prefill_tokens_per_s: float = 2_000, decode_tokens_per_s: float = 80) -> dict:
    prefill = prompt_tokens / prefill_tokens_per_s * 1000
    decode = output_tokens / decode_tokens_per_s * 1000
    return {"retrieval": retrieval_ms, "prefill": prefill, "decode": decode,
            "total": retrieval_ms + prefill + decode}


SCENARIOS = {
    "FAQ produk statis": dict(doc_tokens=1_500, question_tokens=40, answer_tokens=220,
                              train_tokens=2_000_000, changes_per_month=0),
    "Gaya brand + format ketat": dict(doc_tokens=0, question_tokens=40, answer_tokens=220,
                                      train_tokens=5_000_000, changes_per_month=0),
    "Harga & promo harian": dict(doc_tokens=2_500, question_tokens=40, answer_tokens=220,
                                 train_tokens=2_000_000, changes_per_month=30),
    "Dokumen legal 200 halaman": dict(doc_tokens=8_000, question_tokens=40, answer_tokens=300,
                                      train_tokens=10_000_000, changes_per_month=1),
}


def main() -> None:
    print("=== 1. Biaya per request ===")
    print(f"harga: input ${PRICE_IN}/1M, output ${PRICE_OUT}/1M")
    print(f"{'skenario':<26} {'tanpa RAG $':>12} {'RAG $':>10} {'selisih $':>11}")
    for name, s in SCENARIOS.items():
        base = baseline_cost(s["question_tokens"], s["answer_tokens"])
        rag = rag_cost(s["doc_tokens"], s["question_tokens"], s["answer_tokens"])
        print(f"{name:<26} {base:>12.6f} {rag:>10.6f} {rag - base:>11.6f}")

    print("\n=== 2. Titik impas fine-tuning ===")
    print(f"{'skenario':<26} {'FT total $':>11} {'RAG extra $':>12} {'impas (request)':>16}")
    for name, s in SCENARIOS.items():
        ft = finetune_cost(s["train_tokens"])["total"]
        base = baseline_cost(s["question_tokens"], s["answer_tokens"])
        rag = rag_cost(s["doc_tokens"], s["question_tokens"], s["answer_tokens"])
        extra = rag - base
        be = break_even_requests(ft, extra)
        label = "tidak pernah" if be == float("inf") else f"{be:,.0f}"
        print(f"{name:<26} {ft:>11,.2f} {extra:>12.6f} {label:>16}")

    print("\ninterpretasi: kalau skenario 'FAQ produk statis' hanya menerima")
    print("5.000 request/bulan, RAG tetap lebih murah daripada fine-tuning.")
    print("fine-tuning baru menang kalau volume sangat besar DAN pengetahuan stabil.")

    print("\n=== 3. Latensi ===")
    s = SCENARIOS["Dokumen legal 200 halaman"]
    rag_lat = latency_ms(s["doc_tokens"] + s["question_tokens"], s["answer_tokens"], retrieval_ms=120)
    ft_lat = latency_ms(s["question_tokens"], s["answer_tokens"])
    print(f"{'komponen':<12} {'RAG (ms)':>10} {'FT (ms)':>10}")
    for key in ["retrieval", "prefill", "decode", "total"]:
        print(f"{key:<12} {rag_lat[key]:>10.1f} {ft_lat[key]:>10.1f}")
    print("\nRAG menambah retrieval + prefill. FT hanya menghemat prompt,")
    print("tetapi tidak bisa memperbarui fakta tanpa training ulang.")

    print("\n=== 4. Matriks keputusan (skor 1-5, bobot) ===")
    criteria = {
        "kesegaran data": 0.25,
        "biaya per request": 0.20,
        "kualitas gaya/format": 0.20,
        "waktu implementasi": 0.20,
        "kebutuhan audit/sumber": 0.15,
    }
    scores = {
        "RAG": {"kesegaran data": 5, "biaya per request": 2, "kualitas gaya/format": 3,
                "waktu implementasi": 4, "kebutuhan audit/sumber": 5},
        "LoRA/SFT": {"kesegaran data": 1, "biaya per request": 5, "kualitas gaya/format": 5,
                     "waktu implementasi": 2, "kebutuhan audit/sumber": 2},
        "RAG + LoRA": {"kesegaran data": 5, "biaya per request": 3, "kualitas gaya/format": 5,
                       "waktu implementasi": 1, "kebutuhan audit/sumber": 5},
    }
    print(f"{'kriteria':<24} {'bobot':>6} " + " ".join(f"{k:>11}" for k in scores))
    for criterion, weight in criteria.items():
        row = " ".join(f"{scores[k][criterion] * weight:>11.2f}" for k in scores)
        print(f"{criterion:<24} {weight:>6.2f} {row}")
    totals = {k: sum(scores[k][c] * w for c, w in criteria.items()) for k in scores}
    print("-" * 70)
    print(f"{'TOTAL':<24} {'1.00':>6} " + " ".join(f"{totals[k]:>11.2f}" for k in scores))
    best = max(totals, key=totals.get)
    print(f"\nrekomendasi berdasarkan bobot di atas: {best}")

    print("\n=== 5. Aturan praktis ===")
    print("- Data berubah cepat, butuh sitasi  -> RAG.")
    print("- Gaya dan format harus konsisten   -> LoRA/SFT.")
    print("- Keduanya penting                   -> RAG untuk fakta, LoRA untuk gaya.")
    print("- Jangan fine-tune untuk fakta yang bisa berubah besok.")


if __name__ == "__main__":
    main()
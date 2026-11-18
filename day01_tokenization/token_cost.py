"""Token cost: menghitung biaya API dari jumlah token.

Biaya LLM API biasanya dihitung per 1 juta token, dipisah menjadi:
- input / prompt token  (lebih murah)
- output / completion token (lebih mahal, karena decoding berjalan token demi token)
- cached input token    (paling murah, kalau provider mendukung prompt caching)

Harga di bawah adalah ILUSTRASI untuk latihan (USD per 1 juta token, region
Singapore/internasional). Selalu cek halaman harga provider sebelum memakai
angka ini untuk keputusan nyata.

Jalankan: python day01_tokenization/token_cost.py
"""

from __future__ import annotations

# (input $/1M, output $/1M) - ilustrasi, bukan harga resmi terbaru
PRICES: dict[str, tuple[float, float]] = {
    "qwen-turbo": (0.05, 0.20),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
}

CACHE_INPUT_DISCOUNT = 0.20  # cached input ~20% dari harga input normal
IDR_PER_USD = 16_300         # asumsi kurs ilustratif, bukan kurs resmi


def _validate_tokens(**counts: int) -> None:
    for name, value in counts.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} harus int, bukan {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} tidak boleh negatif: {value}")


def _price(model: str) -> tuple[float, float]:
    if model not in PRICES:
        raise ValueError(f"model tidak dikenal: {model!r}. Pilihan: {sorted(PRICES)}")
    return PRICES[model]


def cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> dict:
    """Biaya satu request, dipecah per komponen."""
    _validate_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    price_in, price_out = _price(model)
    input_cost = prompt_tokens / 1_000_000 * price_in
    output_cost = completion_tokens / 1_000_000 * price_out
    total = input_cost + output_cost
    return {
        "input": input_cost,
        "output": output_cost,
        "total": total,
        "avg_per_token": total / max(prompt_tokens + completion_tokens, 1),
    }


def cost_with_cache(prompt_tokens: int, cached_tokens: int, completion_tokens: int, model: str) -> float:
    """Biaya kalau sebagian prompt diambil dari prompt cache."""
    _validate_tokens(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
                     completion_tokens=completion_tokens)
    if cached_tokens > prompt_tokens:
        raise ValueError("cached_tokens tidak boleh melebihi prompt_tokens")
    price_in, _ = _price(model)
    fresh = prompt_tokens - cached_tokens
    input_cost = (fresh * price_in + cached_tokens * price_in * CACHE_INPUT_DISCOUNT) / 1_000_000
    return input_cost + cost_usd(0, completion_tokens, model)["output"]


def monthly_cost(requests_per_day: int, prompt_tokens: int, completion_tokens: int, model: str) -> float:
    _validate_tokens(requests_per_day=requests_per_day)
    return requests_per_day * 30 * cost_usd(prompt_tokens, completion_tokens, model)["total"]


def main() -> None:
    print("=== 1. Harga ilustrasi (USD / 1 juta token) ===")
    print(f"{'model':<12} {'input':>8} {'output':>8} {'output/input':>13}")
    for name, (p_in, p_out) in PRICES.items():
        print(f"{name:<12} {p_in:>8.2f} {p_out:>8.2f} {p_out / p_in:>12.1f}x")

    print("\n=== 2. Biaya satu request ===")
    prompt_tokens, completion_tokens = 1_200, 300
    print(f"prompt {prompt_tokens} token, completion {completion_tokens} token")
    print(f"{'model':<12} {'input $':>10} {'output $':>10} {'total $':>10} {'Rp':>12}")
    for model in PRICES:
        c = cost_usd(prompt_tokens, completion_tokens, model)
        print(f"{model:<12} {c['input']:>10.6f} {c['output']:>10.6f} "
              f"{c['total']:>10.6f} {c['total'] * IDR_PER_USD:>12,.2f}")

    print("\n=== 3. Prompt panjang (RAG) mendominasi input ===")
    scenarios = [
        ("chat biasa", 800, 250),
        ("RAG 3 dokumen", 6_000, 400),
        ("RAG + riwayat panjang", 24_000, 500),
        ("agent dengan banyak tool call", 60_000, 800),
    ]
    model = "qwen-plus"
    print(f"model: {model}")
    print(f"{'skenario':<30} {'prompt':>8} {'out':>6} {'total $':>10} {'/bulan (5k req/hari) $':>24}")
    for name, p_tok, c_tok in scenarios:
        total = cost_usd(p_tok, c_tok, model)["total"]
        print(f"{name:<30} {p_tok:>8,} {c_tok:>6} {total:>10.5f} "
              f"{monthly_cost(5_000, p_tok, c_tok, model):>24,.2f}")

    print("\n=== 4. Efek prompt caching ===")
    p_tok, c_tok, cached = 6_000, 400, 5_500
    normal = cost_usd(p_tok, c_tok, model)["total"]
    cached_cost = cost_with_cache(p_tok, cached, c_tok, model)
    print(f"prompt {p_tok:,} token, {cached:,} di antaranya dari cache")
    print(f"tanpa cache : ${normal:.6f}")
    print(f"dengan cache: ${cached_cost:.6f}  (hemat {(1 - cached_cost / normal):.0%})")
    print(f"per bulan 150k request: hemat ${monthly_cost(5_000, p_tok, c_tok, model) - 30 * 5_000 * cached_cost:,.2f}")

    print("\n=== 5. Output token lebih mahal, tapi output lebih sedikit ===")
    print("RAG 24k prompt + 500 output -> didominasi input")
    print("Chat kreatif 300 prompt + 2.000 output -> didominasi output")
    a = cost_usd(24_000, 500, model)
    b = cost_usd(300, 2_000, model)
    print(f"  RAG        : input ${a['input']:.5f} ({a['input'] / a['total']:.0%}), "
          f"output ${a['output']:.5f} ({a['output'] / a['total']:.0%})")
    print(f"  Kreatif    : input ${b['input']:.5f} ({b['input'] / b['total']:.0%}), "
          f"output ${b['output']:.5f} ({b['output'] / b['total']:.0%})")

    print("\n=== 6. Tokenizer mempengaruhi biaya ===")
    print("teks yang sama, bahasa berbeda, jumlah token berbeda:")
    samples = {
        "Inggris": "The model predicts the next token from the previous tokens.",
        "Indonesia": "Model memprediksi token berikutnya dari token sebelumnya.",
        "Jawa": "Model ngrakitoken sabanjure saka token sadurunge.",
    }
    for lang, text in samples.items():
        est = round(len(text) / 4)  # heuristik kasar
        print(f"  {lang:<10} {len(text):>3} karakter ~ {est:>3} token "
              f"(~${est / 1_000_000 * PRICES[model][0]:.8f} per request)")


if __name__ == "__main__":
    main()
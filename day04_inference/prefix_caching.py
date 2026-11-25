"""Prefix caching (prompt caching): memakai ulang perhitungan prompt statis.

Ide: banyak request berbagi prefix yang sama persis, misalnya system prompt,
definisi tool, atau dokumen kebijakan. KV cache dari prefix itu tidak berubah,
jadi provider bisa menyimpannya dan melewati prefill.

Syarat agar cache kena:
- prefix harus identik byte-per-byte (termasuk spasi dan urutan pesan),
- prefix harus berada di AWAL prompt (bukan di tengah),
- belum kedaluwarsa (TTL biasanya menit sampai jam).

Keuntungan:
- biaya input turun (token cache biasanya 10-25% harga normal),
- TTFT turun karena prefill lebih pendek,
- throughput naik karena pekerjaan prefill dihindari.

Eksperimen ini menghitung penghematan untuk beberapa tingkat cache hit.

Jalankan: python day04_inference/prefix_caching.py
"""

from __future__ import annotations

PRICE_IN = 0.40          # USD / 1M token input
CACHE_PRICE_FACTOR = 0.20  # token cache ~20% harga input
PRICE_OUT = 1.20

PREFILL_TOKENS_PER_S = 2_000


def request_cost(prompt_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    fresh = prompt_tokens - cached_tokens
    if cached_tokens > prompt_tokens:
        raise ValueError("cached_tokens melebihi prompt_tokens")
    return (fresh * PRICE_IN + cached_tokens * PRICE_IN * CACHE_PRICE_FACTOR +
            output_tokens * PRICE_OUT) / 1e6


def prefill_latency_s(prompt_tokens: int, cached_tokens: int) -> float:
    return (prompt_tokens - cached_tokens) / PREFILL_TOKENS_PER_S


def monthly_savings(requests: int, prompt_tokens: int, cached_tokens: int,
                    output_tokens: int) -> dict:
    without = request_cost(prompt_tokens, 0, output_tokens) * requests
    with_cache = request_cost(prompt_tokens, cached_tokens, output_tokens) * requests
    return {"without": without, "with": with_cache, "saved": without - with_cache,
            "percent": (without - with_cache) / without if without else 0.0}


def cache_is_worth_it(prefix_tokens: int, requests_per_hour: int, storage_cost_per_gb_hour: float,
                      cache_bytes: int, ttl_hours: float) -> dict:
    """Perbandingan kasar: penghematan token vs biaya menyimpan cache."""
    saved_tokens = prefix_tokens * requests_per_hour * ttl_hours
    saved_usd = saved_tokens * PRICE_IN * (1 - CACHE_PRICE_FACTOR) / 1e6
    storage_usd = storage_cost_per_gb_hour * (cache_bytes / 1e9) * ttl_hours
    return {"saved_usd": saved_usd, "storage_usd": storage_usd,
            "worth_it": saved_usd > storage_usd}


def main() -> None:
    system_prompt = 1_800      # instruksi + persona
    tool_schema = 700          # definisi tool JSON
    document = 2_000           # dokumen referensi statis
    question = 120             # bagian yang berubah setiap request
    output = 300

    prefix = system_prompt + tool_schema + document
    prompt = prefix + question

    print("=== 1. Komposisi prompt ===")
    print(f"prefix statis : {prefix:,} token (system + tools + dokumen)")
    print(f"dynamic suffix: {question:,} token (pertanyaan user)")
    print(f"total prompt  : {prompt:,} token")
    print(f"porsi prefix  : {prefix / prompt:.1%} dari prompt")

    print("\n=== 2. Biaya per request ===")
    print(f"{'skenario':<20} {'input $':>10} {'output $':>10} {'total $':>10} {'TTFT ms':>9}")
    for label, cached in [("tanpa cache", 0), ("cache hit", prefix)]:
        cost = request_cost(prompt, cached, output)
        ttft = prefill_latency_s(prompt, cached) * 1000
        print(f"{label:<20} {cost - request_cost(0, 0, output):>10.6f} "
              f"{request_cost(0, 0, output):>10.6f} {cost:>10.6f} {ttft:>9.0f}")

    print("\n=== 3. Pengaruh cache hit rate ===")
    requests = 100_000  # per bulan
    print(f"{'hit rate':>9} {'cached token':>14} {'biaya/bulan $':>15} {'hemat $':>10} {'hemat %':>8}")
    for hit_rate in [0.0, 0.3, 0.5, 0.8, 0.95, 0.99]:
        cached = int(prefix * hit_rate)
        r = monthly_savings(requests, prompt, cached, output)
        print(f"{hit_rate:>8.0%} {cached:>14,} {r['with']:>15,.2f} {r['saved']:>10,.2f} "
              f"{r['percent']:>7.0%}")
    print("\npada 100k request/bulan, cache hit 80% memotong biaya input drastis.")
    print("output tetap dominan kalau jawabannya panjang.")

    print("\n=== 4. Ke mana cache 'hilang' ===")
    cases = [
        ("prefix diubah 1 karakter", "cache miss total (hash prefix berubah)"),
        ("system prompt di tengah", "tidak bisa di-cache (bukan prefix)"),
        ("timestamp di awal prompt", "cache miss setiap request"),
        ("urutan tool diacak", "cache miss walau isinya sama"),
        ("TTL habis", "prefill ulang penuh"),
        ("user A dan user B", "cache terpisah per prefix, aman secara isolasi"),
    ]
    print(f"{'penyebab':<28} {'efek':<45}")
    for cause, effect in cases:
        print(f"{cause:<28} {effect:<45}")

    print("\n=== 5. Pola prompt yang ramah cache ===")
    print("[static system prompt][static tools][static docs][dynamic user input]")
    print("taruh yang berubah paling akhir, dan jangan menyisipkan timestamp/acak")
    print("di bagian awal.")

    print("\n=== 6. Biaya menyimpan cache vs penghematan ===")
    # cache 2 layer contoh: 28 layer, 4 kv head, head_dim 128, fp16
    cache_bytes = 2 * 28 * 4 * 128 * prefix * 2
    r = cache_is_worth_it(prefix_tokens=prefix, requests_per_hour=50,
                          storage_cost_per_gb_hour=0.0001, cache_bytes=cache_bytes,
                          ttl_hours=1)
    print(f"ukuran KV cache prefix: {cache_bytes / 1e6:.1f} MB")
    print(f"penghematan 1 jam     : ${r['saved_usd']:.4f}")
    print(f"biaya storage 1 jam   : ${r['storage_usd']:.6f}")
    print(f"layak?                : {'YA' if r['worth_it'] else 'TIDAK'}")
    print("\nuntuk provider cloud, biaya storage cache biasanya jauh lebih kecil")
    print("daripada penghematan prefill; kendala utamanya adalah memori GPU.")


if __name__ == "__main__":
    main()
"""Estimasi memori inference: bobot + KV cache + activation + logits.

Inference bukan hanya "bobot dimuat". Setiap token yang sedang diproses
membutuhkan ruang untuk activation, dan setiap token yang sudah diproses
meninggalkan KV cache.

Rumus:
    weights    = n_params * bytes_per_weight
    kv_cache   = 2 * n_layers * n_kv_heads * head_dim * seq * batch * kv_bytes
    activations= batch * seq * d_model * bytes * factor
    logits     = batch * seq * vocab * bytes

``factor`` adalah perkiraan kasar jumlah tensor sementara per layer pada
inference (norm, proyeksi, FFN intermediate). Nilainya bergantung pada
implementasi; di sini dipakai 8 sebagai orde besaran.

Jalankan: python day04_inference/memory_estimator.py
"""

from __future__ import annotations

CONFIGS = {
    "Qwen2.5 0.5B": dict(n_params=494_000_000, n_layers=24, n_heads=14,
                         n_kv_heads=2, head_dim=64, d_model=896, vocab=151_936),
    "LLaMA-3 8B": dict(n_params=8_030_000_000, n_layers=32, n_heads=32,
                       n_kv_heads=8, head_dim=128, d_model=4096, vocab=128_256),
    "LLaMA-2 70B": dict(n_params=68_980_000_000, n_layers=80, n_heads=64,
                        n_kv_heads=8, head_dim=128, d_model=8192, vocab=32_000),
}


def estimate(cfg: dict, seq_len: int, batch: int = 1, weight_bytes: int = 2,
             kv_bytes: int = 2, activation_bytes: int = 2,
             activation_factor: int = 8) -> dict:
    weights = cfg["n_params"] * weight_bytes
    kv = (2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"]
          * seq_len * batch * kv_bytes)
    activations = batch * seq_len * cfg["d_model"] * activation_bytes * activation_factor
    logits = batch * seq_len * cfg["vocab"] * weight_bytes
    total = weights + kv + activations + logits
    return {
        "weights": weights,
        "kv_cache": kv,
        "activations": activations,
        "logits": logits,
        "total": total,
        "per_token_kv": kv / max(seq_len * batch, 1),
    }


def max_batch(cfg: dict, seq_len: int, budget_gb: float, **kwargs) -> int:
    """Batch maksimum yang muat, dengan pembulatan ke bawah."""
    one = estimate(cfg, seq_len, batch=1, **kwargs)["total"]
    weights = cfg["n_params"] * kwargs.get("weight_bytes", 2)
    # bobot hanya dihitung sekali; sisa budget dibagi per sequence
    per_seq = one - weights
    remaining = budget_gb * 1e9 - weights
    if per_seq <= 0 or remaining <= 0:
        return 0
    return int(remaining // per_seq)


def gb(value: float) -> float:
    return value / 1e9


def main() -> None:
    cfg = CONFIGS["LLaMA-3 8B"]

    print("=== 1. Rincian memori (LLaMA-3 8B, fp16, batch 1) ===")
    print(f"{'context':>9} {'weights':>9} {'KV':>8} {'act':>8} {'logits':>8} {'total':>9}")
    for ctx in [1_000, 4_000, 16_000, 64_000, 128_000]:
        m = estimate(cfg, ctx)
        print(f"{ctx:>9,} {gb(m['weights']):>8.2f}G {gb(m['kv_cache']):>7.2f}G "
              f"{gb(m['activations']):>7.2f}G {gb(m['logits']):>7.2f}G {gb(m['total']):>8.2f}G")

    print("\n=== 2. Porsi KV cache ===")
    for ctx in [1_000, 16_000, 128_000]:
        m = estimate(cfg, ctx)
        print(f"ctx {ctx:>7,}: KV = {m['kv_cache'] / m['total']:>5.0%} dari total, "
              f"{m['per_token_kv'] / 1024:.1f} KB per token")

    print("\n=== 3. Efek kuantisasi bobot ===")
    print(f"{'presisi':<12} {'1k ctx':>9} {'16k ctx':>9} {'128k ctx':>9}")
    for label, weight_bytes in [("fp16", 2), ("int8", 1), ("int4", 0.5)]:
        row = [gb(estimate(cfg, ctx, weight_bytes=weight_bytes)["total"]) for ctx in (1_000, 16_000, 128_000)]
        print(f"{label:<12} {row[0]:>8.1f}G {row[1]:>8.1f}G {row[2]:>8.1f}G")
    print("\nkuantisasi bobot tidak menyentuh KV cache; pada konteks panjang,")
    print("cache menjadi bottleneck yang tidak terbantu oleh int4.")

    print("\n=== 4. Batch maksimum untuk RAM 32 GB (fp16) ===")
    print(f"{'context':>9} {'batch maks':>11} {'total GB':>10}")
    for ctx in [1_000, 4_000, 16_000, 32_000]:
        b = max_batch(cfg, ctx, budget_gb=32)
        total = gb(estimate(cfg, ctx, batch=max(b, 1))["total"])
        print(f"{ctx:>9,} {b:>11} {total:>9.1f}G")
    print("\nuntuk RAM 16 GB, model 8B fp16 (16.1 GB bobot saja) sudah tidak muat.")
    print("kolom 'logits' menghitung logits untuk SEMUA posisi; implementasi")
    print("nyata biasanya hanya menghitung posisi terakhir saat decode, sehingga")
    print("angka itu adalah batas atas untuk prefill.")

    print("\n=== 5. Bandingkan konfigurasi model ===")
    print(f"{'model':<16} {'params':>9} {'weights':>9} {'KV/token':>10} {'8k ctx':>8}")
    for name, c in CONFIGS.items():
        m = estimate(c, 8_000)
        print(f"{name:<16} {c['n_params'] / 1e9:>8.2f}B {gb(m['weights']):>8.1f}G "
              f"{m['per_token_kv'] / 1024:>9.0f}K {gb(m['total']):>7.1f}G")
    print("\nLLaMA-2 70B punya KV per token 2.5x lebih besar (80 layer vs 32),")
    print("tetapi bobotnya 8x lebih besar. GQA menahan pertumbuhan cache, bukan")
    print("membuatnya identik.")

    print("\n=== 6. Implikasi untuk CPU-only ===")
    print("- RAM tidak secepat VRAM; membaca bobot + cache setiap token lambat.")
    print("- Model 0.5B-3B masih realistis untuk eksperimen di CPU.")
    print("- Model 7B+ di CPU: mungkin secara memori, tetapi jauh dari interaktif.")
    print("- Day 04 ini fokus pada pemahaman mekanisme, bukan kecepatan.")


if __name__ == "__main__":
    main()
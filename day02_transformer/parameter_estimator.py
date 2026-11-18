"""Estimasi jumlah parameter transformer dari konfigurasinya.

Rumus (tanpa bias, SwiGLU, GQA):

    attention = d^2 (Q) + 2 * d * n_kv * head_dim (K,V) + d^2 (O)
    ffn       = 3 * d * d_ff
    per layer = attention + ffn + 4d (dua LayerNorm/RMSNorm)
    total     = n_layers * per_layer
                + vocab * d            (embedding)
                + vocab * d (jika tidak tied)

Angka hasil rumus ini bisa dicocokkan dengan ukuran model yang dipublikasikan.
Kalau cocok, berarti kita sudah memahami isi arsitekturnya.

Jalankan: python day02_transformer/parameter_estimator.py
"""

from __future__ import annotations


def attention_params(d_model: int, n_heads: int, n_kv_heads: int) -> int:
    head_dim = d_model // n_heads
    q = d_model * d_model
    k = d_model * n_kv_heads * head_dim
    v = d_model * n_kv_heads * head_dim
    o = d_model * d_model
    return q + k + v + o


def estimate(config: dict) -> dict:
    d = config["d_model"]
    layers = config["n_layers"]
    attn = attention_params(d, config["n_heads"], config.get("n_kv_heads", config["n_heads"]))
    # 3 untuk SwiGLU (gate, up, down), 2 untuk FFN GELU/ReLU klasik
    ffn = config.get("ffn_multiplier", 3) * d * config["d_ff"]
    norms = 4 * d  # 2 norm per layer, masing-masing gamma (+ beta)
    per_layer = attn + ffn + norms

    emb = config["vocab_size"] * d
    pos_emb = config.get("max_positions", 0) * d
    lm_head = 0 if config.get("tied_embeddings", False) else config["vocab_size"] * d
    transformer = layers * per_layer

    return {
        "attention": layers * attn,
        "ffn": layers * ffn,
        "norms": layers * norms,
        "transformer": transformer,
        "embedding": emb,
        "pos_embedding": pos_emb,
        "lm_head": lm_head,
        "total": transformer + emb + pos_emb + lm_head,
        "per_layer": per_layer,
    }


CONFIGS = {
    "GPT-2 small": dict(vocab_size=50_257, d_model=768, n_layers=12, n_heads=12,
                        n_kv_heads=12, d_ff=3_072, tied_embeddings=True,
                        ffn_multiplier=2, max_positions=1_024, published=0.124),
    "LLaMA-2 7B": dict(vocab_size=32_000, d_model=4_096, n_layers=32, n_heads=32,
                       n_kv_heads=32, d_ff=11_008, tied_embeddings=False, published=6.74),
    "LLaMA-2 70B": dict(vocab_size=32_000, d_model=8_192, n_layers=80, n_heads=64,
                        n_kv_heads=8, d_ff=28_672, tied_embeddings=False, published=68.98),
    "Qwen2.5 7B": dict(vocab_size=152_064, d_model=3_584, n_layers=28, n_heads=28,
                       n_kv_heads=4, d_ff=18_944, tied_embeddings=False, published=7.62),
}


def main() -> None:
    print("=== 1. Estimasi vs parameter yang dipublikasikan ===")
    print(f"{'model':<13} {'estimasi':>10} {'published':>10} {'selisih':>9}")
    for name, cfg in CONFIGS.items():
        total = estimate(cfg)["total"] / 1e9
        print(f"{name:<13} {total:>9.2f}B {cfg['published']:>9.2f}B "
              f"{abs(total - cfg['published']) / cfg['published']:>8.2%}")
    print("\nselisih kecil berasal dari bias, norm, dan pembulatan konfigurasi.")

    print("\n=== 2. Ke mana parameter pergi? (LLaMA-2 7B) ===")
    cfg = CONFIGS["LLaMA-2 7B"]
    parts = estimate(cfg)
    for key in ["attention", "ffn", "norms", "embedding", "lm_head"]:
        print(f"  {key:<11} {parts[key] / 1e9:>6.3f}B  ({parts[key] / parts['total']:.1%})")

    print("\n=== 3. Efek GQA pada parameter (LLaMA-2 70B) ===")
    base = CONFIGS["LLaMA-2 70B"]
    for n_kv in [64, 16, 8, 4, 1]:
        variant = {**base, "n_kv_heads": n_kv}
        total = estimate(variant)["total"] / 1e9
        attn = estimate(variant)["attention"] / 1e9
        print(f"n_kv_heads={n_kv:>3}: total {total:>6.2f}B | attention {attn:>6.2f}B "
              f"({attn / total:.1%})")
    print("GQA menghemat parameter K/V, tetapi porsi attention memang kecil;")
    print("manfaat utamanya adalah memori KV cache saat inference (lihat Day 04).")

    print("\n=== 4. Efek ukuran vocabulary ===")
    for vocab in [32_000, 50_257, 128_256, 152_064]:
        cfg2 = {**CONFIGS["LLaMA-2 7B"], "vocab_size": vocab}
        parts2 = estimate(cfg2)
        print(f"vocab={vocab:>7,}: embedding+head {2 * parts2['embedding'] / 1e9:.2f}B "
              f"({2 * parts2['embedding'] / parts2['total']:.1%})")
    print("vocab besar memperbaiki tokenisasi multibahasa tetapi menambah parameter")
    print("dan memori logits: satu forward butuh vocab * batch * seq angka.")

    print("\n=== 5. Memori bobot untuk inference ===")
    print(f"{'model':<13} {'fp32':>8} {'fp16':>8} {'int8':>8} {'int4':>8}")
    for name, cfg3 in list(CONFIGS.items())[1:]:
        total = estimate(cfg3)["total"]
        print(f"{name:<13} {total * 4 / 1e9:>7.1f}G {total * 2 / 1e9:>7.1f}G "
              f"{total * 1 / 1e9:>7.1f}G {total * 0.5 / 1e9:>7.1f}G")
    print("\nhanya bobot. KV cache dan activation belum dihitung.")

    print("\n=== 6. Parameter vs FLOPs ===")
    print("setiap token melewati ~2 * N parameter (multiply + add),")
    print("jadi 7B parameter ~ 14 GFLOP per token forward.")
    print("dengan 50 token/detik, CPU butuh ~700 GFLOP/s efektif -")
    print("itulah sebabnya inference model besar praktis butuh GPU.")


if __name__ == "__main__":
    main()
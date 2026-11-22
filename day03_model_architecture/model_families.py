"""Keluarga arsitektur model: dense decoder-only, Mixture-of-Experts, encoder-decoder.

Yang dibandingkan:
1. Dense decoder-only (GPT, LLaMA, Qwen): setiap token melewati seluruh
   parameter. Sederhana, mudah di-batch, dan menjadi standar model chat.
2. Mixture-of-Experts (MoE): FFN diganti banyak "expert"; setiap token hanya
   melewati top-k expert. Total parameter besar, tapi parameter aktif per token
   kecil -> inference lebih murah dari ukuran totalnya.
3. Encoder-decoder (T5, BART): encoder membaca seluruh input (bidirectional),
   decoder menghasilkan output secara autoregressive. Bagus untuk
   terjemahan/ringkasan, kurang populer untuk chat.

Istilah "7B" pada model MoE hampir selalu berarti parameter total, bukan
parameter aktif. Ini sumber kebingungan umum saat memperkirakan biaya.

Jalankan: python day03_model_architecture/model_families.py
"""

from __future__ import annotations


def attention_params(d_model: int, n_heads: int, n_kv_heads: int) -> int:
    head_dim = d_model // n_heads
    return d_model * d_model + 2 * d_model * n_kv_heads * head_dim + d_model * d_model


def dense_layer_params(cfg: dict) -> dict:
    attn = attention_params(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"])
    ffn = 3 * cfg["d_model"] * cfg["d_ff"]
    return {"attention": attn, "ffn": ffn, "total": attn + ffn + 4 * cfg["d_model"]}


def moe_layer_params(cfg: dict) -> dict:
    """Setiap expert adalah FFN SwiGLU penuh; attention tetap satu."""
    attn = attention_params(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"])
    expert = 3 * cfg["d_model"] * cfg["d_ff"]
    total_ffn = cfg["n_experts"] * expert
    active_ffn = cfg["top_k"] * expert
    shared = 3 * cfg["d_model"] * cfg.get("shared_ffn", 0)
    return {
        "attention": attn,
        "experts_total": total_ffn,
        "experts_active": active_ffn,
        "shared": shared,
        "total": attn + total_ffn + shared + 4 * cfg["d_model"],
        "active": attn + active_ffn + shared + 4 * cfg["d_model"],
    }


DENSE = {
    "LLaMA-3 8B": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                       d_ff=14336, vocab_size=128_256),
    "Qwen2.5 0.5B": dict(d_model=896, n_layers=24, n_heads=14, n_kv_heads=2,
                         d_ff=4864, vocab_size=151_936, tied_embeddings=True),
}

MOE = {
    "Mixtral 8x7B": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                         d_ff=14336, n_experts=8, top_k=2, vocab_size=32_000),
    "Qwen1.5-MoE-A2.7B": dict(d_model=2048, n_layers=24, n_heads=16, n_kv_heads=16,
                              d_ff=5632, n_experts=60, top_k=4, vocab_size=151_936,
                              shared_ffn=1408),
}


def summarize_dense(cfg: dict) -> dict:
    per_layer = dense_layer_params(cfg)
    emb = cfg["vocab_size"] * cfg["d_model"]
    head = 0 if cfg.get("tied_embeddings", False) else emb
    total = cfg["n_layers"] * per_layer["total"] + emb + head
    return {"total": total, "active": total, "per_layer": per_layer}


def summarize_moe(cfg: dict) -> dict:
    per_layer = moe_layer_params(cfg)
    emb = 2 * cfg["vocab_size"] * cfg["d_model"]
    return {
        "total": cfg["n_layers"] * per_layer["total"] + emb,
        "active": cfg["n_layers"] * per_layer["active"] + emb,
        "per_layer": per_layer,
    }


def main() -> None:
    print("=== 1. Dense decoder-only ===")
    print(f"{'model':<18} {'total':>8} {'aktif':>8} {'porsi FFN':>10}")
    for name, cfg in DENSE.items():
        s = summarize_dense(cfg)
        p = s["per_layer"]
        print(f"{name:<18} {s['total'] / 1e9:>7.2f}B {s['active'] / 1e9:>7.2f}B "
              f"{p['ffn'] / p['total']:>9.0%}")
    print("\ncatatan: angka 'total' di atas termasuk tabel embedding.")
    print("sebagian laporan hanya menghitung parameter non-embedding, sehingga")
    print("angkanya lebih kecil; selalu cek definisi sebelum membandingkan.")

    print("\n=== 2. Mixture-of-Experts ===")
    print(f"{'model':<20} {'total':>8} {'aktif':>8} {'rasio':>7} {'top-k':>6}")
    for name, cfg in MOE.items():
        s = summarize_moe(cfg)
        print(f"{name:<20} {s['total'] / 1e9:>7.2f}B {s['active'] / 1e9:>7.2f}B "
              f"{s['active'] / s['total']:>6.0%} {cfg['top_k']:>6}")

    print("\n=== 3. Ke mana parameter MoE pergi? (Mixtral 8x7B) ===")
    cfg = MOE["Mixtral 8x7B"]
    p = moe_layer_params(cfg)
    for key in ["attention", "experts_total", "experts_active"]:
        print(f"  {key:<15} {p[key] / 1e6:>8.1f}M per layer")
    print(f"  total model  : {summarize_moe(cfg)['total'] / 1e9:.2f}B")
    print(f"  aktif/token  : {summarize_moe(cfg)['active'] / 1e9:.2f}B")
    print("hanya 2 dari 8 expert yang dipakai per token, jadi FLOPs ~ model 13B,")
    print("tetapi memori bobot tetap harus menampung semua 46B parameter.")

    print("\n=== 4. Trade-off MoE ===")
    print(f"{'aspek':<22} {'dense':<22} {'MoE':<22}")
    rows = [
        ("memori bobot", "sebanding parameter", "sebanding TOTAL parameter"),
        ("FLOPs/token", "sebanding parameter", "sebanding parameter AKTIF"),
        ("kualitas per FLOP", "baseline", "lebih tinggi pada skala besar"),
        ("kompleksitas serving", "rendah", "routing + load balancing"),
        ("batch kecil", "efisien", "boros (expert tidak terpakai merata)"),
    ]
    for a, b, c in rows:
        print(f"{a:<22} {b:<22} {c:<22}")

    print("\n=== 5. Encoder-decoder (T5-style) ===")
    enc = DENSE["Qwen2.5 0.5B"]
    print("encoder: attention bidirectional (tanpa causal mask)")
    print("decoder: causal self-attention + cross-attention ke output encoder")
    print(f"cross-attention menambah 2 proyeksi K/V: ~{2 * enc['d_model'] ** 2 / 1e6:.1f}M per layer")
    print("karena input dibaca dua kali (encode lalu cross-attend), latensi")
    print("lebih tinggi, tetapi cocok untuk tugas seq2seq dengan input panjang.")

    print("\n=== 6. Model lain yang perlu dikenal ===")
    print("RWKV / Mamba (SSM): state linear, kompleksitas O(n), bukan attention.")
    print("Hybrid (Jamba, Zamba): campuran attention dan SSM di dalam satu model.")
    print("Semua tetap memakai tokenizer, embedding, dan head yang sama;")
    print("yang berbeda adalah cara mencampur informasi antar posisi.")


if __name__ == "__main__":
    main()
"""Metode adaptasi model: full fine-tuning, LoRA, QLoRA, prefix/prompt tuning, RAG.

Perbandingan difokuskan pada angka, bukan opini:
- berapa parameter yang benar-benar diubah,
- berapa memori training,
- apakah bisa beberapa adapter untuk satu base model,
- apakah pengetahuan baru bisa masuk tanpa training.

Semua angka memakai model 7B (d_model=4096, n_layers=32) sebagai contoh.

Jalankan: python day03_model_architecture/adaptation_methods.py
"""

from __future__ import annotations

BASE_PARAMS = 7_000_000_000
BYTES_FP16 = 2


def lora_params(d_model: int, d_ff: int, n_layers: int, rank: int,
                targets: tuple[str, ...] = ("q", "k", "v", "o", "gate", "up", "down")) -> int:
    per_layer = 0
    for name in targets:
        out_features = d_ff if name in ("gate", "up", "down") else d_model
        per_layer += rank * (d_model + out_features)
    return per_layer * n_layers


def prefix_params(n_layers: int, n_heads: int, head_dim: int, prefix_len: int) -> int:
    """Prefix tuning: vektor key/value yang dipelajari untuk setiap layer."""
    return 2 * n_layers * n_heads * prefix_len * head_dim


def prompt_params(d_model: int, n_tokens: int) -> int:
    """Prompt tuning: embedding lembut (soft prompt) di input."""
    return n_tokens * d_model


def memory_estimate(n_trainable: int, base_bytes: int = BASE_PARAMS * BYTES_FP16,
                    train_base: bool = False) -> dict:
    """Perkiraan memori training (tanpa activation).

    - bobot base: selalu ada (fp16).
    - LoRA/adapter: gradient + Adam state hanya untuk parameter kecil.
    - full FT: gradient + 2 state Adam + fp32 master untuk SEMUA parameter.
    """
    adapter_weights = n_trainable * BYTES_FP16
    adapter_opt = n_trainable * (BYTES_FP16 + 8)
    if train_base:
        base_train = BASE_PARAMS * (BYTES_FP16 + 8 + 2)
    else:
        base_train = 0
    total = base_bytes + adapter_weights + adapter_opt + base_train
    return {"base": base_bytes, "trainable_state": adapter_weights + adapter_opt,
            "base_training_state": base_train, "total": total}


METHODS = {
    "Full fine-tuning": dict(trainable=BASE_PARAMS, train_base=True),
    "LoRA r=8 (semua proyeksi)": dict(trainable=lora_params(4096, 11008, 32, 8)),
    "LoRA r=16 (q,v saja)": dict(trainable=lora_params(4096, 11008, 32, 16, ("q", "v"))),
    "QLoRA r=16 (4-bit base)": dict(trainable=lora_params(4096, 11008, 32, 16), quantized=True),
    "Prefix tuning (20 token)": dict(trainable=prefix_params(32, 32, 128, 20)),
    "Prompt tuning (100 token)": dict(trainable=prompt_params(4096, 100)),
    "RAG (tanpa training)": dict(trainable=0),
}


def main() -> None:
    print("=== 1. Parameter yang diubah (model 7B) ===")
    print(f"{'metode':<30} {'trainable':>14} {'% dari 7B':>10}")
    for name, meta in METHODS.items():
        n = meta["trainable"]
        print(f"{name:<30} {n:>14,} {n / BASE_PARAMS:>9.3%}")

    print("\n=== 2. Memori training (tanpa activation) ===")
    print(f"{'metode':<30} {'total GB':>10} {'hemat vs full':>14}")
    full = memory_estimate(BASE_PARAMS, train_base=True)["total"]
    for name, meta in METHODS.items():
        if name.startswith("RAG"):
            print(f"{name:<30} {'0.0':>10} {'100%':>14}")
            continue
        base_bytes = BASE_PARAMS * BYTES_FP16
        if meta.get("quantized"):
            base_bytes = BASE_PARAMS * 0.5  # 4-bit
        total = memory_estimate(meta["trainable"], base_bytes, meta.get("train_base", False))["total"]
        print(f"{name:<30} {total / 1e9:>10.1f} {(1 - total / full):>13.0%}")

    print("\ncatatan: activation dan KV cache belum dihitung. Pada sequence panjang,")
    print("activation bisa jauh melebihi bobot, dan gradient checkpointing menukar")
    print("compute dengan memori.")

    print("\n=== 3. Kemampuan yang berbeda ===")
    rows = [
        ("menambah pengetahuan baru", "lemah", "lemah", "kuat"),
        ("mengubah gaya/format", "kuat", "kuat", "sedang"),
        ("data berubah tiap hari", "mahal", "mahal", "murah"),
        ("multi-tenant (banyak adapter)", "tidak", "ya", "ya"),
        ("tanpa label (unsupervised)", "tidak", "tidak", "ya"),
    ]
    print(f"{'kebutuhan':<30} {'full FT':<12} {'LoRA':<10} {'RAG':<8}")
    for a, b, c, d in rows:
        print(f"{a:<30} {b:<12} {c:<10} {d:<8}")

    print("\n=== 4. Multi-adapter serving ===")
    n_adapters = 20
    one_lora = lora_params(4096, 11008, 32, 8)
    print(f"1 base model    : {BASE_PARAMS / 1e9:.1f}B parameter ({BASE_PARAMS * 2 / 1e9:.1f} GB fp16)")
    print(f"{n_adapters} adapter LoRA: {n_adapters * one_lora / 1e6:.1f}M parameter tambahan")
    print(f"total           : {(BASE_PARAMS + n_adapters * one_lora) * 2 / 1e9:.1f} GB fp16")
    print(f"kalau tiap pelanggan punya full fine-tune sendiri:")
    print(f"  {n_adapters} x {BASE_PARAMS * 2 / 1e9:.1f} GB = {n_adapters * BASE_PARAMS * 2 / 1e9:.0f} GB")
    print("inilah alasan LoRA menjadi standar untuk produk multi-pelanggan.")

    print("\n=== 5. Urutan yang masuk akal dalam praktik ===")
    print("1. Coba prompt engineering + RAG dulu (tanpa training, bisa langsung diuji).")
    print("2. Kalau gaya/format belum konsisten -> SFT dengan LoRA pada data kecil.")
    print("3. Kalau preferensi manusia penting -> DPO setelah SFT.")
    print("4. Full fine-tuning hanya kalau ada data besar dan kebutuhan yang tidak")
    print("   bisa dicapai dengan adapter.")
    print("5. QLoRA kalau GPU terbatas: base 4-bit, adapter tetap fp16/bf16.")

    print("\n=== 6. Batasan yang jujur ===")
    print("- LoRA tidak menambah pengetahuan faktual secara andal.")
    print("- RAG bergantung pada kualitas retrieval; sampah masuk, sampah keluar.")
    print("- Semua metode di atas membutuhkan evaluasi, bukan hanya loss training.")


if __name__ == "__main__":
    main()
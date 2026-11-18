"""Satu decoder block lengkap: pre-norm + attention + residual + FFN + residual.

Urutan operasi (pre-norm, dipakai GPT-3/LLaMA/Qwen):

    h = LayerNorm(x)
    x = x + Attention(h)
    h = LayerNorm(x)
    x = x + FFN(h)

Residual connection (+) adalah alasan utama model dalam bisa dilatih: gradien
punya jalur lurus ke belakang, dan setiap layer hanya perlu belajar "perubahan
kecil" terhadap representasi yang sudah ada (residual stream).

Jalankan: python day02_transformer/decoder_block.py
"""

from __future__ import annotations

import numpy as np

from attention import attention_params, multi_head_attention
from feed_forward import make_swiglu, swiglu_forward, swiglu_params
from normalization import layer_norm, norm_params


class DecoderBlock:
    """Satu blok decoder minimal, tanpa dropout dan tanpa bias."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(d_model)
        self.d_model = d_model
        self.n_heads = n_heads

        # --- attention: 4 proyeksi persegi ---
        self.wq = rng.normal(0, scale, (d_model, d_model))
        self.wk = rng.normal(0, scale, (d_model, d_model))
        self.wv = rng.normal(0, scale, (d_model, d_model))
        self.wo = rng.normal(0, scale, (d_model, d_model))

        # --- normalization ---
        self.g1, self.b1 = np.ones(d_model), np.zeros(d_model)
        self.g2, self.b2 = np.ones(d_model), np.zeros(d_model)

        # --- FFN ---
        self.ffn = make_swiglu(d_model, d_ff, seed=seed + 1)

    def forward(self, x: np.ndarray, causal: bool = True) -> np.ndarray:
        # 1. attention dengan residual
        h = layer_norm(x, self.g1, self.b1)
        x = x + multi_head_attention(h, self.wq, self.wk, self.wv, self.wo,
                                     self.n_heads, causal=causal)
        # 2. FFN dengan residual
        h = layer_norm(x, self.g2, self.b2)
        x = x + swiglu_forward(h, self.ffn)
        return x

    def num_params(self) -> dict:
        d = self.d_model
        d_ff = self.ffn["W_gate"].shape[1]
        attn = attention_params(d)
        ffn = swiglu_params(d, d_ff)
        norms = 2 * norm_params(d, "layernorm")
        return {"attention": attn, "ffn": ffn, "norm": norms,
                "total": attn + ffn + norms}


def build_stack(n_layers: int, d_model: int, n_heads: int, d_ff: int) -> list[DecoderBlock]:
    return [DecoderBlock(d_model, n_heads, d_ff, seed=i) for i in range(n_layers)]


def main() -> None:
    d_model, n_heads, d_ff, seq = 64, 4, 172, 6
    block = DecoderBlock(d_model, n_heads, d_ff)
    x = np.random.default_rng(0).normal(size=(seq, d_model))

    print("=== 1. Forward satu block ===")
    h = layer_norm(x, block.g1, block.b1)
    attn_out = multi_head_attention(h, block.wq, block.wk, block.wv, block.wo, n_heads)
    after_attn = x + attn_out
    h2 = layer_norm(after_attn, block.g2, block.b2)
    ffn_out = swiglu_forward(h2, block.ffn)
    out = after_attn + ffn_out

    print(f"input           : {x.shape}")
    print(f"setelah norm    : {h.shape}")
    print(f"setelah attn    : {attn_out.shape}")
    print(f"setelah residual: {after_attn.shape}")
    print(f"setelah FFN     : {ffn_out.shape}")
    print(f"output block    : {out.shape}")
    print(f"norma input     : {np.linalg.norm(x):.2f}")
    print(f"norma output    : {np.linalg.norm(out):.2f}")

    print("\n=== 2. Distribusi parameter satu block ===")
    p = block.num_params()
    for key, value in p.items():
        print(f"  {key:<10} {value:>9,}  ({value / p['total']:.0%})")
    print("\nFFN mendominasi parameter; attention relatif kecil.")

    print("\n=== 3. Menumpuk 12 block ===")
    blocks = build_stack(12, d_model, n_heads, d_ff)
    h = x.copy()
    norms = []
    for i, blk in enumerate(blocks, 1):
        h = blk.forward(h)
        norms.append(float(np.linalg.norm(h)))
        if i in (1, 4, 8, 12):
            print(f"layer {i:>2}: norma residual stream = {norms[-1]:.2f}")
    print("residual stream tumbuh perlahan, bukan meledak -> ini efek pre-norm.")

    print("\n=== 4. Kausalitas tetap terjaga setelah 12 block ===")
    x_modified = x.copy()
    x_modified[-1] += 10.0
    h2 = x_modified.copy()
    for blk in blocks:
        h2 = blk.forward(h2)
    diff = np.abs(h - h2).max(axis=-1)
    print(f"selisih per posisi: {np.round(diff, 6)}")
    print("posisi terakhir berubah, posisi sebelumnya tidak -> tidak ada kebocoran.")
    print("catatan: ini hanya benar kalau setiap attention memakai causal mask.")

    print("\n=== 5. Skala model nyata ===")
    real = {"LLaMA-7B": (4096, 32, 11008, 32), "Qwen2.5-7B": (3584, 28, 18944, 28)}
    print(f"{'model':<12} {'block params':>14} {'x layers':>12}")
    for name, (d, heads, ff, layers) in real.items():
        per_block = attention_params(d) + swiglu_params(d, ff) + 4 * d
        print(f"{name:<12} {per_block / 1e6:>13.1f}M {per_block * layers / 1e9:>11.2f}B")

    print("\n=== 6. Hal yang sengaja dihilangkan di demo ini ===")
    print("- dropout (hanya aktif saat training)")
    print("- bias pada proyeksi (model modern sering mematikannya)")
    print("- RoPE (posisi diputar di dalam attention, lihat rope.py)")
    print("- GQA (lihat attention_variants.py)")
    print("Semuanya menambah detail, bukan mengubah alur besar di atas.")


if __name__ == "__main__":
    main()
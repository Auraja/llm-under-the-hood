# Day 02 - Transformer

## Tujuan

Membuka "kotak hitam" satu decoder block: embedding, posisi, attention, FFN,
normalization, residual. Semua diimplementasikan dengan NumPy agar bentuk
tensor dan operasinya terlihat.

Setelah hari ini, kamu bisa menjawab:

- Kenapa attention butuh posisi, dan apa bedanya absolute PE dengan RoPE?
- Kenapa skor attention dibagi `sqrt(d_k)`?
- Apa yang sebenarnya dihemat MQA/GQA?
- Di mana sebagian besar parameter model berada?
- Kenapa pre-norm lebih stabil untuk model dalam?

## Eksperimen

| File | Yang dilakukan |
|---|---|
| `embeddings.py` | Lookup vs one-hot matmul, cosine similarity, weight tying |
| `positional_encoding.py` | PE sinusoidal, sifat jarak, keterbatasannya |
| `rope.py` | Rotasi q/k, bukti dot product bergantung jarak relatif |
| `attention.py` | Softmax, causal mask, MHA, cross-check PyTorch |
| `attention_variants.py` | MHA/MQA/GQA, parameter, ukuran KV cache |
| `feed_forward.py` | FFN 4x, GELU/SiLU, SwiGLU dan aturan 8/3 |
| `normalization.py` | LayerNorm vs RMSNorm, pre-norm vs post-norm |
| `decoder_block.py` | Satu block lengkap + stack 12 layer |
| `parameter_estimator.py` | Estimasi parameter dan bobot memori |

```bash
python day02_transformer/attention.py
python day02_transformer/rope.py
python day02_transformer/parameter_estimator.py
```

Semua file mandiri, CPU only, tanpa unduhan model.

## Konsep inti

### Attention

```
scores  = Q K^T / sqrt(d_k)
scores  = where(causal_mask, scores, -inf)
weights = softmax(scores)
output  = weights V
```

- `sqrt(d_k)` menjaga varians skor tetap ~1. Tanpa itu softmax menjadi
  hampir one-hot dan gradien mengecil.
- Causal mask membuat token hanya melihat masa lalu. Ini yang membedakan
  encoder (bidirectional) dari decoder (autoregressive).
- Attention tidak mencampur informasi antar head secara langsung; setiap head
  punya proyeksi sendiri, lalu digabung lewat `W_o`.

### Posisi

Attention murni bersifat permutation equivariant. Posisi harus disuntikkan.

- **Absolute PE**: vektor sin/cos ditambahkan ke embedding. Murah, tetapi
  terikat panjang maksimum training.
- **RoPE**: q dan k diputar; hasilnya hanya bergantung pada jarak relatif.
  Tidak menambah parameter dan dipakai hampir semua model modern.

### FFN dan parameter

FFN bekerja per token, tanpa interaksi antar posisi. Karena lebarnya 2.7-4x
`d_model`, FFN menyumbang ~65-75% parameter transformer.

### Normalization

LayerNorm dan RMSNorm menormalkan per token, bukan per batch, sehingga
konsisten saat decoding satu token. Pre-norm (`x + sublayer(norm(x))`)
menjaga residual stream tetap bersih dan membuat model dalam lebih stabil.

## Hasil

Angka lengkap ada di [`results.md`](results.md). Ringkasnya:

- Incremental attention dengan cache identik dengan full attention
  (selisih 0).
- NumPy vs PyTorch `scaled_dot_product_attention`: selisih 2.2e-16.
- Estimasi parameter cocok dengan angka publik: LLaMA-2 7B 6.74B, 70B 68.98B,
  Qwen2.5 7B 7.62B (selisih < 0.1%).
- FFN = 64% parameter LLaMA-2 7B; attention 32%.
- GQA-8 memotong KV cache 4x dibanding MHA (pada 8k token: 1.0 GB vs 4.1 GB).
- Bobot LLaMA-2 70B: 138 GB fp16, 34.5 GB int4 (belum termasuk KV cache).

## Pertanyaan untuk didiskusikan

1. Kalau attention bisa melihat semua token, kenapa masih perlu banyak layer?
2. Kenapa menambah `n_kv_heads` menghemat memori inference tetapi hampir tidak
   mengubah jumlah parameter?
3. Kalau FFN mendominasi parameter, kenapa arsitektur tidak menghapus attention?
4. Apa yang terjadi pada model kalau `sqrt(d_k)` dihilangkan saat training?

## Lanjut ke Day 03

Kita sudah punya satu block dan bisa menghitung ukurannya. Berikutnya: bagaimana
model dilatih (loss, gradient, DPO), dan bagaimana ia diadaptasi tanpa melatih
ulang seluruh bobot (LoRA, quantization, RAG).
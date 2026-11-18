# Hasil Eksekusi - Day 02: Transformer

| Item | Nilai |
|---|---|
| Tanggal | 17 November 2026 |
| Python / NumPy | 3.12.6 / 2.2.6 |
| PyTorch | 2.5.1+cpu |
| Perangkat | CPU 12 core, 16 GB RAM |

Semua angka di bawah berasal dari eksekusi nyata, bukan estimasi manual.

## 1. Attention (`attention.py`)

Skor `QK^T / sqrt(8)` untuk 4 token:

```
[[ 1.00  1.45 -0.32  0.26]
 [-0.43 -1.99 -0.71 -0.36]
 [ 0.10  0.29 -0.22 -1.59]
 [-0.50 -1.69  1.40  0.08]]
```

Setelah causal mask, baris 0 hanya menyisakan token 0 -> bobot `[1, 0, 0, 0]`.
Jumlah bobot setiap baris = 1.0.

Uji kebocoran: mengubah key/value token terakhir

| Posisi output | Selisih |
|---|---:|
| 0-2 | 0.00e+00 |
| 3 | 9.98e+00 |

Kausalitas terjaga. Cross-check dengan PyTorch
`F.scaled_dot_product_attention(..., is_causal=True)`: selisih maksimum
**2.22e-16**.

Pengaruh skala `sqrt(d_k)` (d_k=256):

| Versi | Nilai dot |
|---|---:|
| mentah | +20.5 |
| dibagi `sqrt(256)` | +1.3 |

Tanpa skala, softmax atas nilai puluhan menjadi hampir one-hot.

Multi-head (d_model=32, 4 head): output `(4, 32)`, setiap head 8 dimensi,
4.096 parameter untuk 4 proyeksi.

Kompleksitas skor attention: 8.192 token = 67 juta elemen per head
(256 MB fp32). Ini belum termasuk bobot dan nilai.

## 2. Varian attention (`attention_variants.py`)

Parameter proyeksi (d_model 4096, 32 layer, tanpa bias):

| Varian | Q head | KV head | Parameter | vs MHA |
|---|---:|---:|---:|---:|
| MHA (LLaMA-2 7B) | 32 | 32 | 67.108.864 | 1.00x |
| GQA (Mistral 7B) | 32 | 8 | 41.943.040 | 0.62x |
| MQA (PaLM-style) | 32 | 1 | 34.603.008 | 0.52x |

KV cache fp16, batch 1:

| Varian | 1k token | 32k token | 128k token |
|---|---:|---:|---:|
| MHA (32 KV) | 500 MB | 16.0 GB | 64.0 GB |
| GQA (8 KV) | 125 MB | 4.0 GB | 16.0 GB |
| MQA (1 KV) | 15.6 MB | 0.5 GB | 2.0 GB |

Temuan utama: parameter attention hanya turun 38% dari MHA ke MQA, tetapi
KV cache turun **32x**. Manfaat GQA/MQA ada di inference, bukan di ukuran
model.

## 3. Embedding (`embeddings.py`)

- Lookup vs one-hot matmul: selisih **0.00e+00** (identik).
- Memori untuk 3 token: one-hot 23.4 KB vs lookup 1.5 KB.
- `cosine(a, 2a) = 1.000`, `cosine(a, -a) = -1.000`: embedding hanya bermakna
  secara arah; LayerNorm/RMSNorm memang menormalkan magnitudo.
- Weight tying menghemat 64.000 parameter pada contoh; pada Qwen2.5-7B
  tabel embedding saja 0.54B parameter.

Embedding acak membuat tetangga terdekat hampir selalu dirinya sendiri
(sim=1.000). Kemiripan semantik muncul setelah training, bukan dari
inisialisasi.

## 4. Posisi

### Sinusoidal (`positional_encoding.py`)

- Bentuk PE `(128, 16)`, rentang nilai [-1, 1].
- Posisi 0 selalu `[0, 1, 0, 1, ...]`.
- Rata-rata dot produk menurut jarak:

| Jarak | Rata-rata dot |
|---:|---:|
| 0 | 8.000 |
| 1 | 7.485 |
| 2 | 6.368 |
| 4 | 5.560 |
| 8 | 4.697 |
| 16 | 4.214 |
| 32 | 3.541 |
| 64 | 4.908 |

Trennya menurun, tetapi tidak monoton sempurna karena komponen cos punya
rata-rata positif.

### RoPE (`rope.py`)

- Rotasi tidak mengubah norma: 1.8627 -> 1.8627.
- Dot product hanya bergantung pada jarak:

| m | n | m-n | q_m . k_n |
|---:|---:|---:|---:|
| 5 | 5 | 0 | -1.467987 |
| 5 | 4 | 1 | -1.695379 |
| 50 | 49 | 1 | -1.695379 |
| 10 | 6 | 4 | -1.302115 |
| 100 | 96 | 4 | -1.302115 |

Jarak yang sama -> nilai sama persis. Bandingkan dengan absolute PE:

| Jarak | RoPE@30 | sinus@10 | sinus@60 |
|---:|---:|---:|---:|
| 0 | -1.4680 | 4.1844 | 2.8953 |
| 2 | -1.3804 | 2.5572 | 1.3087 |
| 8 | -1.3369 | 2.4764 | 1.6289 |
| 32 | -0.9143 | 0.5054 | 0.5322 |

RoPE konsisten antar posisi; sinusoidal berubah karena posisi hanya
ditambahkan. RoPE juga tidak menambah parameter.

Periode rotasi (dim 64): base 10.000 -> 6.28 token (tercepat) sampai 47.117
token (terlambat). base 500.000 -> 2.084.765 token, sehingga posisi jauh
masih bisa dibedakan (long context).

## 5. FFN dan normalization

### FFN (`feed_forward.py`)

| Item | Nilai |
|---|---:|
| FFN klasik 4x | 33.088 parameter |
| Attention (d=64) | 16.384 parameter |
| Rasio | 2.02x |
| SwiGLU (d_ff 8/3) | 33.792 parameter |

Aktivasi pada x = -1: ReLU 0, GELU -0.159, SiLU -0.269. GELU/SiLU membiarkan
sinyal negatif kecil lewat.

Skala nyata: FFN menyumbang 67% (LLaMA-7B) dan 72% (LLaMA-70B) parameter
per layer.

Mengubah satu token tidak mengubah output token lain sama sekali
(selisih 0.00e+00) — FFN bekerja per posisi.

### Normalization (`normalization.py`)

| Operasi | Mean output | Var/RMS | Parameter per layer (d=4096) |
|---|---|---:|---:|
| LayerNorm | 0.0 | 1.0 | 8.192 |
| RMSNorm | tidak dipaksa 0 | 1.0 | 4.096 |

Pertumbuhan residual stream 24 layer:

| Layer | Pre-norm | Post-norm |
|---:|---:|---:|
| 1 | 19.37 | 16.00 |
| 4 | 26.34 | 16.00 |
| 8 | 32.00 | 16.00 |
| 16 | 39.71 | 16.00 |
| 24 | 46.40 | 16.00 |

Pre-norm membiarkan residual tumbuh perlahan (*identity path* bersih),
post-norm memaksa skala di setiap layer tetapi gradien harus melewati banyak
normalisasi.

## 6. Decoder block (`decoder_block.py`)

Satu block (d=64, 4 head, d_ff=172):

| Komponen | Parameter | Porsi |
|---|---:|---:|
| Attention | 16.384 | 33% |
| FFN | 33.024 | 66% |
| Norm | 256 | 1% |
| **Total** | **49.664** | 100% |

Stack 12 block: norma residual 31.71 -> 51.31 -> 63.41 -> 78.45. Tidak
meledak.

Kausalitas setelah 12 block: selisih per posisi `[0, 0, 0, 0, 0, 10]` — hanya
posisi terakhir berubah.

Skala nyata: LLaMA-7B 202.4M parameter per block; Qwen2.5-7B 255.1M.

## 7. Estimasi parameter (`parameter_estimator.py`)

| Model | Estimasi | Publikasi | Selisih |
|---|---:|---:|---:|
| GPT-2 small | 0.12B | 0.12B | 0.29% |
| LLaMA-2 7B | 6.74B | 6.74B | 0.02% |
| LLaMA-2 70B | 68.98B | 68.98B | 0.00% |
| Qwen2.5 7B | 7.62B | 7.62B | 0.06% |

Rumus yang dipakai terbukti cukup akurat untuk model tanpa bias.

Distribusi LLaMA-2 7B:

| Bagian | Parameter | Porsi |
|---|---:|---:|
| FFN | 4.329B | 64.2% |
| Attention | 2.147B | 31.9% |
| Embedding | 0.131B | 1.9% |
| LM head | 0.131B | 1.9% |

Efek GQA pada LLaMA-2 70B:

| n_kv_heads | Total | Attention |
|---:|---:|---:|
| 64 (MHA) | 78.37B | 21.47B |
| 8 | 68.98B | 12.08B |
| 1 (MQA) | 67.80B | 10.91B |

Memori bobot:

| Model | fp32 | fp16 | int8 | int4 |
|---|---:|---:|---:|---:|
| LLaMA-2 7B | 27.0G | 13.5G | 6.7G | 3.4G |
| LLaMA-2 70B | 275.9G | 138.0G | 69.0G | 34.5G |

## Kesimpulan

1. Attention adalah satu-satunya tempat informasi berpindah antar posisi;
   FFN, LayerNorm, dan embedding semuanya per-token.
2. Posisi wajib ada. RoPE lebih baik untuk generalisasi panjang karena
   sifatnya hanya bergantung pada jarak relatif.
3. Sebagian besar parameter ada di FFN, sehingga estimasi ukuran model harus
   menghitung `3 * d * d_ff * n_layers`, bukan attention saja.
4. GQA/MQA hampir tidak mengubah ukuran model, tetapi memotong KV cache
   secara drastis. Itu optimasi inference, bukan optimasi parameter.
5. Rumus parameter sederhana bisa mereproduksi angka publik dengan error
   < 0.3%, yang berarti arsitektur model modern bisa dipahami tanpa membaca
   kode implementasinya.

## Keterbatasan

- Demo tidak memakai bias, dropout, atau RoPE di dalam attention block.
- Estimasi mengabaikan bias; model yang memakai bias akan sedikit berbeda.
- Uji numerik memakai float64 dan matriks kecil; perilaku float16 di GPU
  bisa berbeda (terutama overflow dan presisi).
- Label "publikasi" untuk Qwen/LLaMA mengikuti kartu model, yang kadang
  menghitung atau tidak menghitung embedding secara berbeda.
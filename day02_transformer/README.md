# Day 02 - Transformer

Setelah Day 01, satu token cuma sebaris angka yang berdiri sendiri. Hari ini aku
bongkar satu decoder block sampai kelihatan bentuk tensor dan angkanya:
embedding, posisi, attention, FFN, normalisasi, residual.

Semua NumPy, CPU only, tanpa `from_pretrained`.

## Kenapa aku tidak mengikuti urutan file

Urutan di repo ini (`embeddings → posisi → rope → attention`) masuk akal buat
penulis kode, tapi tidak buat pembaca. Aku pakai urutan ini:

```
embedding → attention → posisi → rope → varian attention → ffn → norm → block → hitung parameter
```

Alasannya: posisi baru terasa perlu **setelah** tahu bahwa attention itu buta
urutan. Kalau dibaca dari awal, "kenapa perlu posisi" cuma jadi hafalan.

## Satu baris attention

```
scores  = Q K^T / sqrt(d_k)
scores  = where(causal_mask, scores, -inf)
weights = softmax(scores)
output  = weights V
```

Tiga hal yang akhirnya klik buat aku:

- **`sqrt(d_k)` bukan hiasan.** Dengan `d_k=256`, dot mentah bisa +20.5 dan
  setelah diskala jadi +1.3. Tanpa skala, softmax jadi nyaris one-hot: model
  cuma melihat satu token, dan gradiennya mengecil sampai training mandek.
- **Causal mask itu keharusan, bukan optimasi.** Kalau token boleh melihat ke
  depan saat training, model cuma perlu menyalin jawaban. Loss turun, model
  kelihatan hebat, lalu gagal total saat dipakai — karena saat menulis, kata
  berikutnya memang belum ada. Aku mengujinya dengan mengubah token terakhir
  dan memastikan output token sebelumnya **tidak berubah** (0.00e+00).
- **Attention tidak mencampur head secara langsung.** Tiap head punya proyeksi
  sendiri, digabung lewat `W_o`.

## Posisi: dua cara

Attention murni **permutation equivariant** — tukar urutan token, hasilnya ikut
tertukar. Artinya tanpa posisi, "kucing makan ikan" dan "ikan makan kucing"
identik.

- **Absolute PE**: vektor sin/cos ditambahkan ke embedding. Murah, tapi terikat
  panjang training.
- **RoPE**: q dan k **diputar**, dan hasil dot product-nya hanya bergantung
  jarak relatif. Tidak menambah parameter, dan dipakai hampir semua model
  modern (plus YaRN untuk memperpanjang konteks).

## Sisanya, singkat

- **FFN** bekerja per token tanpa interaksi antar posisi, lebarnya 2.7–4x
  `d_model`. Justru di sini sebagian besar parameter tinggal.
- **LayerNorm / RMSNorm** menormalkan per token, bukan per batch, jadi
  konsisten saat decoding satu token. BatchNorm tidak bisa.
- **Pre-norm** (`x + sublayer(norm(x))`) menjaga residual stream tetap bersih;
  model dalam jadi lebih stabil. Post-norm (transformer asli) lebih rewel dan
  butuh warmup.

## Angka yang aku hitung sendiri

| Yang dihitung | Hasil |
|---|---|
| NumPy vs PyTorch `scaled_dot_product_attention` | selisih **2.2e-16** |
| Estimasi parameter LLaMA-2 7B / 70B | **6.74B / 68.98B** (selisih <0.1%) |
| Estimasi parameter Qwen2.5 7B | **7.62B** |
| Porsi FFN di LLaMA-2 7B | **64.2%** (attention 31.9%) |
| KV cache MHA vs GQA-8 (d=4096, 32 head) | 4.096 MB → **1.024 MB** (4x) |
| Bobot LLaMA-2 70B | 138 GB fp16, 34.5 GB int4 |

Yang bikin kaget: **GQA bukan soal menghemat parameter** (attention cuma ~32%
dari total). Yang dihemat adalah **KV cache saat inference**, dan itulah yang
bikin konteks 128k bisa jalan.

Detail ada di [`results.md`](results.md).

## Yang belum aku pahami

1. Kalau attention sudah bisa melihat semua token sekaligus, kenapa masih butuh
   12+ layer? Apa yang bisa dilakukan layer ke-10 yang tidak bisa dilakukan
   layer pertama?
2. Kalau FFN memegang 64% parameter, kenapa arsitekturnya tidak menghapus
   attention saja?
3. Menambah `n_kv_heads` menghemat memori inference tapi hampir tidak mengubah
   parameter. Di titik mana penghematan itu mulai merusak kualitas?

## Lanjut

Sekarang aku punya satu block dan bisa menghitung ukurannya. Berikutnya:
bagaimana model dilatih (loss, gradient, DPO), dan bagaimana ia diadaptasi tanpa
melatih ulang seluruh bobot. Itu Day 03.
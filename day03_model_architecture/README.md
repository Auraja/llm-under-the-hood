# Day 03 - Model Architecture, Training Concepts, Adaptation

## Tujuan

Memahami apa yang terjadi **di sekitar** bobot: bagaimana model dilatih, apa
arti loss dan perplexity, kenapa ada MoE, dan bagaimana model diadaptasi tanpa
melatih ulang semuanya.

Batasan penting: **tidak ada training atau fine-tuning** di hari ini. Semua
angka dihitung dari rumus dan simulasi kecil yang bisa diperiksa dengan
kalkulator. Satu-satunya "gradient step" adalah aritmetika satu parameter
untuk menjelaskan bentuk update.

## Eksperimen

| File | Yang dilakukan |
|---|---|
| `model_families.py` | Dense vs MoE, parameter total vs aktif, encoder-decoder |
| `training_concepts.py` | Shift target, gradient satu langkah, memori training, LR schedule, DPO |
| `loss_perplexity.py` | Cross-entropy, perplexity, temperature, bits/token |
| `adaptation_methods.py` | Full FT vs LoRA vs QLoRA vs prefix/prompt tuning vs RAG |
| `lora_estimator.py` | Rumus parameter LoRA, merge, memori adapter |
| `quantization_estimator.py` | Simulasi int8/int4 per-tensor/per-group + memori |
| `rag_vs_finetuning.py` | Titik impas biaya, latensi, matriks keputusan |

```bash
python day03_model_architecture/loss_perplexity.py
python day03_model_architecture/lora_estimator.py
python day03_model_architecture/rag_vs_finetuning.py
```

## Konsep inti

### Causal LM dan loss

Satu sequence panjang menghasilkan `n-1` contoh training: posisi `i` menjadi
input, posisi `i+1` menjadi target. Karena itu pretraining sangat efisien.

`loss = -log p(target)`, dan `perplexity = exp(loss)`. Perplexity adalah
jumlah pilihan efektif yang dipertimbangkan model. PPL hanya sebanding kalau
tokenizer-nya sama.

### Memori training

Untuk setiap parameter: bobot (2 byte fp16), gradient (2 byte), state Adam
(fp32 m + v = 8 byte), dan master weight fp32 (4 byte). Total ~16 byte per
parameter. Model 7B -> ~112 GB hanya untuk state training, belum activation.
Inilah alasan LoRA/QLoRA ada.

### LoRA

`W' = W + (alpha/r) * B A`. Parameter tambahan `r * (d_in + d_out)` per modul.
Untuk 7B, semua proyeksi dengan r=16 hanya ~40 juta parameter (0.57%).
Karena `B` diinisialisasi nol, adapter awalnya tidak mengubah output. Setelah
training, `W'` bisa di-merge sehingga tidak ada overhead inference.

### DPO

Membandingkan log-probabilitas chosen vs rejected **relatif terhadap model
referensi**, lalu melewatkannya ke sigmoid. Tidak butuh reward model terpisah
dan tidak butuh sampling RL.

### MoE

Setiap token hanya melewati top-k expert. Mixtral 8x7B: total 46.7B parameter,
aktif 12.9B per token. Memori bobot tetap sebesar parameter total, tetapi FLOPs
per token jauh lebih kecil.

### Quantization

Mengurangi bit per bobot. int4 per-tensor rusak oleh outlier; per-group
(64-128) memulihkan sebagian besar kualitas. Kuantisasi tidak membantu KV
cache, jadi pada konteks panjang manfaatnya terbatas.

### RAG vs fine-tuning

RAG menambah token di prompt; fine-tuning mengubah bobot. Hitung titik impas:
biaya training sekali vs tambahan token per request. Untuk data yang berubah,
RAG menang secara default.

## Hasil

Angka lengkap ada di [`results.md`](results.md). Ringkasnya:

- Perplexity 1/p untuk satu token; baseline acak vocab 50k = 50.000.
- LoRA r=16 semua proyeksi: 39.98M parameter (0.57%), memori adapter 480 MB
  vs 112 GB state full fine-tuning (~233x lebih kecil).
- Merge LoRA identik dengan output unmerged (selisih 1.4e-15).
- int4 per-tensor pada matriks dengan outlier: relative MSE 2.6e-2; per-group
  128 turun ke 1.4e-2.
- Mixtral 8x7B: 46.70B total, 12.88B aktif (28%).
- Titik impas RAG vs fine-tuning untuk FAQ: ~404.000 request.

## Pertanyaan untuk didiskusikan

1. Kalau LoRA hanya 0.5% parameter, kenapa ia bisa mengubah gaya jawaban?
2. Kenapa DPO membandingkan dengan model referensi, bukan probabilitas absolut?
3. Kapan MoE justru lebih mahal daripada dense pada serving?
4. Kalau kuantisasi 4-bit "hampir sama", kenapa tidak semua model dirilis 4-bit?

## Lanjut ke Day 04

Sekarang kita tahu isi model dan biaya melatihnya. Berikutnya: bagaimana model
menghasilkan token satu per satu, kenapa token pertama lambat, dan mengapa
konteks panjang menghabiskan memori.
# Hasil Eksekusi - Day 03: Architecture, Training Concepts, Adaptation

Tidak ada model yang dilatih di hari ini. Semua angka adalah hasil perhitungan
eksplisit dan simulasi kecil di CPU.

| Item | Nilai |
|---|---|
| Tanggal | 17 November 2026 |
| Python / NumPy | 3.12.6 / 2.2.6 |
| Perangkat | CPU 12 core, 16 GB RAM |

## 1. Keluarga model (`model_families.py`)

| Model | Total | Aktif | Porsi FFN |
|---|---:|---:|---:|
| LLaMA-3 8B (dense) | 8.03B | 8.03B | 81% |
| Qwen2.5 0.5B (dense) | 0.49B | 0.49B | 88% |
| Mixtral 8x7B (MoE) | 46.70B | 12.88B | — |
| Qwen1.5-MoE-A2.7B | 51.06B | 4.55B | — |

MoE Mixtral: 12.88B aktif = 28% dari total. Secara FLOPs setara model ~13B,
tetapi memori bobot tetap 46.7B. Setiap layer: attention 41.9M, expert total
1.409,3M, expert aktif 352,3M.

Qwen1.5-MoE: 60 expert, top-4, plus shared expert. Hanya 9% parameter aktif
per token — ekstrem, tetapi biaya memori dan kompleksitas routing tinggi.

## 2. Konsep training (`training_concepts.py`)

### Target causal LM

| Posisi | Input | Target |
|---:|---|---|
| 0 | `<bos>` | `the` |
| 1 | `<bos> the` | `cat` |
| 2 | `<bos> the cat` | `sat` |
| 3 | `<bos> the cat sat` | `down` |
| 4 | `<bos> the cat sat down` | `<eos>` |

5 token -> 4 contoh training dari satu sequence.

### Satu langkah gradient (aritmetika)

```
loss    = (0.0 - 3.0)^2          = 9.0000
gradien = 2 * (0.0 - 3.0)        = -6.0000
update  = 0.0 - 0.1 * (-6.0)     = 0.6000
loss baru                        = 5.7600
```

Tidak ada training di sini; ini hanya menunjukkan bentuk update. Di model
nyata gradien dihitung dengan backpropagation, tetapi arah update-nya sama.

### Memori training 7B (fp16)

| Komponen | GB |
|---|---:|
| Bobot | 14.0 |
| Gradient | 14.0 |
| Optimizer (Adam fp32 m+v) | 56.0 |
| Master weights fp32 | 28.0 |
| **Total** | **112.0** |

Inference hanya butuh 14.0 GB -> training ~8x lebih besar, belum activation.
Angka ini yang membuat LoRA/QLoRA relevan.

### Learning rate schedule

| Step | LR |
|---:|---:|
| 0 | 3.0e-06 |
| 50 | 1.53e-04 |
| 100 | 3.00e-04 |
| 500 | 1.76e-04 |
| 999 | 9.14e-10 |

### DPO

| Skenario | Margin | Loss | P(chosen) |
|---|---:|---:|---:|
| Model sudah benar | 1.500 | 0.201 | 81.8% |
| Model ragu | 0.150 | 0.621 | 53.7% |
| Model salah | -1.100 | 1.387 | 25.0% |

DPO membandingkan selisih log-prob terhadap model referensi. Model yang
sudah benar mendapat margin positif besar; model yang salah mendapat margin
negatif.

## 3. Loss dan perplexity (`loss_perplexity.py`)

| p(target) | Loss | Perplexity | Bits |
|---:|---:|---:|---:|
| 0.99 | 0.010 | 1.01 | 0.014 |
| 0.90 | 0.105 | 1.11 | 0.152 |
| 0.50 | 0.693 | 2.00 | 1.000 |
| 0.10 | 2.303 | 10.00 | 3.322 |
| 0.01 | 4.605 | 100.00 | 6.644 |

Baseline acak: vocab 32k -> PPL 32.000; vocab 50k -> 50.257; vocab 128k ->
128.256. Sebuah model harus jauh di bawah baseline ini.

Sequence 6 token dengan probabilitas bervariasi: mean loss 0.5437,
perplexity 1.72, 0.784 bit/token.

Efek temperature pada token target yang sama:

| T | p(target) | Loss | PPL |
|---:|---:|---:|---:|
| 0.5 | 0.946 | 0.055 | 1.057 |
| 1.0 | 0.756 | 0.280 | 1.324 |
| 2.0 | 0.528 | 0.639 | 1.894 |

Loss turun saat T<1, tetapi itu hanya efek mempertajam distribusi, bukan
tanda model lebih benar.

## 4. Metode adaptasi (`adaptation_methods.py`)

| Metode | Trainable | % dari 7B | Memori training | Hemat vs full |
|---|---:|---:|---:|---:|
| Full fine-tuning | 7.000.000.000 | 100% | 182 GB | 0% |
| LoRA r=8 semua proyeksi | 19.988.480 | 0.286% | 14.2 GB | 92% |
| LoRA r=16 q,v | 8.388.608 | 0.120% | 14.1 GB | 92% |
| QLoRA r=16 (base 4-bit) | 39.976.960 | 0.571% | 4.0 GB | 98% |
| Prefix tuning 20 token | 5.242.880 | 0.075% | 14.1 GB | 92% |
| Prompt tuning 100 token | 409.600 | 0.006% | 14.0 GB | 92% |
| RAG | 0 | 0% | 0 | 100% |

Catatan: "memori training" di sini selalu menyertakan bobot base. Full FT
menambah gradient + optimizer untuk semua parameter; LoRA hanya untuk adapter.

Multi-adapter serving: 1 base 7B (14.0 GB) + 20 adapter r=8 (399.8M parameter,
0.8 GB) = 14.8 GB, dibandingkan 20 full fine-tune (280 GB).

## 5. LoRA (`lora_estimator.py`)

Parameter per modul, r=8:

| Modul | Bentuk | Penuh | LoRA | Rasio |
|---|---:|---:|---:|---:|
| q_proj | 4096x4096 | 16.777.216 | 65.536 | 0.39% |
| gate_proj | 11008x4096 | 45.088.768 | 120.832 | 0.27% |

Total 7B (32 layer):

| Target | r=8 | r=16 | r=64 |
|---|---:|---:|---:|
| q, v | 4.19M | 8.39M | 33.55M |
| q, k, v, o | 8.39M | 16.78M | 67.11M |
| semua proyeksi | 19.99M | 39.98M | 159.91M |

Memori adapter r=16 semua proyeksi:

| Komponen | MB |
|---|---:|
| Bobot fp16 | 80.0 |
| Gradient fp16 | 80.0 |
| Optimizer Adam | 319.8 |
| **Total** | **479.7** |

Dibandingkan 112 GB state full fine-tuning -> **233x lebih hemat**.

Merge: adapter dengan `B=0` memberikan selisih 0.00e+00 terhadap base. Setelah
B diisi, output unmerged vs merged berbeda 1.44e-15 (presisi float). Secara
matematis identik.

## 6. Quantization (`quantization_estimator.py`)

Matriks 4096x4096 dengan 3 outlier ekstrem:

| Skema | Relative MSE | Max error | Cosine |
|---|---:|---:|---:|
| fp16 (referensi) | 0 | 0.0000 | 1.00000 |
| int8 per-tensor | 8.95e-05 | 0.0059 | 0.99996 |
| int8 per-group 128 | 4.22e-05 | 0.0057 | 0.99998 |
| int4 per-tensor | 2.55e-02 | 0.0780 | 0.98747 |
| int4 per-group 128 | 1.38e-02 | 0.0569 | 0.99318 |
| int4 per-group 32 | 9.41e-03 | 0.0465 | 0.99533 |

Penyebab int4 per-tensor gagal: skala 0.214 vs magnitudo bobot rata-rata
0.0159 (rasio 13x), sehingga sebagian besar bobot dibulatkan ke 0.
Per-group memperkecil skala lokal.

Memori bobot 7B:

| Presisi | GB | Hemat vs fp16 |
|---|---:|---:|
| fp32 | 28.0 | -100% |
| fp16 | 14.0 | 0% |
| int8 | 7.0 | 50% |
| int4 | 3.5 | 75% |

## 7. RAG vs fine-tuning (`rag_vs_finetuning.py`)

Biaya per request (qwen-plus, ilustrasi):

| Skenario | Tanpa RAG | Dengan RAG | Selisih |
|---|---:|---:|---:|
| FAQ produk statis | $0.000280 | $0.000928 | +$0.000648 |
| Gaya brand | $0.000280 | $0.000328 | +$0.000048 |
| Harga & promo harian | $0.000280 | $0.001328 | +$0.001048 |
| Dokumen legal 200 halaman | $0.000376 | $0.003624 | +$0.003248 |

Titik impas fine-tuning:

| Skenario | Biaya FT | RAG per request | Impas |
|---|---:|---:|---:|
| FAQ produk statis | $261.60 | $0.000648 | 403.704 request |
| Gaya brand | $294.00 | $0.000048 | 6.125.000 request |
| Harga & promo harian | $261.60 | $0.001048 | 249.618 request |
| Dokumen legal | $348.00 | $0.003248 | 107.143 request |

Latensi dokumen legal:

| Komponen | RAG | Fine-tuned |
|---|---:|---:|
| Retrieval | 120 ms | 0 ms |
| Prefill | 4.020 ms | 20 ms |
| Decode | 3.750 ms | 3.750 ms |
| **Total** | **7.890 ms** | **3.770 ms** |

Artinya RAG 2x lebih lambat pada prompt 8k, tetapi bisa memperbarui fakta
tanpa training. Matriks keputusan berbobot: RAG 3.80, LoRA 2.95, RAG+LoRA 3.80
(seri; RAG dipilih karena implementasinya lebih cepat).

## Kesimpulan

1. "7B" pada MoE berarti total, bukan aktif. Selalu periksa keduanya sebelum
   memperkirakan biaya serving.
2. Memori training ~8x memori inference. LoRA memotong bagian terbesar dengan
   mengubah <1% parameter.
3. Perplexity hanya sebanding untuk tokenizer yang sama, dan loss rendah tidak
   menjamin jawaban benar.
4. Quantization 4-bit aman untuk model besar dengan group size kecil; per-tensor
   4-bit rusak oleh outlier.
5. Untuk pengetahuan yang berubah, RAG hampir selalu menang. Fine-tuning cocok
   untuk gaya, format, dan pola, bukan fakta.
6. DPO mengubah preferensi tanpa reward model terpisah, tetapi tetap butuh
   model referensi dan pasangan chosen/rejected yang berkualitas.

## Keterbatasan

- Harga training dan API adalah ilustrasi, bukan tarif aktual.
- Estimasi memori mengabaikan activation dan fragmentasi.
- Kualitas LoRA/quantization tidak diukur; hanya ukuran dan error numerik.
- Matriks keputusan memakai bobot subjektif yang ditulis di kode.
# Day 03 - Model Architecture, Training Concepts, Adaptation

Hari ini bukan soal membangun model, tapi soal menghitung **apa yang terjadi di
sekitar bobot**: bagaimana ia dilatih, apa arti loss, kenapa ada MoE, dan
bagaimana ia diadaptasi tanpa melatih ulang semuanya.

Batasan yang aku pegang: **tidak ada training atau fine-tuning di folder ini.**
Semua angka dihitung dari rumus dan simulasi kecil yang bisa dicek pakai
kalkulator. Satu-satunya "gradient step" adalah aritmetika satu parameter, hanya
untuk memperlihatkan bentuk update-nya.

## Tiga angka yang mengubah cara aku melihat fine-tuning

**1. State training itu 16 byte per parameter.**
Bobot fp16 (2) + gradient (2) + dua state Adam fp32 (8) + master weight fp32 (4).
Model 7B → ~112 GB, belum termasuk activation. Bukan karena modelnya besar,
tapi karena **Adam menyimpan dua salinan tambahan per parameter**.

**2. LoRA r=16 di semua proyeksi cuma 39.98M parameter (0.57%).**
Rumusnya `r × (d_in + d_out)` per modul. State-nya 480 MB, dibanding 112 GB
untuk full fine-tuning — sekitar **233x lebih kecil**. Karena `B` diinisialisasi
nol, adapter awalnya tidak mengubah output sama sekali; dan setelah training
`W' = W + (alpha/r)·BA` bisa di-merge sehingga tidak ada overhead inference.
Aku mengecek merge-nya: identik dengan versi unmerged, selisih **1.4e-15**.

**3. int4 per-tensor rusak, per-group selamat.**
Pada matriks dengan outlier, int4 per-tensor memberi relative MSE **2.55e-2**.
Skala per-tensor 13x lebih besar dari magnitudo bobot rata-rata, sehingga bobot
normal jatuh di bawah satu level kuantisasi dan dibulatkan ke nol. Dengan
per-group 128, MSE turun ke **1.38e-2**.

## Loss dan perplexity

Satu sequence panjang menghasilkan `n-1` contoh training: posisi `i` jadi input,
`i+1` jadi target. Karena itu pretraining efisien.

`loss = -log p(target)`, `perplexity = exp(loss)`. Perplexity bisa dibaca
sebagai "jumlah pilihan efektif yang dipertimbangkan model". Baseline acak
dengan vocab 50k = 50.000.

Catatan penting: perplexity **hanya sebanding kalau tokenizer-nya sama**. Ini
nyambung ke pelajaran Day 01 — ganti tokenizer, angka PPL tidak bisa
dibandingkan lagi.

## MoE

Setiap token hanya melewati top-k expert. Mixtral 8x7B: **46.70B total, 12.88B
aktif** per token (28%). Memori bobot tetap sebesar parameter total, tapi FLOPs
per token jauh lebih kecil.

## RAG vs fine-tuning

RAG menambah token di prompt; fine-tuning mengubah bobot. Aku hitung titik
impasnya untuk kasus FAQ: **~404.000 request**. Di bawah itu, fine-tuning lebih
murah; di atas itu, RAG yang menang.

Tapi biaya bukan satu-satunya kriteria. Untuk data yang berubah-ubah, RAG menang
secara default — bukan karena lebih murah, tapi karena fine-tuning untuk fakta
yang bisa berubah besok itu investasi yang salah.

## DPO, singkat

Membandingkan log-probabilitas chosen vs rejected **relatif terhadap model
referensi**, lalu melewatkannya ke sigmoid. Tidak butuh reward model terpisah
dan tidak butuh sampling RL. Bagian "relatif terhadap referensi" itu yang
membuatnya stabil — kalau pakai probabilitas absolut, model bisa "menang" dengan
cara menurunkan keduanya.

## Yang belum aku pahami

1. Kalau LoRA cuma 0.5% parameter, kenapa ia bisa mengubah gaya jawaban secara
   drastis? Di mana "gaya" itu sebenarnya disimpan?
2. Kapan MoE justru **lebih mahal** daripada dense saat serving? (petunjuk:
   memori total yang harus disiapkan)
3. Kuantisasi 4-bit "hampir sama" dengan fp16, tapi kenapa tidak semua model
   dirilis 4-bit?

Detail angka ada di [`results.md`](results.md).

## Lanjut

Aku sudah tahu isi model dan biaya melatihnya. Berikutnya: bagaimana model
menghasilkan token satu per satu, kenapa token pertama lambat, dan kenapa
konteks panjang menghabiskan memori. Itu Day 04.
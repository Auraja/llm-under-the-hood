# Hasil Eksekusi - Day 01: Tokenization

Dijalankan pada:

| Item | Nilai |
|---|---|
| Tanggal | 17 November 2026 |
| OS | Windows 11 (10.0.26300) |
| Python | 3.12.6 |
| NumPy | 2.2.6 |
| tiktoken | 0.14.0 |
| CPU/RAM | 12 core, 16 GB |

Semua eksperimen offline. tiktoken memakai cache lokal `cl100k_base`
(unduhan pertama ~3.7 detik, setelah itu instan).

## 1. BPE dari nol (`bpe.py`)

Corpus: 2.738 karakter, 229 kata unik. Setelah 40 merge:

| Metrik | Nilai |
|---|---|
| Merge rules | 40 |
| Vocabulary | 76 token |
| Merge teratas | `a`+`n` (94), `e`+`n` (77), `an`+`</w>` (53), `a`+`</w>` (45) |

Merge pertama langsung menemukan bahwa `</w>` (akhir kata) adalah simbol yang
sangat sering: kata berakhiran `n`, `a`, dan `i` mendominasi corpus. Corpus
Indonesia juga memunculkan prefiks/sufiks yang sering (`men`, `kan`, `di`) lebih
cepat daripada corpus Inggris pada jumlah merge yang sama.

Kompresi `"token bukan kata"`:

```
BPE lokal : 6 token  (token</w>, bu, kan</w>, k, at, a</w>)
tiktoken  : 4 token  (token, " bu", "kan", " kata")
```

Dua hal penting:

1. **Roundtrip berhasil** (`decode(encode(x)) == x`).
2. **BPE lokal lebih buruk** karena hanya dilatih pada 2.6 KB teks. Tokenizer
   produksi dilatih pada ratusan GB dan punya 100k token, sehingga bisa
   menyimpan kata utuh sebagai satu token.

Karakter di luar vocabulary (mis. emoji) ditolak secara eksplisit:

```
strict=True  -> ValueError: karakter tidak ada di vocabulary: '\U0001f600'
strict=False -> 2 id, karakter asing dibuang diam-diam
```

Ini keputusan desain: tokenizer produksi memakai byte-level fallback supaya
tidak ada teks yang hilang. Membuang karakter tanpa peringatan adalah bug yang
sulit dilacak, jadi default di sini adalah `strict=True`.

## 2. Perbandingan tokenizer (`tokenizer_comparison.py`)

| Teks | char | word | BPE lokal | tiktoken |
|---|---:|---:|---:|---:|
| `Halo dunia` | 10 | 2 | 8 | 4 |
| `Tokenisasi bukan sekadar memecah teks pada spasi.` | 49 | 7 | 29 | 16 |
| `LLM mempelajari pola dari data, bukan aturan tata bahasa.` | 57 | 9 | 34 | 20 |
| `def attention(q, k, v): return softmax(q @ k.T) @ v` | 51 | 10 | 36 | 17 |
| `Mempertanggungjawabkan` | 22 | 1 | 16 | 9 |
| `2026-11-17T09:00:00Z` | 20 | 1 | 3 | 13 |
| `    kode menjorok dengan     spasi aneh   ` | 42 | 5 | 21 | 12 |
| **TOTAL** | **251** | **35** | **147** | **91** |

Rasio kompresi: BPE lokal 1.71 char/token, tiktoken 2.76 char/token.

Temuan:

- Tokenizer lokal menang pada timestamp (`2026-...` = 3 token) hanya karena
  karakter tersebut kebetulan sering muncul bersamaan di corpus kecil.
- `Mempertanggungjawabkan`: tiktoken memecah menjadi morfem yang masuk akal
  untuk bahasa Indonesia (`M`, `emp`, `ert`, `ang`, `g`, `ung`, `jaw`, `ab`,
  `kan`) — jadi tokenizer Inggris pun sebagian "paham" sufiks `-kan`. BPE lokal
  menjadi 16 potongan yang lebih acak.
- Kalimat `Model bahasa mempelajari pola dari data.` (40 karakter) = 12 token
  tiktoken (3.33 char/token), sedangkan `The language model learns patterns
  from data.` (45 karakter) = 8 token (5.62 char/token). Teks Indonesia lebih
  mahal per karakter di tokenizer berbahasa Inggris.
- Kode Python lebih mahal daripada prosa: 17 token untuk 51 karakter.

## 3. Special token (`special_tokens.py`)

- `"halo<|im_end|>dunia"` -> 3 id. Tanpa pengenalan special, string itu
  dipecah menjadi 10 potongan dan struktur giliran hilang.
- Input user di-encode dengan `allow_special=False` -> tidak ada token kontrol
  yang bisa disuntikkan.
- Loss masking pada contoh chat: prompt 7 token + completion 5 token dari 12,
  sehingga hanya 5 token (42%) yang berkontribusi ke loss.
- Label padding `-100` dan causal shift bekerja sesuai harapan.

## 4. Chat template (`chat_template.py`)

Percakapan 4 pesan -> 256 karakter. Tanpa template hanya 133 karakter.
Overhead template = 123 karakter (**48%** dari prompt), ~30.8 karakter
(~7.7 token) per pesan.

| Kasus | Karakter | Perkiraan token |
|---|---:|---:|
| 20 pesan pendek (`ok`) | 607 | ~152 |
| 1 pesan panjang (`ok ` x20) | 94 | ~24 |

Isi sama, biaya 6x berbeda. Ini alasan menggabungkan pesan pendek atau
mengurangi jumlah turn bermanfaat.

Format alternatif untuk percakapan yang sama: ChatML 256 karakter, Llama-2
170 karakter. Token dan posisinya berbeda, walau isi percakapan identik.

`add_generation_prompt=True` menambahkan header assistant kosong di akhir:

| Versi | Jumlah id (tokenizer mainan) | Akhiran |
|---|---:|---|
| Tanpa header | 32 | `<|im_end|>` |
| Dengan header | 34 | `<|im_start|>assistant` |

Tanpa header, model harus menebak apakah gilirannya sudah dimulai atau masih
melanjutkan pesan user. Overhead-nya kecil (~2 token pada tokenizer mainan,
~6 token dengan heuristik 4 karakter/token), tetapi efeknya besar pada
konsistensi jawaban.

## 5. Context window (`context_window.py`)

Skenario: 28 pesan, 1.307 token, budget 220 token dengan **reserve 40 token**
untuk jawaban, sehingga hanya 180 token tersedia untuk prompt.

Setiap pesan dihitung `estimasi isi + 4 token overhead`. Angka 180 adalah batas
efektif, bukan 220 — inilah yang sering terlewat saat menghitung context window.

| Strategi | Pesan tersisa | Dibuang | Token | Utilisasi | Ingat "4271"? |
|---|---:|---:|---:|---:|---|
| `drop_oldest` | 4 | 24 | 177 | 98% | TIDAK |
| `keep_system` | 4 | 24 | 137 | 76% | TIDAK |
| `head_tail` | 6 | 22 | 168 | 93% | YA |
| `truncate_content` | 28 | 0 | 180 | 100% | YA |

Perbedaan strategi sekarang nyata:

- `drop_oldest` membuang pesan tertua termasuk system prompt, sehingga aturan
  dasar model hilang lebih dulu.
- `keep_system` mempertahankan system prompt, tetapi fakta di awal percakapan
  tetap hilang.
- `head_tail` menyimpan awal + akhir, mengorbankan tengah percakapan.
- `truncate_content` mempertahankan semua pesan dan memotong isi pesan latihan
  yang panjang, sehingga "4271" selamat — dengan konsekuensi teks di tengah
  menjadi terpotong-potong.

Budget ekstrem diuji agar tidak crash:

| Budget | Pesan | Token | Fits |
|---:|---:|---:|---|
| 1 | 0 | 0 | True |
| 4 | 0 | 0 | True |
| 5 | 1 | 5 | True |
| 12 | 1 | 12 | True |

Semua strategi diakhiri dengan pemotongan isi sebagai jaring pengaman,
sehingga hasilnya tidak pernah melebihi budget.

Biaya attention:

| Context | Pasangan attention | Relatif 1k |
|---:|---:|---:|
| 1.000 | 500.500 | 1x |
| 4.000 | 8.002.000 | 16x |
| 32.000 | 512.016.000 | 1.023x |
| 128.000 | 8.192.064.000 | 16.368x |

Konteks 4x lebih panjang = ~16x pekerjaan attention.

## 6. Biaya token (`token_cost.py`)

Satu request 1.200 prompt + 300 completion (harga ilustrasi, USD/1M token):

| Model | Input | Output | Total |
|---|---:|---:|---:|
| qwen-turbo | 0.05 | 0.20 | $0.000120 |
| qwen-plus | 0.40 | 1.20 | $0.000840 |
| qwen-max | 1.60 | 6.40 | $0.003840 |
| gpt-4o-mini | 0.15 | 0.60 | $0.000360 |

Skenario volume 5.000 request/hari dengan qwen-plus:

| Skenario | Prompt | Output | Total/bulan |
|---|---:|---:|---:|
| Chat biasa | 800 | 250 | $93 |
| RAG 3 dokumen | 6.000 | 400 | $432 |
| RAG + riwayat panjang | 24.000 | 500 | $1.530 |
| Agent banyak tool call | 60.000 | 800 | $3.744 |

Prompt caching (6.000 token, 5.500 cached): $0.002880 -> $0.001120, hemat 61%.
Pada 150.000 request/bulan, penghematannya $264.

Distribusi biaya berbeda per kasus:

- RAG 24k prompt + 500 output: 94% biaya di input.
- Chat kreatif 300 prompt + 2.000 output: 95% biaya di output.

## Kesimpulan

1. Token adalah unit ekonomi sekaligus unit komputasi. Semua keputusan
   (panjang prompt, bahasa, format) berdampak langsung ke biaya.
2. BPE adalah algoritma sederhana: frekuensi pasangan + merge berurutan.
   Perbedaan kualitas tokenizer terutama berasal dari data dan ukuran
   vocabulary, bukan dari kerumitan algoritma.
3. Special token dan chat template bukan detail kosmetik; keduanya menentukan
   keamanan (injeksi) dan efisiensi (overhead 48% pada contoh).
4. Tidak ada strategi pemotongan konteks yang bebas kerugian: membuang pesan
   menghilangkan fakta utuh, sedangkan memotong isi merusak kalimat. Sisakan
   reserve untuk output; budget prompt efektif selalu lebih kecil dari context
   window.
5. Prompt caching adalah optimasi dengan rasio manfaat/biaya terbaik ketika
   prefix benar-benar statis.

## Keterbatasan

- Harga di `token_cost.py` adalah ilustrasi, bukan tarif resmi terbaru.
- BPE lokal memakai level kata + `</w>`, bukan byte-level seperti tiktoken.
  Karakter di luar vocabulary dibuang saat `strict=False`; tokenizer produksi
  memakai byte fallback sehingga tidak ada kehilangan.
- Heuristik 4 karakter/token hanya perkiraan; untuk angka pasti pakai
  tokenizer model.
- Evaluasi "ingat 4271" hanya mendeteksi keberadaan string, bukan pemahaman
  model.
- Estimasi token di `context_window.py` tidak memperhitungkan tokenizer bahasa
  Indonesia yang biasanya lebih boros daripada teks Inggris.
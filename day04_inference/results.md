# Hasil Eksekusi - Day 04: Inference

| Item | Nilai |
|---|---|
| Tanggal | 17 November 2026 |
| Python / NumPy | 3.12.6 / 2.2.6 |
| Perangkat | CPU 12 core, 16 GB RAM, Windows 11 |
| API | qwen-plus @ DashScope Singapore (OpenAI-compatible) |

Semua eksperimen lokal berjalan tanpa API key. Bagian API dicatat terpisah
di bagian 8.

## 1. KV cache (`kv_cache.py`)

Incremental attention vs full attention, 40 token:

| Item | Nilai |
|---|---|
| Selisih maksimum | 6.66e-16 |
| Bentuk output | (40, 16) |

Kerja attention saat generate:

| Output token | Tanpa cache | Dengan cache | Rasio |
|---:|---:|---:|---:|
| 10 | 55 | 55 | 1.0x |
| 50 | 1.275 | 95 | 13.4x |
| 100 | 5.050 | 145 | 34.8x |
| 500 | 125.250 | 545 | 229.8x |
| 1.000 | 500.500 | 1.045 | 478.9x |
| 4.000 | 8.002.000 | 4.045 | **1.978x** |

Memori cache Qwen2.5-7B (28 layer, 4 KV head, head_dim 128, fp16):

| Context | Batch 1 | Batch 8 | Batch 32 |
|---:|---:|---:|---:|
| 1.000 | 0.06 GB | 0.46 GB | 1.84 GB |
| 8.000 | 0.46 GB | 3.67 GB | 14.68 GB |
| 32.000 | 1.84 GB | 14.68 GB | 58.72 GB |
| 128.000 | 7.34 GB | 58.72 GB | 234.88 GB |

Membaca cache saat decode (asumsi 50 GB/s):

| Context | Data per token | Waktu baca |
|---:|---:|---:|
| 1.000 | 0.06 GB | 1.15 ms |
| 32.000 | 1.84 GB | 36.70 ms |

## 2. Decoding (`decoding_simulation.py`)

Model mainan 7 token (tabel transisi), dengan EOS alami:

| Metode | Hasil | Total logprob | Rata-rata |
|---|---|---:|---:|
| Greedy | `the cat sat on mat` | -1.513 | -0.252 |
| Beam 1 | `the mat` | -1.484 | -0.495 |
| Beam 2 | `the cat sat on mat` | -1.513 | -0.252 |

Beam search memilih `the mat` karena total logprob sedikit lebih tinggi,
padahal rata-rata per token lebih buruk. Ini contoh klasik **length bias**:
logprob total selalu turun seiring panjang sequence.

Length normalization mengubah pemenang:

| Penalty | Beam teratas | Skor |
|---:|---|---:|
| 0.0 | `the mat` | -1.484 |
| 0.7 | `the cat sat on mat` | -0.387 |
| 1.0 | `the cat sat on mat` | -0.216 |

Sampling 5x per setelan:

| Setelan | Output unik |
|---|---:|
| T=0.5 | 2 |
| T=1.0 | 5 |
| T=1.0 top_k=2 | 4 |
| T=1.0 top_p=0.8 | 3 |
| T=2.0 | 5 |

Biaya: greedy 12 forward pass untuk 12 token, beam width 3 = 36, beam width
8 = 96.

## 3. Parameter generasi (`generation_parameters.py`)

Distribusi logits dengan 10 token (entropy menaik seiring T):

| Setelan | Entropy | Token teratas |
|---|---:|---|
| T=0.2 | 0.21 | the 0.95 |
| T=1.0 | 1.63 | the 0.43 |
| T=2.5 | 2.15 | the 0.22 |

Top-k dan top-p:

| Setelan | Token disimpan | Entropy |
|---|---:|---:|
| top_k=1 | 1 | 0.00 |
| top_k=3 | 3 | 0.99 |
| top_k=10 | 10 | 1.63 |
| top_p=0.5 | 2 | 0.65 |
| top_p=0.8 | 4 | 1.19 |
| top_p=0.95 | 7 | 1.53 |
| top_p=1.0 | 10 | 1.63 |

`top_k=1` identik dengan greedy. `top_p` beradaptasi: pada distribusi tajam
hanya 2 token disimpan, pada distribusi datar 7 token.

## 4. Prefix caching (`prefix_caching.py`)

Prompt 4.620 token: 4.500 statis (97.4%) + 120 dinamis.

| Skenario | Input | Output | Total | TTFT |
|---|---:|---:|---:|---:|
| Tanpa cache | $0.001848 | $0.000360 | $0.002208 | 2.310 ms |
| Cache hit | $0.000408 | $0.000360 | $0.000768 | 60 ms |

100.000 request/bulan:

| Hit rate | Biaya/bulan | Hemat |
|---:|---:|---:|
| 0% | $220.80 | 0% |
| 50% | $148.80 | 33% |
| 80% | $105.60 | 52% |
| 99% | $78.24 | 65% |

KV cache prefix 4.500 token kira-kira 258 MB. Menyimpannya 1 jam jauh lebih
murah daripada prefill berulang (layak secara biaya). Kendala nyata adalah
memori GPU, bukan biaya storage.

## 5. Speculative decoding (`speculative_decoding.py`)

Target p dan draft q memiliki overlap `alpha = 0.890`.

Simulasi 20.000 langkah, gamma=4:

| Metrik | Simulasi | Teori |
|---|---:|---:|
| Token per langkah | 4.02 | 4.01 |
| Semua proposal diterima | 63.0% | 60.9% |

Distribusi output vs target:

| Token | Target p | Simulasi |
|---:|---:|---:|
| 0 | 0.400 | 0.400 |
| 1 | 0.250 | 0.252 |
| 2 | 0.150 | 0.150 |
| 3 | 0.100 | 0.098 |
| 4 | 0.070 | 0.069 |
| 5 | 0.030 | 0.031 |

Selisih maksimum 0.002 -> distribusi target terjaga. Speculative decoding
bukan aproksimasi.

Speedup:

| gamma | c=0.02 | c=0.05 | c=0.10 | c=0.20 |
|---:|---:|---:|---:|---:|
| 1 | 1.85x | 1.80x | 1.72x | 1.58x |
| 2 | 2.58x | 2.44x | 2.24x | 1.92x |
| 4 | 3.72x | 3.35x | 2.87x | 2.23x |
| 8 | 5.09x | 4.22x | 3.28x | 2.27x |

Kualitas draft: alpha 0.99 -> 4.08x, alpha 0.89 -> 3.35x, alpha 0.75 ->
2.54x (semua pada c=0.05). Draft yang buruk bisa membuat decoding lebih
lambat dari baseline.

## 6. Memori inference (`memory_estimator.py`)

LLaMA-3 8B, fp16, batch 1:

| Context | Bobot | KV cache | Activation | Logits | Total |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 16.06 GB | 0.13 GB | 0.07 GB | 0.26 GB | 16.51 GB |
| 4.000 | 16.06 GB | 0.52 GB | 0.26 GB | 1.03 GB | 17.87 GB |
| 16.000 | 16.06 GB | 2.10 GB | 1.05 GB | 4.10 GB | 23.31 GB |
| 64.000 | 16.06 GB | 8.39 GB | 4.19 GB | 16.42 GB | 45.06 GB |
| 128.000 | 16.06 GB | 16.78 GB | 8.39 GB | 32.83 GB | 74.06 GB |

Porsi KV cache: 1% (1k), 9% (16k), 23% (128k). Logits dihitung untuk semua
posisi; implementasi nyata biasanya hanya menghitung posisi terakhir saat
decode, sehingga angka itu batas atas.

Efek kuantisasi bobot:

| Presisi | 1k ctx | 16k ctx | 128k ctx |
|---|---:|---:|---:|
| fp16 | 16.5 GB | 23.3 GB | 74.1 GB |
| int8 | 8.4 GB | 13.2 GB | 49.6 GB |
| int4 | 4.3 GB | 8.2 GB | 37.4 GB |

int4 memotong bobot 4x, tetapi KV cache tetap sama. Pada 128k, cache 16.8 GB
dan logits 32.8 GB menjadi bottleneck.

Batch maksimum (RAM 32 GB, fp16): 35 pada 1k ctx, 8 pada 4k, 2 pada 16k, 1
pada 32k. Untuk RAM 16 GB, model 8B fp16 (16.1 GB bobot) sudah tidak muat.

Perbandingan model:

| Model | Bobot | KV/token | Total @8k |
|---|---:|---:|---:|
| Qwen2.5 0.5B | 1.0 GB | 12 KB | 3.6 GB |
| LLaMA-3 8B | 16.1 GB | 128 KB | 19.7 GB |
| LLaMA-2 70B | 138.0 GB | 320 KB | 142.1 GB |

## 7. Benchmark lokal (`benchmark.py`)

Matmul NumPy (proxy prefill):

| Ukuran | Waktu | GFLOPS |
|---:|---:|---:|
| 512 | 2.6 ms | 103.7 |
| 1024 | 13.5 ms | 159.6 |
| 2048 | 77.7 ms | 221.0 |

Bandwidth copy: **18.1 GB/s**.

Estimasi batas atas decode:

| Model | Bobot | KV@4k | Token/detik |
|---|---:|---:|---:|
| 0.5B fp16 | 1.0 GB | 0.05 GB | 17.4 |
| 8B fp16 | 16.1 GB | 0.54 GB | 1.1 |
| 70B int4 | 34.5 GB | 1.34 GB | 0.5 |

Prefill 2.000 token pada 8B = 32.120 GFLOP -> ~174 detik di CPU ini
(compute-bound). Decode 100 token ~89 detik (bandwidth-bound). Ini
mengonfirmasi bahwa CPU hanya cocok untuk eksperimen mekanisme, bukan
interaksi real-time.

## 8. Hasil API

Endpoint Token Plan dan model yang dipakai:

```
base_url : https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
model    : deepseek-v4.1-flash-1017
```

Sebelumnya endpoint diisi `dashscope-intl.aliyuncs.com` dan model `qwen-plus`,
sehingga provider menolak dengan 401/404. Dua penyebabnya: key Token Plan
terikat ke endpoint **workspace** (`{workspace}.{region}.maas.aliyuncs.com`),
dan nama model publik seperti `qwen-plus` tidak tersedia di workspace.

Ping (`max_tokens=256`, temperature 0.2):

| Metrik | Nilai |
|---|---|
| Latency | 3.481 ms |
| Prompt token | 42 |
| Completion token | 108 (termasuk reasoning) |
| Reasoning | 347 karakter |
| Finish reason | stop |

Streaming:

| Metrik | Nilai |
|---|---|
| TTFT | 1.116 ms |
| Total | 1.147 ms |
| Output | `apel, jeruk, mangga` |
| Kecepatan | ~94,8 kata/detik |

Token prompt untuk permintaan yang sama (temperature 0):

| Bahasa | Prompt token | Jawaban |
|---|---:|---|
| Inggris | 39 | 129 karakter |
| Indonesia | 41 | kosong (budget habis untuk reasoning) |
| Jawa | 42 | kosong (budget habis untuk reasoning) |

Dengan `max_tokens=128`, model reasoning menghabiskan seluruh budget pada
`reasoning_content` sehingga `content` kosong dan `finish_reason="length"`.
Reasoning token **tetap dihitung dan ditagih** sebagai completion token.

System prompt 3.800 karakter -> 1.037 prompt token (~3,7 karakter/token),
konsisten dengan heuristik Day 01.

## Kesimpulan

1. KV cache mengubah biaya decode dari kuadratik menjadi linear. Ini optimasi
   terpenting di inference, dan tidak mengubah hasil sama sekali.
2. Token pertama lambat karena prefill; token berikutnya mahal karena harus
   membaca bobot + cache setiap langkah (bandwidth-bound).
3. Konteks panjang bukan hanya soal uang: pada 128k token, KV cache + logits
   mendominasi memori, dan kuantisasi bobot tidak membantu bagian itu.
4. Greedy/beam memaksimalkan probabilitas, bukan kualitas; beam punya length
   bias yang nyata (terbukti pada model mainan).
5. Speculative decoding memberi speedup tanpa mengubah distribusi, tetapi
   hanya jika draft model cukup akurat dan compute masih tersedia.
6. Prefix caching adalah optimasi termurah untuk sistem dengan prompt statis
   panjang: TTFT turun 97%, biaya input turun ~78% pada contoh ini.

## Keterbatasan

- Angka bandwidth/GFLOPS sangat bergantung pada mesin dan beban lain.
- Estimasi token/detik adalah batas atas teoretis, bukan hasil inference nyata.
- Model decoding mainan memakai tabel transisi, bukan neural network.
- Simulasi speculative memakai draft i.i.d.; draft nyata bergantung konteks.
- Angka latency API sangat bergantung pada jaringan dan antrean provider;
  satu run tidak cukup untuk kesimpulan performa.
- Model reasoning mengirim `reasoning_content` yang tidak selalu ditampilkan
  oleh UI, tetapi tetap dihitung dalam completion token.
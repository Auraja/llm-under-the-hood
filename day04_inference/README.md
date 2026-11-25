# Day 04 - Inference

Tiga hari sebelumnya isinya membangun dan menghitung. Hari ini kebalikannya:
model sudah jadi, dan pertanyaannya berubah jadi **kenapa lambat, kenapa mahal,
kenapa memori habis**.

Semua eksperimen lokal jalan tanpa API key. Bagian API cuma tambahan buat lihat
perilaku nyata (latency, usage, streaming).

## Dua fase yang menjelaskan hampir semua keluhan soal kecepatan

Orang biasanya bilang "LLM lambat" tanpa membedakan dua hal yang sangat beda:

| | **Prefill** | **Decode** |
|---|---|---|
| Yang dikerjakan | seluruh prompt diproses sekaligus | satu token per langkah |
| Terbatas oleh | compute (matmul besar) | **bandwidth memori** |
| Terasa sebagai | waktu tunggu sebelum token pertama (TTFT) | kecepatan mengetik |
| Naik saat konteks panjang? | ya, kuadratik | ya, karena cache makin besar dibaca |

Decode lambat bukan karena kurang CPU. Setiap token harus **membaca seluruh
bobot dan seluruh KV cache dari RAM**. Itu pekerjaan memindahkan data, bukan
menghitung.

## KV cache: dari O(T²) jadi O(T)

Key dan value token lama tidak berubah, karena causal mask melarangnya melihat
masa depan. Jadi tidak ada gunanya menghitung ulang.

Aku mengeceknya bukan dengan argumen, tapi dengan membandingkan hasil:
incremental attention identik dengan full attention, **selisih 6.66e-16** (level
epsilon float64).

Kerja yang dihemat, output 4.000 token:

```
tanpa cache : 8.002.000 operasi pasangan
dengan cache:     4.045
rasio       : ~1978x
```

Ukuran cache per token: `2 × n_layers × n_kv_heads × head_dim × bytes`.
Untuk LLaMA-3 8B (GQA-8, fp16) ≈ **128 KB per token** — artinya 128k token
menghabiskan ±16 GB hanya untuk cache, satu sequence.

Ini juga alasan GQA di Day 02 penting: bukan menghemat parameter, tapi
memotong cache 4x.

## Prefix caching: hemat yang paling gampang, tapi rapuh

Kalau banyak request berbagi prefix identik (system prompt, tools, dokumen),
provider bisa melewati prefill.

```
prefix statis 4.500 token (97.4% prompt) + suffix dinamis 120 token
tanpa cache : $0.002208, TTFT 2.310 ms
cache hit   : $0.000768, TTFT    60 ms
```

Syaratnya kejam: prefix harus **di awal**, **identik sampai karakter**, dan
belum kedaluwarsa. Yang paling sering merusak tanpa sadar:

- system prompt diubah satu karakter → cache miss total
- timestamp atau ID request di awal prompt → miss setiap request
- urutan definisi tool diacak → miss walau isinya sama
- system prompt ditaruh di tengah percakapan → memang bukan prefix

## Sampling: mengubah cara memilih, bukan pengetahuan

- **temperature** mengubah bentuk distribusi (tajam ↔ datar)
- **top-k** memotong ke k token teratas
- **top-p** memotong pada massa probabilitas
- **repetition penalty** menurunkan logit token yang sudah muncul

Tidak satu pun menambah pengetahuan model. Semuanya cuma mengubah cara memilih
dari distribusi yang sama.

## Speculative decoding: menebak banyak, verifikasi sekali

Draft model mengusulkan `gamma` token, target model memverifikasi sekaligus.
Token diterima dengan peluang `min(1, p/q)`; kalau ditolak, sample dari residual
`max(0, p−q)`.

Yang penting: distribusinya **tetap sama dengan target model**. Speedup bukan
dari aproksimasi, tapi dari memanfaatkan compute yang menganggur.

```
gamma=4, c=0.05:
  draft sangat mirip (alpha 0.99) -> 4.08x
  draft mirip       (alpha 0.89) -> 3.35x
  draft lemah       (alpha 0.75) -> 2.54x
```

Kalau draft-nya jelek, decoding bisa lebih lambat dari baseline. Jadi bukan
gratis.

## Memori: urutan yang harus dihitung

`weights + kv_cache + activations + logits`

Kuantisasi hanya menyentuh bagian pertama. Di konteks panjang, cache bisa
mendominasi — dan itu tidak bisa dikuantisasi dengan cara yang sama.

## Catatan API (pengalaman nyata, bukan dokumentasi)

`api_client.py` membaca key dari `.env` dan tidak pernah mencetak key penuh.
Dua jebakan yang aku kena sendiri saat pakai Alibaba Token Plan:

1. **Endpoint bukan `dashscope-intl`.** Token Plan memakai endpoint per
   workspace: `https://{workspace_id}.ap-southeast-1.maas.aliyuncs.com/...`.
   Pakai endpoint publik hasilnya 401.
2. **Nama model berbeda.** `qwen-plus`/`qwen-max` itu nama DashScope publik;
   workspace Token Plan hanya menerima model seperti `qwen3.8-flash`,
   `deepseek-v4.1-flash-1017`, `glm-5.3`, atau `auto`.

Model reasoning mengirim proses berpikir di `reasoning_content`. Token
reasoning **ikut dihitung sebagai completion token** — kalau `max_tokens`
terlalu kecil, jawaban finalnya kosong dengan `finish_reason="length"`.
Naikkan ke ~256.

## Angka

Detail di [`results.md`](results.md).

| Yang diukur | Hasil |
|---|---|
| Incremental vs full attention | selisih 6.66e-16 |
| Kerja attention 4.000 token output | 8.002.000 → 4.045 (~1978x) |
| KV cache LLaMA-3 8B | 128 KB/token → 0.5 GB @4k, 16 GB @128k |
| Prefix caching | $0.002208 → $0.000768; TTFT 2.310 ms → 60 ms |
| Speculative (gamma=4, alpha=0.89) | 3.35x |
| Matmul NumPy di CPU ini | 103–221 GFLOPS |
| Bandwidth copy | 18.1 GB/s saat dicatat |

Catatan kejujuran: waktu aku jalankan ulang `benchmark.py` beberapa hari
kemudian, bandwidth-nya keluar **12.1 GB/s**, bukan 18.1. Mesin yang sama,
beban lain. Jadi angka di tabel itu urutan besarnya, bukan konstanta.

## Yang belum aku pahami

1. Kenapa KV cache tidak bisa dipakai ulang kalau system prompt diubah satu
   karakter? (secara teknis: karena posisi token bergeser — tapi kenapa tidak
   ada cara mem-"geser" cache-nya?)
2. Kalau speculative decoding tidak mengubah distribusi, dari mana speedup-nya
   kalau GPU sudah jenuh?
3. Kapan context window besar justru **tidak** layak secara ekonomi, walaupun
   modelnya sanggup?
4. Kenapa batch besar tidak selalu menaikkan throughput pada MoE?

## Selesai: Day 01-04

```
token -> attention -> block -> model -> loss -> adaptasi -> inference
```

Setiap lapisan menambah jenis biaya yang berbeda:

- token menentukan **harga**
- attention menentukan **kompleksitas**
- KV cache menentukan **memori**
- bandwidth menentukan **kecepatan**

Dan yang paling sering dilupakan: keempatnya bertemu di satu angka — jumlah
token dalam context window.
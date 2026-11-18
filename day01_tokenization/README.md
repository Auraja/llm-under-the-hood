# Day 01 - Tokenization

## Tujuan

Memahami bahwa model tidak pernah melihat teks, hanya urutan id. Semua biaya,
batas konteks, dan perilaku "model salah hitung huruf" berakar di lapisan ini.

Setelah hari ini, kamu bisa menjawab:

- Kenapa jumlah token bukan jumlah kata dan bukan jumlah karakter?
- Bagaimana BPE menemukan potongan kata yang sering muncul?
- Kenapa special token harus satu id, dan apa risikonya kalau tidak?
- Kenapa percakapan panjang lebih mahal daripada satu pesan panjang?
- Bagaimana strategi memotong riwayat saat melewati context window?

## Eksperimen

| File | Yang dilakukan |
|---|---|
| `bpe.py` | Melatih BPE dari nol (Counter + merge loop), lalu encode/decode |
| `tokenizer_comparison.py` | Membandingkan karakter, kata, BPE lokal, dan tiktoken |
| `special_tokens.py` | BOS/EOS/PAD, loss masking, prompt injection lewat token kontrol |
| `chat_template.py` | Meratakan pesan chat jadi satu string, menghitung overhead |
| `context_window.py` | Simulasi overflow dan empat strategi pemotongan |
| `token_cost.py` | Menghitung biaya input/output/cache dari jumlah token |

Jalankan:

```bash
python day01_tokenization/bpe.py
python day01_tokenization/tokenizer_comparison.py
python day01_tokenization/special_tokens.py
python day01_tokenization/chat_template.py
python day01_tokenization/context_window.py
python day01_tokenization/token_cost.py
```

Tidak ada file di hari ini yang membutuhkan API key. `tokenizer_comparison.py`
memakai tiktoken; kalau file encoding belum ter-cache dan mesin offline,
bagian tiktoken dilewati dan sisanya tetap berjalan.

## Konsep inti

### BPE

1. Mulai dari karakter.
2. Hitung pasangan yang paling sering muncul.
3. Gabungkan menjadi simbol baru, catat aturannya.
4. Ulangi. Encode = terapkan aturan sesuai urutan rank.

Implementasi lokal memakai penanda `</w>` (akhir kata). Tokenizer produksi
memakai byte-level BPE dengan regex pre-tokenization, tetapi prinsipnya sama.

Satu detail yang penting untuk produksi: `encode()` menolak karakter yang tidak
pernah dilihat saat training (`strict=True`). Tokenizer nyata memakai byte
fallback; tanpa itu, emoji atau aksara baru akan hilang diam-diam.

### Special token dan masking

- `<|im_start|>` dan `<|im_end|>` harus dipetakan ke satu id.
- Input user tidak boleh di-encode dengan `allow_special=True`, kalau tidak
  user bisa menyuntikkan penanda peran.
- Saat training chat, token prompt dan padding diberi label `-100` supaya
  tidak ikut dihitung dalam loss.

### Chat template dan context window

Percakapan diratakan menjadi satu sequence. Setiap pesan menambah overhead
(role + newline + penanda akhir). Saat memanggil model chat, gunakan
`add_generation_prompt=True` agar header assistant kosong ditambahkan di akhir;
tanpa itu model harus menebak apakah gilirannya sudah dimulai.

Kalau melebihi budget, pilih strategi: buang yang terlama, selalu pertahankan
system prompt, simpan awal+akhir, atau potong isi pesan. Sisakan juga
`reserve_tokens` untuk output — context window dipakai bersama oleh prompt dan
jawaban, jadi budget prompt selalu lebih kecil dari angka context window.

Tidak ada strategi yang tanpa biaya: `drop_oldest` bisa menghilangkan fakta
penting (bahkan system prompt), `head_tail` menyisakan celah di tengah,
`truncate_content` merusak kalimat.

## Hasil

Angka lengkap ada di [`results.md`](results.md). Ringkasnya:

- BPE lokal dengan 40 merge menghasilkan vocabulary 87 token.
- Frasa `"a token is not a word"`: 21 karakter, 6 token tiktoken, 10 token
  BPE lokal.
- Tokenizer Inggris (cl100k) mencapai ~3.4 karakter/token; BPE lokal 1.4.
- Overhead template ChatML ~48% dari prompt percakapan pendek.
- Percakapan 1.307 token harus dipotong ke 180 token efektif (budget 220 minus
  reserve 40 untuk jawaban). `head_tail` dan `truncate_content` mempertahankan
  angka penting di awal; `drop_oldest`/`keep_system` tidak.
- Attention atas 128k token = ~16.000x pekerjaan dibanding 1k token.

## Pertanyaan untuk didiskusikan

1. Kalau tokenizer A lebih efisien untuk bahasa Indonesia, apa efeknya pada
   biaya dan pada kualitas jawaban?
2. Kenapa `max_tokens` tidak bisa menyelamatkan prompt yang sudah melebihi
   context window?
3. Kalau system prompt 90% dari prompt, optimasi mana yang paling masuk akal:
   memendekkan system prompt atau mengaktifkan prompt caching?

## Lanjut ke Day 02

Sekarang teks sudah menjadi urutan id. Pertanyaan berikutnya: bagaimana urutan
itu diproses menjadi representasi yang saling terkait? Itu attention.
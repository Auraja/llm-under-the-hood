# Day 01 - Tokenization

Hari ini aku mau tahu apa yang sebenarnya terjadi antara "aku mengetik kalimat"
dan "model mulai menghitung". Ternyata jawabannya bukan huruf, bukan kata, tapi
sesuatu di antaranya yang dipilih berdasarkan frekuensi.

Semua kode di folder ini jalan di CPU, tanpa API key, tanpa unduh model.

## Yang aku kerjakan, urutannya

Urutan ini bukan urutan yang nyaman, tapi urutan yang masuk akal buat aku:
lihat dulu masalahnya, baru mekanismenya.

1. **`tokenizer_comparison.py`** — membandingkan empat cara memotong teks:
   per karakter, per kata, BPE lokal, dan tiktoken. Di sini kelihatan bahwa
   jumlah token bukan sifat teks, tapi sifat teks + tokenizer.
2. **`bpe.py`** — BPE dari nol, cuma `Counter` dan loop merge. Tidak ada
   library, supaya kelihatan bahwa idenya memang sesederhana itu.
3. **`special_tokens.py`** — BOS/EOS/PAD, plus kenapa input user tidak boleh
   di-encode dengan special token aktif.
4. **`chat_template.py`** — percakapan diratakan jadi satu string, dan berapa
   token yang hilang cuma untuk format.
5. **`context_window.py`** — simulasi konteks penuh dan empat cara memotongnya.
6. **`token_cost.py`** — dari mana angka tagihan.

```bash
python day01_tokenization/bpe.py
python day01_tokenization/tokenizer_comparison.py
```

## BPE dalam satu paragraf

Mulai dari karakter, tandai akhir kata dengan `</w>`, hitung pasangan yang
paling sering muncul, gabungkan, ulangi. Encode = terapkan aturan merge yang
sama dengan urutan yang sama. Aturan yang dipelajari lebih awal menang.

Implementasi di sini level kata. Tokenizer produksi pakai byte-level BPE dengan
regex pre-tokenization, tapi prinsipnya sama.

Efek sampingnya yang menarik: karena yang digabung adalah yang **sering**,
hasilnya mirip morfologi. Corpus Indonesia menghasilkan merge `m`+`en` → `men`
dan `k`+`an</w>` → `kan</w>` tanpa satu baris pun kode linguistik.

## Yang bikin aku koreksi tulisan sendiri

Awalnya aku menulis bahwa tokenizer lokal "menang" pada timestamp
(`2026-11-17T09:00:00Z` = 3 token). Setelah dicek, ternyata menangnya karena
**18 dari 20 karakter dibuang** — corpus latihan tidak punya digit, dan
`strict=False` membuangnya diam-diam.

Artinya rasio kompresi yang aku tulis sebelumnya (1.71 char/token) salah: dia
membagi panjang teks mentah dengan token dari teks yang sudah kehilangan isi.
Angka yang jujur **1.21 char/token**, dari 178 karakter yang benar-benar
ter-encode. Sekarang tabelnya punya kolom `hilang` supaya kesalahan ini tidak
terulang.

Pelajaran yang aku catat: token yang lebih sedikit belum tentu lebih baik —
bisa jadi ada informasi yang hilang di jalan.

## Angka lain yang aku catat

- Corpus 2.738 karakter, 40 merge → vocabulary 76 token.
- `"token bukan kata"`: 16 karakter, 6 token BPE lokal, 4 token tiktoken.
- tiktoken cl100k ≈ 2.76 char/token pada 7 teks uji; BPE lokal 1.21.
- Teks Indonesia lebih mahal di tokenizer berbahasa Inggris: kalimat 40
  karakter jadi 12 token (3.33 char/token), padahal kalimat Inggris 45
  karakter cuma 8 token (5.62 char/token).
- Template ChatML menambah overhead ~48% pada percakapan 4 pesan pendek.
- 20 pesan pendek `"ok"` = 607 karakter, isi yang sama dalam 1 pesan = 94
  karakter. Enam kali lebih mahal.
- Percakapan 1.307 token harus dipotong ke 180 token (budget 220 − reserve 40).
  Hanya `head_tail` dan `truncate_content` yang masih menyimpan fakta di awal.
- Prompt caching memotong biaya input 61% saat 92% prompt-nya statis.
- Attention atas 128k token ≈ 16.000x pekerjaan dibanding 1k token.

Detail lengkap ada di [`results.md`](results.md).

## Yang masih mengganjal

1. `drop_oldest` membuang system prompt lebih dulu. Kalau system prompt wajib
   ada, kenapa strategi ini masih dipakai di banyak library?
2. Overhead template bisa 48% pada percakapan pendek. Apakah model benar-benar
   butuh semua penanda itu, atau itu warisan desain?
3. Kalau tokenizer yang lebih cocok untuk bahasa Indonesia bikin token lebih
   sedikit, apakah kualitas jawaban ikut naik, atau cuma tagihannya turun?

## Lanjut

Teks sudah jadi urutan id. Pertanyaan berikutnya: bagaimana urutan itu diproses
supaya token saling "melihat"? Itu Day 02.
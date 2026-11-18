# llm-under-the-hood

Belajar cara kerja LLM dengan nulis sendiri bagian-bagiannya: tokenizer, attention,
decoder block, loss, LoRA, KV cache, sampai cara sampling.

Nggak pakai model pretrained dan nggak ada training. Semua dijalanin di CPU, dan
setiap angka (jumlah token, parameter, memori, biaya) dihitung pakai rumus yang bisa
dicek ulang. Kalau ada bagian yang nggak bisa diverifikasi, aku tandain sebagai
ilustrasi.

## Progres

- [x] Day 01 - tokenization
- [ ] Day 02 - transformer (attention, rope, jumlah parameter)
- [ ] Day 03 - arsitektur model & cara adaptasi (LoRA, quantization, RAG)
- [ ] Day 04 - inference (KV cache, decoding, biaya)

## Batasan

- CPU only, tanpa GPU.
- Nggak download bobot pretrained, jadi nggak ada `from_pretrained(...)`.
- Nggak ada fine-tuning; efek LoRA/quantization/RAG dihitung, bukan dilatih.
- Library: NumPy, PyTorch (CPU), tiktoken, pytest.
- API cuma dipakai di Day 04 (opsional). Selain itu semua jalan tanpa key.

## Jalanin

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

PyTorch CPU aja biar downloadnya nggak berat:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Test semua eksperimen (tanpa internet):

```bash
python -m pytest -q
```

Atau jalanin satu file, misal:

```bash
python day01_tokenization/bpe.py
```

## API key (opsional)

Day 04 punya bagian yang manggil API. Tanpa key, bagian itu otomatis di-skip dan
sisanya tetap jalan.

```bash
cp .env.example .env
# isi DASHSCOPE_API_KEY
```

`.env` udah masuk `.gitignore`, jangan taruh key di kode atau di `.env.example`.
Default endpoint-nya Alibaba Cloud Model Studio (Singapore), bisa diganti lewat
`DASHSCOPE_BASE_URL`.

## Isi repo

```text
data/sample_corpus.txt
day01_tokenization/
├── bpe.py
├── tokenizer_comparison.py
├── special_tokens.py
├── chat_template.py
├── context_window.py
├── token_cost.py
├── README.md
└── results.md
tests/
└── test_tokenization.py
```

## Day 01 - angka yang keluar

Semua dijalanin di CPU (12 core, 16 GB RAM, Python 3.12). Detail lengkapnya ada di
`day01_tokenization/results.md`.

- Corpus 2.738 karakter (bahasa Indonesia), 40 merge rules → vocab 76 token.
- Frasa `"token bukan kata"`: 16 karakter, 6 token BPE lokal, 4 token tiktoken.
- cl100k ~2.76 karakter/token buat 7 teks uji (campur Indonesia + kode), BPE lokal 1.71.
- `"Mempertanggungjawabkan"`: 9 token tiktoken (memecah morfem `emp`/`ang`/`jaw`/`ab`/`kan`), 16 token BPE lokal.
- Template ChatML nambah overhead ~48% di percakapan pendek.
- Percakapan 1.307 token harus dipotong ke 180 token (budget 220 dikurangi 40 buat
  jawaban). Cuma `head_tail` dan `truncate_content` yang masih nyimpen fakta di awal.
- Prompt caching motong biaya input 61% kalau 92% prompt-nya statis.
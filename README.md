# llm-under-the-hood

Belajar cara kerja LLM dengan nulis sendiri bagian-bagiannya: tokenizer, attention,
decoder block, loss, LoRA, KV cache, sampai cara sampling.

Nggak pakai model pretrained dan nggak ada training. Semua dijalanin di CPU, dan
setiap angka (jumlah token, parameter, memori, biaya) dihitung pakai rumus yang bisa
dicek ulang. Kalau ada bagian yang nggak bisa diverifikasi, aku tandain sebagai
ilustrasi.

## Progres

- [x] Day 01 - tokenization
- [x] Day 02 - transformer (attention, rope, jumlah parameter)
- [x] Day 03 - arsitektur model & cara adaptasi (LoRA, quantization, RAG)
- [x] Day 04 - inference (KV cache, decoding, biaya)

## Batasan

- CPU only, tanpa GPU.
- Nggak download bobot pretrained, jadi nggak ada `from_pretrained(...)`.
- Nggak ada fine-tuning; efek LoRA/quantization/RAG dihitung, bukan dilatih.
- Library inti: NumPy, tiktoken, pytest. PyTorch + openai opsional
  (`requirements-optional.txt`).
- API cuma dipakai di Day 04 (opsional). Selain itu semua jalan tanpa key.

## Jalanin

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

Itu cukup untuk semua eksperimen lokal. Dua dependensi opsional dipisah supaya
tidak menyeret ~2 GB build CUDA:

```bash
pip install -r requirements-optional.txt     # torch (cross-check) + openai (API)
```

Tanpa file opsional itu, repo tetap jalan; `attention.py` cuma melewati bagian
perbandingan PyTorch, dan bagian API di Day 04 otomatis di-skip.

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
day02_transformer/
├── embeddings.py
├── positional_encoding.py
├── rope.py
├── attention.py
├── attention_variants.py
├── feed_forward.py
├── normalization.py
├── decoder_block.py
├── parameter_estimator.py
├── README.md
└── results.md
day03_model_architecture/
├── model_families.py
├── training_concepts.py
├── loss_perplexity.py
├── adaptation_methods.py
├── lora_estimator.py
├── quantization_estimator.py
├── rag_vs_finetuning.py
├── README.md
└── results.md
day04_inference/
├── api_client.py
├── generation_parameters.py
├── decoding_simulation.py
├── kv_cache.py
├── prefix_caching.py
├── speculative_decoding.py
├── memory_estimator.py
├── benchmark.py
├── README.md
└── results.md
tests/
├── test_tokenization.py
├── test_attention.py
├── test_architecture.py
└── test_inference.py
```

## Day 01 - angka yang keluar

Semua dijalanin di CPU (12 core, 16 GB RAM, Python 3.12). Detail lengkapnya ada di
`day01_tokenization/results.md`.

- Corpus 2.738 karakter (bahasa Indonesia), 40 merge rules → vocab 76 token.
- Frasa `"token bukan kata"`: 16 karakter, 6 token BPE lokal, 4 token tiktoken.
- cl100k ~2.76 karakter/token buat 7 teks uji (campur Indonesia + kode), BPE lokal
  1.21 — turun dari 1.71 setelah ketahuan 34 karakter dibuang diam-diam.
- `"Mempertanggungjawabkan"`: 9 token tiktoken (memecah morfem `emp`/`ang`/`jaw`/`ab`/`kan`), 16 token BPE lokal.
- Template ChatML nambah overhead ~48% di percakapan pendek.
- Percakapan 1.307 token harus dipotong ke 180 token (budget 220 dikurangi 40 buat
  jawaban). Cuma `head_tail` dan `truncate_content` yang masih nyimpen fakta di awal.
- Prompt caching motong biaya input 61% kalau 92% prompt-nya statis.

## Day 02 - angka yang keluar

Detail di `day02_transformer/results.md`.

- Estimasi parameter ketemu dengan angka publik: LLaMA-2 7B **6.74B** (selisih
  0.02%), 70B **68.98B** (0.00%), Qwen2.5 7B **7.62B** (0.06%).
- Di LLaMA-2 7B, **64.2%** parameter ada di FFN, **31.9%** di attention.
- GQA-8 memotong KV cache **4x**: 4.096 MB jadi 1.024 MB (d_model 4096, 32 head).
  Yang dihemat cache-nya, bukan parameter utamanya.
- Attention NumPy vs PyTorch: selisih **2.2e-16**.
- Bobot LLaMA-2 70B: 138 GB fp16, 34.5 GB int4 — belum termasuk KV cache.

## Day 03 - angka yang keluar

Detail di `day03_model_architecture/results.md`.

- LoRA r=16 di semua proyeksi: **39.98M** parameter (0.57% dari 7B), memori
  state 479.7 MB vs 112 GB full fine-tuning → **233x** lebih hemat.
- Merge LoRA identik dengan versi unmerged (selisih 1.4e-15).
- int4 per-tensor rusak oleh outlier (relative MSE 2.55e-2); per-group 128
  turun ke 1.38e-2.
- Mixtral 8x7B: 46.70B total, hanya **12.88B aktif** per token (28%).
- Titik impas RAG vs fine-tuning untuk FAQ: **~404.000 request**.

## Day 04 - angka yang keluar

Detail di `day04_inference/results.md`.

- Incremental attention identik dengan full attention (selisih **6.66e-16**).
- Output 4.000 token: kerja attention turun dari 8.002.000 pasangan jadi 4.045
  (~**1978x**).
- KV cache LLaMA-3 8B: 128 KB/token → 0.5 GB @4k, **16 GB @128k**.
- Prefix caching: TTFT **2.310 ms → 60 ms**, biaya per request $0.002208 → $0.000768.
- Speculative decoding gamma=4, alpha=0.89: **3.35x**; alpha=0.75 cuma 2.54x.
- CPU ini: 103–221 GFLOPS matmul, ~18 GB/s bandwidth → model 8B cuma ~1 token/detik.

## Lisensi

MIT — bebas dipakai, diubah, dan dibagikan, asal copyright-nya tetap ikut.
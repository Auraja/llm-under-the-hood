# llm-under-the-hood

Repo buat belajar cara kerja LLM dari dalam. Bukan proyek produk — tujuannya biar
nggak cuma bisa manggil API, tapi ngerti apa yang terjadi di belakangnya.

## Rencana

Ikut materi Day 01-04, tapi tiap konsep dibikin versi kodenya sendiri dan dijalanin
beneran biar keluar angka, bukan cuma diagram.

- [ ] Day 01 - tokenization (teks jadi token, biaya, context window)
- [ ] Day 02 - transformer (attention, rope, jumlah parameter)
- [ ] Day 03 - arsitektur model & cara adaptasi (LoRA, quantization, RAG)
- [ ] Day 04 - inference (KV cache, decoding, biaya)

## Jalanin

```bash
pip install -r requirements.txt
```

## Catatan sementara

- BPE sendiri: corpus 2.649 karakter, 40 merge rules → vocab 87 token.
- Frasa `"a token is not a word"`: 21 karakter, 10 token di BPE lokal, 6 token di
  tiktoken. Jadi buat teks Inggris tokenizer produksi jauh lebih efisien, dan itu
  langsung kerasa di biaya per request.
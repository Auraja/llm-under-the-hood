"""Chat template: bagaimana percakapan diubah menjadi satu sequence token.

Model chat bukan model khusus. Di baliknya, daftar pesan
``[{"role": "system", ...}, {"role": "user", ...}]`` diratakan menjadi teks
dengan penanda peran (ChatML: <|im_start|>role ... <|im_end|>), lalu di-tokenize
seperti teks biasa.

Yang perlu dipahami:
- Setiap pesan punya overhead token (penanda role + newline + end).
- Overhead itu berlipat kalau riwayat percakapan panjang.
- Format berbeda (ChatML, Llama, Mistral) berarti token dan hasil berbeda untuk
  prompt yang isinya sama.
- Jangan menambahkan penanda peran secara manual dari input user; itu celah
  injeksi. Gunakan template milik model.

Jalankan: python day01_tokenization/chat_template.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from special_tokens import ToyTokenizer  # noqa: E402

SPECIALS = ["<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>"]


def render_chatml(
    messages: list[dict],
    empty_system: bool = True,
    add_generation_prompt: bool = False,
) -> str:
    """Ratakan daftar pesan menjadi satu string ChatML.

    - ``empty_system``: lewati pesan system yang isinya kosong.
    - ``add_generation_prompt``: tambahkan header assistant tanpa isi di akhir,
      seperti ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``.
      Tanpa header ini, model tidak tahu bahwa gilirannya sudah dimulai.
    """
    parts: list[str] = ["<|bos|>"]
    for msg in messages:
        content = msg["content"].strip()
        if msg["role"] == "system" and not content and empty_system:
            continue
        parts.append(f"<|im_start|>{msg['role']}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_llama2(messages: list[dict]) -> str:
    """Template gaya Llama 2: system prompt hanya dipakai pada turn user pertama."""
    parts: list[str] = []
    system: str | None = None
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"].strip()
        elif msg["role"] == "user":
            prefix = f"{system} " if system else ""
            parts.append(f"[INST] {prefix}{msg['content'].strip()} [/INST]")
            system = None
        elif msg["role"] == "assistant":
            parts.append(f" {msg['content'].strip()} </s>")
    return "<s>" + "".join(parts)


def render_plain(messages: list[dict]) -> str:
    """Baseline tanpa template: gabung isi pesan apa adanya."""
    return " ".join(m["content"].strip() for m in messages)


def token_count(text: str) -> int:
    """Perkiraan token yang deterministik: 1 token ~ 4 karakter.

    String kosong = 0 token. Untuk angka pasti, pakai tokenizer model
    (lihat tokenizer_comparison.py).
    """
    if not text.strip():
        return 0
    return max(1, round(len(text) / 4))


def main() -> None:
    tok = ToyTokenizer(SPECIALS)

    conversation = [
        {"role": "system", "content": "Kamu adalah asisten yang menjawab singkat."},
        {"role": "user", "content": "Apa itu attention?"},
        {"role": "assistant", "content": "Mekanisme yang membobot relevansi token lain."},
        {"role": "user", "content": "Kenapa perlu causal mask?"},
    ]

    print("=== 1. ChatML rendering ===")
    rendered = render_chatml(conversation)
    print(rendered.replace("\n", "\\n"))

    ids = tok.encode(rendered)
    print(f"\npanjang teks    : {len(rendered)} karakter")
    print(f"jumlah token    : {len(ids)} (tokenizer mainan)")
    print(f"perkiraan token : {token_count(rendered)} (heuristik 4 char/token)")

    print("\n=== 2. Overhead template ===")
    plain = render_plain(conversation)
    overhead_chars = len(rendered) - len(plain)
    print(f"isi pesan saja     : {len(plain)} karakter")
    print(f"dengan template    : {len(rendered)} karakter")
    print(f"overhead           : {overhead_chars} karakter "
          f"({overhead_chars / len(rendered):.0%} dari prompt)")
    print(f"overhead per pesan : {overhead_chars / len(conversation):.1f} karakter "
          f"~ {overhead_chars / len(conversation) / 4:.1f} token")

    print("\n=== 3. Banyak pesan pendek vs satu pesan panjang ===")
    short_msgs = [{"role": "user", "content": "ok"}] * 20
    long_msg = [{"role": "user", "content": "ok " * 20}]
    short_rendered = render_chatml(short_msgs)
    long_rendered = render_chatml(long_msg)
    print(f"20 pesan pendek  : {len(short_rendered):4d} karakter "
          f"(~{token_count(short_rendered)} token)")
    print(f"1 pesan panjang  : {len(long_rendered):4d} karakter "
          f"(~{token_count(long_rendered)} token)")
    print("isinya sama, biayanya beda: overhead template mendominasi percakapan pendek.")

    print("\n=== 4. Template alternatif (format Llama 2 chat) ===")
    llama = render_llama2(conversation)
    print(llama.replace("\n", "\\n"))
    print(f"\nperbandingan panjang prompt untuk percakapan yang sama:")
    print(f"  ChatML : {len(rendered):4d} karakter")
    print(f"  Llama2 : {len(llama):4d} karakter")
    print("karena formatnya beda, jumlah token dan posisi token juga beda.")

    print("\n=== 5. add_generation_prompt ===")
    without = render_chatml(conversation)
    with_prompt = render_chatml(conversation, add_generation_prompt=True)
    print(f"tanpa header assistant : {len(tok.encode(without)):>2} id, "
          f"diakhiri {without[-14:]!r}")
    print(f"dengan generation prompt: {len(tok.encode(with_prompt)):>2} id, "
          f"diakhiri {with_prompt[-22:]!r}")
    print("header kosong di akhir membuat model tahu gilirannya dimulai;")
    print("tanpa itu model harus menebak apakah harus menjawab atau melanjutkan user.")
    print(f"\nbiaya tambahan header: {token_count('<|im_start|>assistant\\n')} token (heuristik)")


if __name__ == "__main__":
    main()
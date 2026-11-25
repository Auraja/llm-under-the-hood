"""Client LLM API minimal (OpenAI-compatible) + pemuat .env.

Dipakai untuk eksperimen inference di Day 04. Semua skrip lain tetap bisa
dijalankan tanpa API key: fungsi jaringan hanya dipanggil kalau key tersedia.

Prinsip:
- API key HANYA dibaca dari `.env` / environment, tidak pernah ditulis di kode.
- Semua panggilan dibungkus supaya error provider tidak membuat skrip crash.
- ``chat()`` mengembalikan usage + latency agar bisa dianalisis.

Endpoint default: Alibaba Cloud Model Studio (Singapore, OpenAI-compatible).
``DASHSCOPE_BASE_URL`` bisa diganti ke provider lain yang kompatibel.

Jalankan: python day04_inference/api_client.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


# --------------------------------------------------------------------- env
def load_env(path: Path | str | None = None) -> dict:
    """Baca file .env tanpa dependensi wajib.

    - Mendukung komentar, nilai berkutip, dan ekspansi ``${VAR}``.
    - Variabel yang sudah ada di environment proses menang atas isi file.
    """
    path = Path(path) if path else ENV_PATH
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    def expand(match: re.Match) -> str:
        return values.get(match.group(1), os.environ.get(match.group(1), ""))

    for key, value in list(values.items()):
        values[key] = re.sub(r"\$\{(\w+)\}", expand, value)

    for key, value in values.items():
        if key in os.environ and os.environ[key]:
            values[key] = os.environ[key]
    return values


CONFIG = load_env()


def api_key() -> str:
    return CONFIG.get("DASHSCOPE_API_KEY") or CONFIG.get("OPENAI_API_KEY") or ""


def base_url() -> str:
    return CONFIG.get("DASHSCOPE_BASE_URL") or CONFIG.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL


def model_name() -> str:
    return CONFIG.get("DASHSCOPE_MODEL") or DEFAULT_MODEL


def has_api_key() -> bool:
    key = api_key()
    return bool(key) and not key.startswith("sk-ISI")


def mask_key(key: str) -> str:
    if not key:
        return "(kosong)"
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


def make_client():
    """Buat client OpenAI-compatible. Import openai dilakukan malas."""
    from openai import OpenAI

    return OpenAI(api_key=api_key(), base_url=base_url(), timeout=60.0)


# ------------------------------------------------------------------- calls
def chat(messages: list[dict], retries: int = 1, **params) -> dict:
    """Satu panggilan chat completion + pengukuran latency.

    Return: content, usage, latency_ms, finish_reason, model.
    """
    if not has_api_key():
        raise RuntimeError(
            "API key belum diisi. Salin .env.example menjadi .env lalu isi "
            "DASHSCOPE_API_KEY. Eksperimen lokal tetap bisa dijalankan tanpa key."
        )

    params.setdefault("model", model_name())
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = make_client().chat.completions.create(messages=messages, **params)
            latency_ms = (time.perf_counter() - started) * 1000
            choice = response.choices[0]
            usage = response.usage
            # model reasoning (mis. deepseek-v4.1) mengirim proses berpikir di
            # field non-standar reasoning_content, terpisah dari jawaban final.
            reasoning = getattr(choice.message, "reasoning_content", None)
            return {
                "content": choice.message.content or "",
                "reasoning": reasoning or "",
                "latency_ms": latency_ms,
                "finish_reason": choice.finish_reason,
                "model": response.model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None),
                                         "cached_tokens", 0) or 0,
            }
        except Exception as exc:  # jaringan, rate limit, key salah, dll.
            last_error = exc
            if attempt < retries:
                time.sleep(1.5)

    raise RuntimeError(f"API call gagal: {type(last_error).__name__}: {last_error}")


def stream_chat(messages: list[dict], **params):
    """Generator potongan teks. ``first_chunk_latency`` diukur oleh pemanggil.

    Contoh:
        started = time.perf_counter()
        for piece in stream_chat(msgs):
            if first is None: ttft = ...
    """
    if not has_api_key():
        raise RuntimeError("API key belum diisi (lihat .env.example).")

    params.setdefault("model", model_name())
    try:
        stream = make_client().chat.completions.create(messages=messages, stream=True, **params)
    except Exception as exc:
        raise RuntimeError(f"API stream gagal: {type(exc).__name__}: {exc}") from exc
    for event in stream:
        if not event.choices:  # chunk terakhir bisa hanya berisi usage
            continue
        delta = event.choices[0].delta
        if delta and delta.content:
            yield delta.content


def main() -> None:
    print("=== Konfigurasi ===")
    print(f"env file     : {ENV_PATH} ({'ada' if ENV_PATH.exists() else 'TIDAK ADA'})")
    print(f"base_url     : {base_url()}")
    print(f"model        : {model_name()}")
    print(f"api key      : {mask_key(api_key())}")
    print(f"key terpakai : {has_api_key()}")

    if not has_api_key():
        print("\nEksperimen lokal di Day 04 tetap bisa dijalankan tanpa key:")
        print("  python day04_inference/generation_parameters.py")
        print("  python day04_inference/decoding_simulation.py")
        print("\nUntuk mencoba API: isi DASHSCOPE_API_KEY di .env (lihat .env.example).")
        return

    print("\n=== Ping API (max_tokens=256) ===")
    try:
        result = chat(
            [{"role": "user", "content": "Jawab dengan satu kalimat: apa itu token?"}],
            max_tokens=256,
            temperature=0.2,
        )
        if result["reasoning"]:
            print(f"reasoning    : {len(result['reasoning'])} karakter "
                  f"(tidak ditampilkan penuh)")
        print(f"jawaban      : {result['content'].strip()}")
        print(f"model        : {result['model']}")
        print(f"latency      : {result['latency_ms']:.0f} ms")
        print(f"token        : prompt={result['prompt_tokens']}, "
              f"completion={result['completion_tokens']}, total={result['total_tokens']}")
        print(f"cached token : {result['cached_tokens']}")
        print(f"finish reason: {result['finish_reason']}")
    except RuntimeError as exc:
        print(exc)

    print("\n=== Streaming ===")
    try:
        started = time.perf_counter()
        first = None
        pieces = []
        for piece in stream_chat(
            [{"role": "user", "content": "Sebutkan tiga kata, dipisah koma."}],
            max_tokens=256,
            temperature=0.2,
        ):
            if first is None:
                first = time.perf_counter() - started
            pieces.append(piece)
        total = time.perf_counter() - started
        print(f"teks         : {''.join(pieces).strip()}")
        print(f"TTFT         : {first * 1000:.0f} ms" if first is not None else "TTFT: n/a")
        print(f"total        : {total * 1000:.0f} ms")
        if first is not None and total > first:
            print(f"decode speed : {len(''.join(pieces).split()) / (total - first):.1f} kata/detik")
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    main()
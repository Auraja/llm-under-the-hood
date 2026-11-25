"""Benchmark: apa yang sebenarnya membatasi kecepatan inference.

Dua rezim yang berbeda:
- PREFILL (prompt panjang): compute-bound. Waktu ~ jumlah FLOP / GFLOPS mesin.
- DECODE (satu token per langkah): memory-bandwidth-bound. Setiap token harus
  membaca seluruh bobot model + KV cache dari RAM.

Bagian 1-3 mengukur mesin lokal (tanpa API, tanpa key).
Bagian 4 opsional mengukur API: latency, TTFT, token/detik, dan biaya.

Jalankan: python day04_inference/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

WEIGHT_BYTES_FP16 = 2


def best_time(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return min(times)


def matmul_gflops(size: int) -> tuple[float, float]:
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    seconds = best_time(lambda: a @ b, repeats=2)
    flops = 2 * size**3
    return seconds, flops / seconds / 1e9


def bandwidth_gbs(size_mb: int) -> float:
    n = size_mb * 1024 * 1024 // 4
    src = np.ones(n, dtype=np.float32)
    dst = np.empty_like(src)
    seconds = best_time(lambda: np.copyto(dst, src), repeats=3)
    return (src.nbytes * 2) / seconds / 1e9  # baca + tulis


def estimate_decode_tokens_per_s(bandwidth: float, n_params: int, kv_bytes: float) -> float:
    """Batas atas kecepatan decode: bandwidth / byte yang dibaca per token."""
    bytes_per_token = n_params * WEIGHT_BYTES_FP16 + kv_bytes
    return bandwidth * 1e9 / bytes_per_token


def main() -> None:
    print("=== 1. Matmul NumPy (proxy untuk prefill) ===")
    print(f"{'ukuran':>8} {'waktu':>10} {'GFLOPS':>9}")
    for size in [512, 1024, 2048]:
        seconds, gflops = matmul_gflops(size)
        print(f"{size:>8} {seconds * 1000:>8.1f} ms {gflops:>9.1f}")

    print("\n=== 2. Bandwidth memori (proxy untuk decode) ===")
    bw = bandwidth_gbs(128)
    print(f"copy 128 MB: ~{bw:.1f} GB/s")
    print("decode membaca bobot dan cache, bukan menghitung matmul besar.")

    print("\n=== 3. Estimasi decode di mesin ini ===")
    models = {
        "0.5B fp16": (494_000_000, 24 * 2 * 2 * 64 * 4096 * 2, 2),
        "8B fp16": (8_030_000_000, 32 * 2 * 8 * 128 * 4096 * 2, 2),
        "70B int4": (68_980_000_000, 80 * 2 * 8 * 128 * 4096 * 2, 0.5),
    }
    print(f"{'model':<12} {'bobot GB':>9} {'KV@4k GB':>10} {'token/s':>9}")
    for name, (params, kv_bytes, bytes_per_weight) in models.items():
        weight_bytes = params * bytes_per_weight
        tps = bw * 1e9 / (weight_bytes + kv_bytes)
        print(f"{name:<12} {weight_bytes / 1e9:>8.1f}G {kv_bytes / 1e9:>9.2f}G {tps:>9.1f}")
    print("\nangka ini batas atas teoretis; implementasi nyata lebih lambat.")
    print("perhatikan: model 70B int4 masih bisa 'muat' tetapi hanya beberapa token/detik.")

    print("\n=== 4. Prefill vs decode ===")
    _, gflops = matmul_gflops(1024)
    n_params = 8_030_000_000
    prompt_tokens = 2_000
    prefill_flops = 2 * n_params * prompt_tokens
    print(f"GFLOPS terukur       : {gflops:.1f}")
    print(f"prefill {prompt_tokens:,} token: {prefill_flops / 1e9:.0f} GFLOP -> "
          f"~{prefill_flops / (gflops * 1e9):.2f} s (compute-bound)")
    print(f"decode 100 token     : dibatasi bandwidth, ~100 / {estimate_decode_tokens_per_s(bw, n_params, 0):.1f} "
          f"= {100 / max(estimate_decode_tokens_per_s(bw, n_params, 0), 0.01):.1f} s")

    print("\n=== 5. Benchmark API (opsional) ===")
    try:
        from api_client import chat, has_api_key, model_name
        from prefix_caching import PRICE_IN, PRICE_OUT
    except ImportError:
        print("api_client tidak ditemukan; lewati.")
        return

    if not has_api_key():
        print("API key tidak ada -> bagian ini dilewati.")
        print("Isi DASHSCOPE_API_KEY di .env lalu jalankan ulang untuk melihat")
        print("latency, token/detik, dan biaya per 1.000 request.")
        return

    prompts = [
        ("pendek", "Sebutkan 3 warna dalam satu baris."),
        ("sedang", "Jelaskan perbedaan prefill dan decode dalam 4 kalimat."),
        ("panjang", "Buat ringkasan 6 poin tentang KV cache untuk engineer pemula."),
    ]
    print(f"model: {model_name()}")
    print(f"{'kasus':<8} {'latency ms':>11} {'prompt':>7} {'out':>5} {'token/s':>8} {'biaya $':>10}")
    for label, prompt in prompts:
        try:
            r = chat([{"role": "user", "content": prompt}], max_tokens=160, temperature=0.2)
        except RuntimeError as exc:
            print(f"{label:<8} gagal: {exc}")
            continue
        seconds = r["latency_ms"] / 1000
        tps = r["completion_tokens"] / seconds if seconds else 0
        cost = (r["prompt_tokens"] * PRICE_IN + r["completion_tokens"] * PRICE_OUT) / 1e6
        print(f"{label:<8} {r['latency_ms']:>11.0f} {r['prompt_tokens']:>7} "
              f"{r['completion_tokens']:>5} {tps:>8.1f} {cost:>10.6f}")

    print("\ncatatan: latency API mencakup jaringan + antrean provider,")
    print("bukan hanya compute model. Bandingkan angka antar-run, bukan absolut.")


if __name__ == "__main__":
    main()
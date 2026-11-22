"""Quantization: berapa memori yang dihemat, dan berapa akurasi yang hilang.

Simulasi di sini melakukan quantize -> dequantize (fake quantization), lalu
mengukur error. Ini persis yang "dirasakan" model ketika bobotnya disimpan
dengan bit lebih sedikit.

Skema yang disimulasikan:
- per-tensor : satu skala untuk seluruh matriks. Paling murah, paling rentan
               terhadap outlier.
- per-channel: satu skala per baris output. Jauh lebih tahan outlier.
- per-group  : satu skala untuk setiap ``group_size`` elemen. Standar di GPTQ/
               AWQ/NF4 untuk 4-bit.

Rumus memori bobot:
    bytes = n_params * bits / 8 + overhead_skala

Jalankan: python day03_model_architecture/quantization_estimator.py
"""

from __future__ import annotations

import numpy as np


def quantize_symmetric(x: np.ndarray, bits: int, group_size: int | None = None,
                       axis: int = -1) -> np.ndarray:
    """Quantize-dequantize simetris (zero point = 0).

    bits=8  -> level integer [-127, 127]
    bits=4  -> level integer [-7, 7]
    """
    qmax = 2 ** (bits - 1) - 1
    x = np.asarray(x, dtype=float)
    flat = x.reshape(-1, x.shape[-1]) if group_size else None

    if group_size is None:
        scale = np.abs(x).max(axis=axis, keepdims=True) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.round(x / scale), -qmax, qmax)
        return q * scale

    # per-group sepanjang dimensi terakhir
    rows, cols = flat.shape
    padded = int(np.ceil(cols / group_size)) * group_size
    padded_x = np.pad(flat, ((0, 0), (0, padded - cols)))
    groups = padded_x.reshape(rows, -1, group_size)
    scale = np.abs(groups).max(axis=-1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(groups / scale), -qmax, qmax) * scale
    return q.reshape(rows, padded)[:, :cols].reshape(x.shape)


def quant_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    # cast ke float64: penjumlahan 16 juta elemen dalam float32 kehilangan presisi
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(reconstructed, dtype=np.float64)
    diff = a - b
    mse = float(np.mean(diff**2))
    signal = float(np.mean(a**2))
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "mse": mse,
        "relative_mse": mse / (signal + 1e-12),
        "max_abs": float(np.abs(diff).max()),
        "cosine": min(cos, 1.0),
    }


def weight_memory_gb(n_params: int, bits: int, group_size: int | None = None) -> float:
    """Memori bobot + overhead skala (fp16 per grup)."""
    data = n_params * bits / 8
    overhead = 0.0
    if group_size:
        overhead = (n_params / group_size) * 2
    return (data + overhead) / 1e9


def main() -> None:
    rng = np.random.default_rng(0)
    n_out, n_in = 4096, 4096

    # Bobot "normal": sebagian besar kecil, dengan beberapa outlier besar
    W = rng.normal(0, 0.02, (n_out, n_in)).astype(np.float32)
    W[0, 0] = 1.5     # outlier ekstrem
    W[1, 5] = -1.2
    W[2, 9] = 0.9

    print("=== 1. Error quantization pada satu matriks 4096x4096 ===")
    print(f"{'skema':<26} {'rel MSE':>10} {'max |err|':>11} {'cosine':>9}")
    for label, bits, group in [
        ("fp16 (referensi)", 16, None),
        ("int8 per-tensor", 8, None),
        ("int8 per-group 128", 8, 128),
        ("int4 per-tensor", 4, None),
        ("int4 per-group 128", 4, 128),
        ("int4 per-group 32", 4, 32),
    ]:
        out = W.astype(np.float64) if bits >= 16 else quantize_symmetric(W, bits, group)
        e = quant_error(W, out)
        print(f"{label:<26} {e['relative_mse']:>10.2e} {e['max_abs']:>11.4f} {e['cosine']:>9.5f}")

    print("\n=== 2. Kenapa per-tensor int4 gagal? ===")
    tensor_scale = np.abs(W).max() / 7
    typical = np.abs(W).mean()
    print(f"skala per-tensor      : {tensor_scale:.5f}")
    print(f"magnitudo bobot rata2 : {typical:.5f}")
    print(f"rasio                 : {tensor_scale / typical:.0f}x")
    print("bobot normal jatuh di bawah 1 level kuantisasi -> dibulatkan ke 0.")
    print("per-group/ per-channel memperkecil skala lokal sehingga detail terjaga.")

    print("\n=== 3. Memori bobot 7B ===")
    n = 7_000_000_000
    print(f"{'presisi':<24} {'GB':>8} {'hemat vs fp16':>14}")
    base = weight_memory_gb(n, 16)
    for label, bits, group in [
        ("fp32", 32, None),
        ("fp16 / bf16", 16, None),
        ("int8", 8, None),
        ("int8 group 128", 8, 128),
        ("int4", 4, None),
        ("int4 group 128", 4, 128),
    ]:
        mem = weight_memory_gb(n, bits, group)
        print(f"{label:<24} {mem:>8.1f} {1 - mem / base:>13.0%}")

    print("\n=== 4. Biaya inference vs presisi ===")
    print("bobot 4-bit menghemat memori dan bandwidth, tetapi:")
    print("- dequantization menambah compute (kadang lebih lambat di CPU),")
    print("- akurasi turun paling terasa pada reasoning dan matematika,")
    print("- kualitas bervariasi per model; harus diukur, bukan diasumsikan.")
    print("aturan praktis: 8-bit hampir selalu aman, 4-bit biasanya aman untuk")
    print("model >=7B dengan group size 64-128, di bawah itu risiko naik.")

    print("\n=== 5. Quantization vs prunning vs distillation ===")
    rows = [
        ("quantization", "mengurangi bit per bobot", "hampir tidak ada perubahan arsitektur"),
        ("pruning", "menghapus bobot/head", "butuh retraining ringan"),
        ("distillation", "model kecil belajar dari model besar", "butuh training penuh"),
    ]
    print(f"{'metode':<14} {'cara':<34} {'catatan':<28}")
    for a, b, c in rows:
        print(f"{a:<14} {b:<34} {c:<28}")
    print("\nquantization paling murah untuk diterapkan karena tidak mengubah")
    print("arsitektur dan tidak memerlukan data training berlabel.")


if __name__ == "__main__":
    main()
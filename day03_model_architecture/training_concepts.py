"""Konsep training tanpa melakukan training.

Yang dibahas:
1. Objective causal language modeling: geser target satu token.
2. Satu langkah gradient descent secara matematis (hanya aritmetika, tidak ada
   loop optimasi pada model).
3. Memori training: parameter + gradient + optimizer state (Adam) + activation.
4. Learning rate schedule, batch accumulation, dan token budget.
5. DPO: alternatif RLHF yang belajar dari pasangan chosen/rejected.

Tidak ada model yang dilatih di file ini. Semua angka adalah perhitungan
eksplisit yang bisa diperiksa dengan kalkulator.

Jalankan: python day03_model_architecture/training_concepts.py
"""

from __future__ import annotations

import math


def shift_targets(tokens: list[str]) -> list[tuple[str, str]]:
    """Causal LM: input = token[:i], target = token[i]."""
    return list(zip(tokens[:-1], tokens[1:]))


def mse_loss(w: float, target: float) -> float:
    return (w - target) ** 2


def gradient(w: float, target: float) -> float:
    return 2 * (w - target)


def sgd_step(w: float, grad: float, lr: float) -> float:
    return w - lr * grad


def training_memory_bytes(n_params: int, bytes_per_weight: int = 2, use_adam: bool = True) -> dict:
    """Memori minimal untuk training (tanpa activation)."""
    weights = n_params * bytes_per_weight
    grads = n_params * bytes_per_weight
    optim = n_params * 8 if use_adam else 0  # fp32 m + fp32 v
    master = n_params * 4 if bytes_per_weight < 4 else 0  # fp32 master copy
    return {
        "weights": weights,
        "gradients": grads,
        "optimizer": optim,
        "master_weights": master,
        "total": weights + grads + optim + master,
    }


def cosine_lr(step: int, total_steps: int, warmup: int, peak: float, minimum: float = 0.0) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def dpo_loss(logp_chosen: float, logp_rejected: float,
             ref_logp_chosen: float, ref_logp_rejected: float, beta: float = 0.1) -> dict:
    """DPO loss.

    Nilai di dalam sigmoid adalah selisih *implied reward*:
        beta * [(log p_chosen - log p_ref_chosen) - (log p_rejected - log p_ref_rejected)]
    Loss = -log sigmoid(margin). Semakin besar margin, semakin kecil loss.
    """
    margin = beta * ((logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected))
    loss = -math.log(1 / (1 + math.exp(-margin)))
    prob = 1 / (1 + math.exp(-margin))
    return {"margin": margin, "loss": loss, "preference_prob": prob}


def main() -> None:
    print("=== 1. Causal LM objective ===")
    tokens = ["<bos>", "the", "cat", "sat", "down", "<eos>"]
    print(f"{'posisi':>6} {'input (<=i)':<28} {'target':>10}")
    for i, (inp, tgt) in enumerate(shift_targets(tokens)):
        prefix = " ".join(tokens[: i + 1])
        print(f"{i:>6} {prefix:<28} {tgt:>10}")
    print("\nsatu sequence panjang menghasilkan len(tokens)-1 contoh training.")
    print("inilah sebabnya 'pretraining' sangat efisien: setiap token menjadi target.")

    print("\n=== 2. Satu langkah gradient descent (aritmetika) ===")
    w, target, lr = 0.0, 3.0, 0.1
    loss = mse_loss(w, target)
    grad = gradient(w, target)
    new_w = sgd_step(w, grad, lr)
    print(f"loss   = (w - target)^2 = ({w} - {target})^2 = {loss:.4f}")
    print(f"gradien= 2(w - target)  = {grad:.4f}")
    print(f"update = w - lr * grad  = {w} - {lr} * {grad:.1f} = {new_w:.4f}")
    print(f"loss baru setelah update: {mse_loss(new_w, target):.4f}")
    print("di model nyata, 'w' berjumlah miliaran dan gradien dihitung dengan")
    print("backpropagation, tetapi bentuk update-nya identik.")

    print("\n=== 3. Memori training vs inference (7B, fp16) ===")
    n = 7_000_000_000
    train = training_memory_bytes(n)
    print(f"{'komponen':<18} {'GB':>8}")
    for key, value in train.items():
        print(f"{key:<18} {value / 1e9:>8.1f}")
    print(f"{'inference (bobot saja)':<18} {n * 2 / 1e9:>8.1f}")
    print(f"\ntraining butuh ~{train['total'] / (n * 2):.0f}x memori bobot inference,")
    print("belum termasuk activation (yang naik linear dengan batch x seq).")
    print("LoRA menghindari gradient + optimizer untuk bobot asli -> jauh lebih murah.")

    print("\n=== 4. Effective batch size ===")
    for micro, accum in [(1, 1), (4, 8), (8, 16)]:
        print(f"micro_batch={micro:>2} x grad_accum={accum:>2} -> effective batch={micro * accum}")
    print("batch besar menstabilkan gradient, tetapi memori activation per step")
    print("hanya ditentukan oleh micro_batch. Gradient accumulation menukar waktu dengan memori.")

    print("\n=== 5. Learning rate schedule ===")
    total, warmup, peak = 1000, 100, 3e-4
    print(f"{'step':>6} {'lr':>12}")
    for step in [0, 50, 99, 100, 250, 500, 750, 999]:
        print(f"{step:>6} {cosine_lr(step, total, warmup, peak):>12.2e}")
    print("warmup mencegah langkah besar di awal saat gradien masih liar;")
    print("cosine decay mengurangi lr menjelang akhir supaya konvergen halus.")

    print("\n=== 6. Token budget ===")
    dataset_tokens = 2_000_000_000
    epochs = 3
    print(f"dataset {dataset_tokens / 1e9:.1f}B token x {epochs} epoch = "
          f"{dataset_tokens * epochs / 1e9:.1f}B token diproses")
    print("Chinchilla rule-of-thumb: ~20 token per parameter untuk compute-optimal.")
    print(f"model 7B -> ~{7e9 * 20 / 1e9:.0f}B token. Lebih dari itu biasanya overtraining")
    print("dengan sengaja (model kecil, data banyak) untuk kualitas per parameter.")

    print("\n=== 7. DPO (Direct Preference Optimization) ===")
    cases = [
        ("model sudah benar", -8.0, -25.0, -12.0, -14.0),
        ("model ragu", -12.0, -13.0, -12.5, -12.0),
        ("model salah", -22.0, -9.0, -13.0, -11.0),
    ]
    print(f"{'skenario':<20} {'margin':>9} {'loss':>8} {'P(chosen)':>11}")
    for name, lc, lr_, rlc, rlr in cases:
        r = dpo_loss(lc, lr_, rlc, rlr, beta=0.1)
        print(f"{name:<20} {r['margin']:>9.3f} {r['loss']:>8.3f} {r['preference_prob']:>10.1%}")
    print("\nyang dibandingkan bukan probabilitas absolut, melainkan selisih terhadap")
    print("model referensi (hasil SFT). Ini mencegah model 'lupa' kemampuan dasarnya.")
    print("DPO tidak butuh reward model terpisah dan tidak butuh sampling RL.")


if __name__ == "__main__":
    main()
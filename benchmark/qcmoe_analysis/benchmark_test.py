#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC-MoE W4A16 / W8A16 Standalone Benchmark
==========================================
Reproduces the FlagGems Triton kernel vs vLLM-style FP16 MoE benchmark
for W4A16 (INT4) and W8A16 (INT8) quantization modes.

Usage:
    python benchmark_test.py              # run both W8A16 and W4A16
    python benchmark_test.py --w4a16      # run W4A16 only
    python benchmark_test.py --w8a16      # run W8A16 only

Output:
    - Terminal: formatted performance table
    - CSV: qcmoe_w8a16_data.csv   (W8A16 results)
           qcmoe_w4a16_data.csv   (W4A16 results)

Dependencies:
    pip install torch triton flag_gems
    # flag_gems must be installed; GemLite conda env has all deps pre-installed.
"""

import argparse
import csv
import time
import os
import sys

import torch
import torch.nn.functional as F

# ── GPU Info ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_SM   = torch.cuda.get_device_capability(0)
    print(f"[ENV] GPU: {GPU_NAME} (sm_{GPU_SM[0]}{GPU_SM[1]})")
    print(f"[ENV] PyTorch: {torch.__version__}")
    import triton
    print(f"[ENV] Triton: {triton.__version__}")
else:
    print("[ERROR] CUDA not available.")
    sys.exit(1)

# ── FlagGems imports ─────────────────────────────────────────────────────────
from flag_gems.ops.qcmoe import fused_moe, QuantConfig, QuantMode

# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-group weight quantization (supports INT4 and INT8)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_weights_moe(weights, w_nbits=8, group_size=128):
    """Per-group min-max quantization for MoE expert weights.

    Args:
        weights:    (E, n_out, k_in) — e.g. W1=(E,K,H), W2=(E,H,K)
        w_nbits:    8 or 4
        group_size: quantization group size along k_in

    Returns:
        W_q:    quantized uint8 weights, same shape as input
        scales: (E, n_out, num_groups)
        zeros:  (E, n_out, num_groups)
    """
    E, n_out, k_in = weights.shape
    num_groups = k_in // group_size
    w_bits = 8 if w_nbits == 8 else 4

    # (E, n_out, num_groups, group_size)
    w_r = weights.view(E, n_out, num_groups, group_size)
    w_min = w_r.min(dim=-1, keepdim=True)[0]
    w_max = w_r.max(dim=-1, keepdim=True)[0]
    scale = (w_max - w_min) / ((2 ** w_bits) - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    W_norm = (w_r - w_min) / (scale + 1e-8)
    W_q = W_norm.round().clamp(0, 2 ** w_bits - 1).to(torch.uint8)

    if w_nbits == 4:
        # pack 2 int4 values per byte
        W_q = W_q.view(E, n_out, num_groups, group_size // 2, 2)
        W_q = (W_q[..., 0] & 0xF) | (W_q[..., 1] << 4)
        W_q = W_q.view(E, n_out, -1)
    else:
        W_q = W_q.view(E, n_out, -1)

    scales = scale.squeeze(-1).view(E, n_out, num_groups)
    zeros  = w_min.squeeze(-1).view(E, n_out, num_groups)
    return W_q, scales, zeros


# ─────────────────────────────────────────────────────────────────────────────
# Reference: vLLM-style pure-PyTorch FP16 SwiGLU MoE
# ─────────────────────────────────────────────────────────────────────────────

def vllm_fp16_moe_from_deq_weights(inp, W1_deq, W2_deq, W3, topk_weights, topk_ids):
    """Pure-PyTorch FP16 SwiGLU MoE on dequantized weights (vLLM-style reference).

    W1_deq: (E, 2*intermediate, H)  - gate and up projections concatenated
    W2_deq: (E, H, intermediate)    - down projection
    Operates in (E, 2*intermediate, H) / (E, H, intermediate) convention.
    """
    import torch.nn.functional as F
    
    M, H = inp.shape
    E = W1_deq.shape[0]
    intermediate_dim = W1_deq.shape[1] // 2
    output = torch.zeros(M, H, dtype=inp.dtype, device=inp.device)

    # Split W1 into gate and up projections
    W1_gate = W1_deq[:, :intermediate_dim, :]    # (E, inter, H)
    W1_up   = W1_deq[:, intermediate_dim:, :]    # (E, inter, H)

    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        tokens_e = mask.nonzero(as_tuple=True)[0]
        weights_e = topk_weights[mask]

        inp_e = inp.index_select(0, tokens_e)
        
        # vLLM-style: SiLU(gate) * up
        gate = torch.mm(inp_e, W1_gate[e].T)   # (num_tokens, inter)
        up   = torch.mm(inp_e, W1_up[e].T)     # (num_tokens, inter)
        act  = F.silu(gate) * up
        down = torch.mm(act, W2_deq[e].T)       # (num_tokens, H)

        down_w = down * weights_e.unsqueeze(1)
        output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, H), down_w)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE       = "cuda"
DTYPE        = torch.float16
GROUP_SIZE   = 128
NUM_EXPERTS  = 8
TOP_K        = 2
NUM_ITERS    = 10
WARMUP       = 3
RANDOM_SEED  = 42

TEST_SHAPES = [
    # ── Seq len sweep: fixed H=1024, K=3584 ──────────────────
    (128,  1024, 3584),
    (256,  1024, 3584),
    (512,  1024, 3584),
    (1024, 1024, 3584),
    (2048, 1024, 3584),
    (4096, 1024, 3584),
    (8192, 1024, 3584),
    (16384, 1024, 3584),
    # ── Hidden dim sweep: fixed S=2048, K=3584 ─────────────────
    (2048, 768,  3584),
    (2048, 1536, 3584),
    (2048, 2048, 3584),
    # ── Inter dim sweep: fixed S=2048, H=1024 ──────────────────
    (2048, 1024, 2048),
    (2048, 1024, 2730),
    (2048, 1024, 4096),
    # ── Large shapes ───────────────────────────────────────────
    (8192, 1536, 3584),
    (16384, 1024, 3584),
]

SHAPE_CATS = (
    ["Seq len sweep"]    * 8 +
    ["Hidden dim sweep"] * 3 +
    ["Inter dim sweep"]  * 3 +
    ["Large shapes"]     * 2
)


# ─────────────────────────────────────────────────────────────────────────────
# Generic benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(mode, w_nbits):
    """Run the benchmark for a specific quantization mode.

    Args:
        mode:    "W8A16" or "W4A16"
        w_nbits: 8 or 4
    """
    device = DEVICE
    dtype  = DTYPE

    results = []
    qlabel  = f"QC {mode}"

    header = f"\n{'='*128}"
    header += f"\nQC-MoE {mode} vs vLLM-style FP16 MoE"
    header += f"\n{'='*128}"
    header += "\n"
    header += (
        f"{'Shape (S,H,K)':<24} {qlabel+' ms':<14} {'FP16 ref ms':<14} "
        f"{qlabel+' TFLOPS':<12} {'FP16 TFLOPS':<14} {'Speedup':<10} {'MaxAbsErr':<12}"
    )
    header += f"\n{'-'*128}\n"
    print(header)

    for idx, (seq_len, hidden_dim, inter_dim) in enumerate(TEST_SHAPES):
        torch.manual_seed(RANDOM_SEED)

        # ── vLLM-style Weights ──────────────────────────────────
        # W1_deq: gate+up concatenated (E, 2*inter, H)
        W1_deq = torch.randn(NUM_EXPERTS, 2 * inter_dim, hidden_dim, dtype=dtype, device=device)
        # W2_deq: down projection (E, H, inter)
        W2_deq = torch.randn(NUM_EXPERTS, hidden_dim, inter_dim, dtype=dtype, device=device)

        # ── Quantize for QC-MoE kernel ─────────────────────────
        W1_q, W1_sc, W1_z = quantize_weights_moe(W1_deq, w_nbits, GROUP_SIZE)
        W2_T, W2_sc, W2_z = quantize_weights_moe(W2_deq.transpose(1, 2), w_nbits, GROUP_SIZE)
        W2_q = W2_T.transpose(1, 2)

        # ── Input and routing ─────────────────────────────────
        inp          = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        topk_weights = torch.ones(seq_len, TOP_K, dtype=dtype, device=device) / TOP_K
        topk_ids     = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K),
                                     dtype=torch.int64, device=device)

        quant_mode   = QuantMode.W8A16 if w_nbits == 8 else QuantMode.W4A16
        quant_label  = "W8A16" if w_nbits == 8 else "W4A16"

        # ── Warmup ────────────────────────────────────────────
        for _ in range(WARMUP):
            _ = fused_moe(
                inp, None, None, None,  # vLLM style
                topk_weights=topk_weights, topk_ids=topk_ids,
                quant_config=QuantConfig(mode=quant_mode, group_size=GROUP_SIZE),
                num_experts=NUM_EXPERTS, top_k=TOP_K,
                w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
                w2_q=W2_q, w2_scales=W2_sc, w2_zeros=W2_z,
            )
            _ = vllm_fp16_moe_from_deq_weights(inp, W1_deq, W2_deq, None, topk_weights, topk_ids)
        torch.cuda.synchronize()

        # ── Time QC kernel ───────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            out_qc = fused_moe(
                inp, None, None, None,  # vLLM style
                topk_weights=topk_weights, topk_ids=topk_ids,
                quant_config=QuantConfig(mode=quant_mode, group_size=GROUP_SIZE),
                num_experts=NUM_EXPERTS, top_k=TOP_K,
                w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
                w2_q=W2_q, w2_scales=W2_sc, w2_zeros=W2_z,
            )
        torch.cuda.synchronize()
        qc_ms = (time.perf_counter() - t0) / NUM_ITERS * 1000

        # ── Time FP16 reference ───────────────────────────────
        t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            out_ref = vllm_fp16_moe_from_deq_weights(inp, W1_deq, W2_deq, None, topk_weights, topk_ids)
        torch.cuda.synchronize()
        ref_ms = (time.perf_counter() - t0) / NUM_ITERS * 1000

        # ── Compute metrics ───────────────────────────────────
        total_flops = seq_len * TOP_K * 4 * hidden_dim * inter_dim
        qc_tflops   = total_flops / qc_ms / 1e9
        ref_tflops  = total_flops / ref_ms / 1e9
        speedup     = ref_ms / qc_ms
        max_abs_err = (out_qc.float() - out_ref.float()).abs().max().item()

        # ── Print row ────────────────────────────────────────
        row = (
            f"{f'({seq_len},{hidden_dim},{inter_dim})':<24} "
            f"{qc_ms:<14.3f} {ref_ms:<14.3f} "
            f"{qc_tflops:<12.2f} {ref_tflops:<14.2f} "
            f"{speedup:<10.2f} {max_abs_err:<12.5f}"
        )
        print(row)

        results.append({
            "mode":       mode,
            "idx":        idx + 1,
            "category":   SHAPE_CATS[idx],
            "shape":      f"({seq_len},{hidden_dim},{inter_dim})",
            "seq_len":     seq_len,
            "hidden_dim":  hidden_dim,
            "inter_dim":   inter_dim,
            "w_nbits":    w_nbits,
            "qc_ms":     round(qc_ms,    3),
            "fp16_ms":   round(ref_ms,   3),
            "qc_tflops": round(qc_tflops,  2),
            "fp16_tflops": round(ref_tflops, 2),
            "speedup":   round(speedup,   2),
            "max_abs_err": round(max_abs_err, 5),
        })

    print(f"\n{'='*128}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results, out_path):
    fieldnames = [
        "mode", "idx", "category", "shape",
        "seq_len", "hidden_dim", "inter_dim", "w_nbits",
        "qc_ms", "fp16_ms",
        "qc_tflops", "fp16_tflops",
        "speedup", "max_abs_err",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] Written: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC-MoE Benchmark")
    parser.add_argument("--w8a16", action="store_true", default=False,
                        help="Run W8A16 benchmark only")
    parser.add_argument("--w4a16", action="store_true", default=False,
                        help="Run W4A16 benchmark only")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_w8a16 = not args.w4a16
    run_w4a16 = not args.w8a16

    if run_w8a16:
        print("\n>>> Running W8A16 benchmark <<<")
        w8a16_results = run_benchmark("W8A16", w_nbits=8)
        save_csv(w8a16_results, os.path.join(script_dir, "qcmoe_w8a16_data.csv"))

    if run_w4a16:
        print("\n>>> Running W4A16 benchmark <<<")
        w4a16_results = run_benchmark("W4A16", w_nbits=4)
        save_csv(w4a16_results, os.path.join(script_dir, "qcmoe_w4a16_data.csv"))

    print(f"\n[OK] Benchmark complete.")
    print(f"      Run plot_results.py to regenerate charts with both modes.")

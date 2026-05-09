"""
Phase-1 multi-shape precision test for fused_marlin_moe (FlagGems wrapper).
"""
import argparse
import time
import torch

from flag_gems.fused.fused_marlin_moe import (
    fused_marlin_moe,
    QUANT_TYPE_UINT4B8,
)


def reference_swiglu_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    M, _ = hidden_states.shape
    _, two_N, _ = w1.shape
    N = two_N // 2
    topk = topk_ids.shape[1]
    out = torch.zeros_like(hidden_states)
    for m in range(M):
        for k in range(topk):
            e = topk_ids[m, k].item()
            w_topk = topk_weights[m, k]
            x = hidden_states[m]
            gate_up = w1[e] @ x
            gate = gate_up[:N]
            up = gate_up[N:]
            act = torch.nn.functional.silu(gate) * up
            y = w2[e] @ act
            out[m] += w_topk.to(y.dtype) * y
    return out


TEST_CASES = [
    ("tiny",          1,   128,   256,  4,  2),
    ("small",         8,   128,   256,  8,  2),
    ("medium",        64,  128,   256,  8,  2),
    ("mid-N1024",     32,  1024,  2048, 8,  2),
    ("mid-largeN",    32,  2048,  4096, 8,  2),
    ("mixtral-b1",    1,   4096,  14336, 8, 2),
    ("mixtral-b16",  16,   4096,  14336, 8, 2),
    ("ds-small",     16,   2048,  1408,  64, 6),
    ("ds-mid",       64,   2048,  1408,  256, 8),
]
DTYPES = [torch.bfloat16, torch.float16]


def run_one(label, M, K, N, E, topk, dtype, device='cuda', seed=0):
    torch.manual_seed(seed)
    scale = 0.05
    hidden = torch.randn(M, K, dtype=dtype, device=device) * scale
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device=device) * scale
    w2 = torch.randn(E, K, N,    dtype=dtype, device=device) * scale
    topk_weights = torch.softmax(torch.randn(M, topk, device=device), dim=-1).to(torch.float32)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.long, device=device)
    w1_scale = torch.ones(E, 2 * N, dtype=dtype, device=device)
    w2_scale = torch.ones(E, K,     dtype=dtype, device=device)

    ref = reference_swiglu_moe(hidden, w1, w2, topk_weights, topk_ids)
    got = fused_marlin_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        bias1=None, bias2=None,
        w1_scale=w1_scale, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )
    max_diff = (ref - got).abs().max().item()
    mean_diff = (ref - got).abs().mean().item()
    atol = 5e-2 if dtype == torch.float16 else 1e-2
    passed = torch.allclose(ref, got, atol=atol, rtol=5e-2)
    has_nan = torch.isnan(got).any().item()
    has_inf = torch.isinf(got).any().item()
    return passed, max_diff, mean_diff, has_nan, has_inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16"])
    args = parser.parse_args()

    cases = TEST_CASES
    if args.only:
        cases = [c for c in cases if args.only in c[0]]
    dtypes = DTYPES
    if args.dtype:
        dtypes = [torch.bfloat16] if args.dtype == "bf16" else [torch.float16]

    n_pass, n_fail, n_err = 0, 0, 0
    rows = []
    for dtype in dtypes:
        dstr = "bf16" if dtype == torch.bfloat16 else "fp16"
        for label, M, K, N, E, topk in cases:
            t0 = time.time()
            try:
                passed, max_d, mean_d, nan, inf = run_one(label, M, K, N, E, topk, dtype)
                ms = (time.time() - t0) * 1000
                status = "PASS" if (passed and not nan and not inf) else "FAIL"
                if status == "PASS":
                    n_pass += 1
                else:
                    n_fail += 1
                rows.append((status, dstr, label, f"M={M} K={K} N={N} E={E} k={topk}",
                             f"max={max_d:.2e}", f"mean={mean_d:.2e}", f"{ms:.0f}ms",
                             "NaN" if nan else ("Inf" if inf else "")))
            except Exception as e:
                n_err += 1
                rows.append(("ERROR", dstr, label, f"M={M} K={K} N={N} E={E} k={topk}",
                             "", "", "", str(e)[:80]))

    print("\n" + "=" * 110)
    print(f"{'STATUS':<6} {'DTYPE':<6} {'LABEL':<14} {'SHAPE':<32} "
          f"{'MAX_DIFF':<14} {'MEAN_DIFF':<14} {'TIME':<8} NOTE")
    print("-" * 110)
    for r in rows:
        print(f"{r[0]:<6} {r[1]:<6} {r[2]:<14} {r[3]:<32} "
              f"{r[4]:<14} {r[5]:<14} {r[6]:<8} {r[7]}")
    print("=" * 110)
    print(f"Total: {n_pass} pass, {n_fail} fail, {n_err} error")


if __name__ == "__main__":
    main()

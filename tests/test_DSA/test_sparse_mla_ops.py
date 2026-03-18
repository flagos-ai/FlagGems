"""
Sparse MLA accuracy + performance tests (vLLM-compatible 3D interface).

Model configurations:
- GLM-5 CONFIG1: d_qk=512, h_q=64, topk=512
- GLM-5 CONFIG2: d_qk=512, h_q=128, topk=1024
- DeepSeek-V3.2: d_qk=576, h_q=128, topk=2048

Usage:
    pytest tests/test_DSA/test_sparse_mla_ops.py -v
    pytest tests/test_DSA/test_sparse_mla_ops.py -v -k "decode"
    pytest tests/test_DSA/test_sparse_mla_ops.py -v -k "benchmark"
"""

import pytest
import torch

from flag_gems.fused.DSA.sparse_mla import sparse_prefill_fwd

from .torch_src.sparse_mla_fwd import ref_sparse_prefill_fwd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Accuracy tests
# ============================================================


@pytest.mark.sparse_mla_forward
@pytest.mark.parametrize(
    "d_qk,h_q,topk,s_kv",
    [
        # GLM-5 CONFIG1: d_qk=512, h_q=64, topk=512
        (512, 64, 512, 8192),
        (512, 64, 512, 32768),
        (512, 64, 512, 49152),
        (512, 64, 512, 65536),
        # GLM-5 CONFIG2: d_qk=512, h_q=128, topk=1024
        (512, 128, 1024, 8192),
        (512, 128, 1024, 32768),
        (512, 128, 1024, 49152),
        (512, 128, 1024, 65536),
        # DeepSeek-V3.2: d_qk=576, h_q=128, topk=2048
        (576, 128, 2048, 8192),
        (576, 128, 2048, 32768),
        (576, 128, 2048, 65536),
        (576, 128, 2048, 98304),
        (576, 128, 2048, 131072),
    ],
)
@pytest.mark.parametrize("s_q", [1, 4096])
def test_sparse_mla_accuracy(s_q, d_qk, h_q, topk, s_kv):
    """Sparse MLA accuracy: Triton kernel vs PyTorch reference."""
    h_kv = 1
    d_v = 512
    dtype = torch.bfloat16
    sm_scale = d_qk ** (-0.5)
    actual_topk = min(topk, s_kv)

    torch.manual_seed(42)
    q = torch.randn(s_q, h_q, d_qk, dtype=dtype, device=device)
    kv = torch.randn(s_kv, h_kv, d_qk, dtype=dtype, device=device)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, actual_topk), dtype=torch.int32, device=device
    )

    # Reference (pure PyTorch)
    ref_out, _, ref_lse = ref_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)

    # Triton kernel
    act_out, _, act_lse = sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)

    # Check correctness
    assert not torch.isnan(act_out).any(), "Output contains NaN"
    max_diff = (ref_out.float() - act_out.float()).abs().max().item()
    assert max_diff < 1e-2, f"Output max diff {max_diff:.6f} exceeds tolerance 1e-2"

    lse_diff = (ref_lse - act_lse).abs().max().item()
    assert lse_diff < 0.1, f"LSE max diff {lse_diff:.6f} exceeds tolerance 0.1"


# ============================================================
# Edge case tests
# ============================================================


@pytest.mark.sparse_mla_forward_edge_cases
@pytest.mark.parametrize(
    "config",
    [
        {"s_q": 1, "s_kv": 64, "h_q": 8, "topk": 32, "d_qk": 576},
        {"s_q": 17, "s_kv": 1030, "h_q": 64, "topk": 256, "d_qk": 576},
        {"s_q": 128, "s_kv": 4096, "h_q": 128, "topk": 512, "d_qk": 512},
    ],
)
def test_sparse_mla_edge_cases(config):
    """Sparse MLA edge cases."""
    d_v = 512
    dtype = torch.bfloat16
    sm_scale = config["d_qk"] ** (-0.5)

    torch.manual_seed(42)
    q = torch.randn(
        config["s_q"], config["h_q"], config["d_qk"], dtype=dtype, device=device
    )
    kv = torch.randn(config["s_kv"], 1, config["d_qk"], dtype=dtype, device=device)
    actual_topk = min(config["topk"], config["s_kv"])
    indices = torch.randint(
        0,
        config["s_kv"],
        (config["s_q"], 1, actual_topk),
        dtype=torch.int32,
        device=device,
    )

    ref_out, _, _ = ref_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
    act_out, _, _ = sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)

    assert not torch.isnan(act_out).any(), "Output contains NaN"
    max_diff = (ref_out.float() - act_out.float()).abs().max().item()
    assert max_diff < 1e-2, f"Output max diff {max_diff:.6f} exceeds tolerance 1e-2"


# ============================================================
# Benchmark tests
# ============================================================


def _bench_single(s_q, d_qk, h_q, topk, s_kv, warmup=25, rep=100):
    """Run a single benchmark config, return (ms_ref, ms_triton, speedup, correct)."""
    from triton.testing import do_bench

    h_kv = 1
    d_v = 512
    dtype = torch.bfloat16
    sm_scale = d_qk ** (-0.5)
    actual_topk = min(topk, s_kv)

    torch.manual_seed(42)
    q = torch.randn(s_q, h_q, d_qk, dtype=dtype, device=device)
    kv = torch.randn(s_kv, h_kv, d_qk, dtype=dtype, device=device)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, actual_topk), dtype=torch.int32, device=device
    )

    # Correctness
    ref_out, _, _ = ref_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
    act_out, _, _ = sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
    max_diff = (ref_out.float() - act_out.float()).abs().max().item()
    correct = max_diff < 1e-2 and not torch.isnan(act_out).any().item()

    # Benchmark
    ms_ref = do_bench(
        lambda: ref_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v),
        warmup=warmup,
        rep=rep,
    )
    ms_triton = do_bench(
        lambda: sparse_prefill_fwd(q, kv, indices, sm_scale, d_v),
        warmup=warmup,
        rep=rep,
    )
    speedup = ms_ref / ms_triton if ms_triton > 0 else 0
    return ms_ref, ms_triton, speedup, correct


# Decode benchmarks (s_q=1)
DECODE_BENCH_CONFIGS = [
    (1, 512, 64, 512, 8192, "GLM5-C1"),
    (1, 512, 64, 512, 65536, "GLM5-C1"),
    (1, 512, 128, 1024, 8192, "GLM5-C2"),
    (1, 512, 128, 1024, 65536, "GLM5-C2"),
    (1, 576, 128, 2048, 8192, "DSV3"),
    (1, 576, 128, 2048, 65536, "DSV3"),
    (1, 576, 128, 2048, 131072, "DSV3"),
]

# Prefill benchmarks (s_q=4096)
PREFILL_BENCH_CONFIGS = [
    (4096, 512, 64, 512, 8192, "GLM5-C1"),
    (4096, 512, 64, 512, 65536, "GLM5-C1"),
    (4096, 512, 128, 1024, 8192, "GLM5-C2"),
    (4096, 512, 128, 1024, 65536, "GLM5-C2"),
    (4096, 576, 128, 2048, 8192, "DSV3"),
    (4096, 576, 128, 2048, 65536, "DSV3"),
    (4096, 576, 128, 2048, 131072, "DSV3"),
]


@pytest.mark.sparse_mla_benchmark
@pytest.mark.parametrize(
    "s_q,d_qk,h_q,topk,s_kv,name",
    DECODE_BENCH_CONFIGS,
    ids=[f"{c[5]}-sq{c[0]}-skv{c[4]}" for c in DECODE_BENCH_CONFIGS],
)
def test_sparse_mla_benchmark_decode(s_q, d_qk, h_q, topk, s_kv, name):
    """Benchmark sparse MLA decode (s_q=1) vs PyTorch reference."""
    ms_ref, ms_triton, speedup, correct = _bench_single(s_q, d_qk, h_q, topk, s_kv)
    print(
        f"\n[{name}] s_q={s_q} s_kv={s_kv}: "
        f"ref={ms_ref:.4f}ms triton={ms_triton:.4f}ms "
        f"speedup={speedup:.2f}x correct={correct}"
    )
    assert correct, "Correctness check failed"


@pytest.mark.sparse_mla_benchmark
@pytest.mark.parametrize(
    "s_q,d_qk,h_q,topk,s_kv,name",
    PREFILL_BENCH_CONFIGS,
    ids=[f"{c[5]}-sq{c[0]}-skv{c[4]}" for c in PREFILL_BENCH_CONFIGS],
)
def test_sparse_mla_benchmark_prefill(s_q, d_qk, h_q, topk, s_kv, name):
    """Benchmark sparse MLA prefill (s_q=4096) vs PyTorch reference."""
    ms_ref, ms_triton, speedup, correct = _bench_single(s_q, d_qk, h_q, topk, s_kv)
    print(
        f"\n[{name}] s_q={s_q} s_kv={s_kv}: "
        f"ref={ms_ref:.4f}ms triton={ms_triton:.4f}ms "
        f"speedup={speedup:.2f}x correct={correct}"
    )
    assert correct, "Correctness check failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

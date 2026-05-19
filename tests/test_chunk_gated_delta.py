import math
import warnings  # <--- ይህንን ከላይ ይጨምሩ

import pytest
import torch

from flag_gems.ops.chunk_gated_delta import (
    FlagOS_ChunkGatedDelta,
    chunk_gated_delta_rule,
    torch_chunk_gated_delta_rule,
)

# በ PyTorch እና Triton መሃል ያለውን ልዩነት 100% ለማጥፋት TF32ን እናጠፋዋለን
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_inputs(B, H, T, D, device, dtype, seed=42, requires_grad=False):
    """Return (q, k, v, beta) freshly drawn from a fixed seed."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    q.requires_grad_(requires_grad)

    # የዴልታ ስሌት እንዳይፈነዳ (Explode እንዳያደርግ) k እና v በ sqrt(D) ተካፍለዋል
    k = torch.randn(B, H, T, D, device=device, dtype=dtype) / math.sqrt(D)
    k.requires_grad_(requires_grad)

    v = torch.randn(B, H, T, D, device=device, dtype=dtype) / math.sqrt(D)
    v.requires_grad_(requires_grad)

    beta = torch.rand(B, H, T, device=device, dtype=dtype).clamp(min=0.01)
    beta.requires_grad_(requires_grad)
    return q, k, v, beta


def _clone_with_grad(*tensors):
    """Clone a group of tensors, detaching and enabling grad."""
    return tuple(t.clone().detach().requires_grad_(True) for t in tensors)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Forward accuracy
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("BT", [16])
@pytest.mark.parametrize(
    "B, H, T, D",
    [
        (1, 1, 16, 16),
        (2, 4, 32, 32),
        (1, 2, 64, 64),
    ],
)
def test_forward_accuracy(B, H, T, D, BT, dtype, device):
    q, k, v, beta = _make_inputs(B, H, T, D, device, dtype)

    out_ref = torch_chunk_gated_delta_rule(q, k, v, beta)
    out_tri = chunk_gated_delta_rule(q, k, v, beta, BT=BT)

    max_diff = (out_tri - out_ref).abs().max().item()
    print(f"\n[INFO] Forward max diff: {max_diff:.2e}  (shape B={B} H={H} T={T} D={D})")

    torch.testing.assert_close(
        out_tri,
        out_ref,
        atol=1e-4,
        rtol=1e-4,
        msg=f"Forward mismatch (max_diff={max_diff:.2e})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Backward accuracy
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("BT", [16])
@pytest.mark.parametrize(
    "B, H, T, D",
    [
        (1, 1, 16, 16),
        (2, 2, 32, 32),
    ],
)
def test_backward_accuracy(B, H, T, D, BT, dtype, device):
    q, k, v, beta = _make_inputs(B, H, T, D, device, dtype)
    torch.manual_seed(99)
    do = torch.randn(B, H, T, D, device=device, dtype=dtype)

    q_r, k_r, v_r, beta_r = _clone_with_grad(q, k, v, beta)
    torch_chunk_gated_delta_rule(q_r, k_r, v_r, beta_r).backward(do)

    q_t, k_t, v_t, beta_t = _clone_with_grad(q, k, v, beta)
    chunk_gated_delta_rule(q_t, k_t, v_t, beta_t, BT=BT).backward(do)

    for name, g_ref, g_tri in [
        ("dq", q_r.grad, q_t.grad),
        ("dk", k_r.grad, k_t.grad),
        ("dv", v_r.grad, v_t.grad),
        ("dbeta", beta_r.grad, beta_t.grad),
    ]:
        max_diff = (g_tri - g_ref).abs().max().item()
        print(f"[INFO] {name} max diff: {max_diff:.2e}")
        torch.testing.assert_close(
            g_tri,
            g_ref,
            atol=1e-3,
            rtol=1e-3,
            msg=f"{name} gradient mismatch (max_diff={max_diff:.2e})",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. nn.Module wrapper smoke test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("BT", [16])
def test_module_wrapper(BT, device):
    B, H, T, D = 1, 1, 16, 16
    q, k, v, beta = _make_inputs(B, H, T, D, device, torch.float32)

    ref = torch_chunk_gated_delta_rule(q, k, v, beta)
    mod = FlagOS_ChunkGatedDelta(BT=BT).to(device)
    out = mod(q, k, v, beta)

    torch.testing.assert_close(
        out, ref, atol=1e-4, rtol=1e-4, msg="Module wrapper output mismatch"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Numerical gradient check
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gradcheck is slow on CPU for Triton ops; skip in CI without GPU",
)
def test_gradcheck(device):
    B, H, T, D = 1, 1, 4, 4
    torch.manual_seed(0)

    q = torch.randn(B, H, T, D, dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=torch.float32, device=device, requires_grad=True)
    beta = (
        torch.rand(B, H, T, dtype=torch.float32, device=device)
        .clamp(min=0.1)
        .requires_grad_(True)
    )

    # Warnings እንዳይታዩ (Ignore እንዲደረጉ) በዚህ አግደናቸዋል
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert torch.autograd.gradcheck(
            torch_chunk_gated_delta_rule,
            (q, k, v, beta),
            eps=1e-3,
            atol=1e-2,
            rtol=1e-2,
            nondet_tol=0.0,
            raise_exception=True,
        ), "gradcheck failed on reference implementation"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge-case: T not divisible by BT
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("T, BT", [(17, 16), (33, 16), (15, 16)])
def test_non_divisible_T(T, BT, device):
    B, H, D = 1, 1, 16
    q, k, v, beta = _make_inputs(B, H, T, D, device, torch.float32)

    out_ref = torch_chunk_gated_delta_rule(q, k, v, beta)
    out_tri = chunk_gated_delta_rule(q, k, v, beta, BT=BT)

    max_diff = (out_tri - out_ref).abs().max().item()
    print(f"\n[INFO] T={T}, BT={BT}: max diff = {max_diff:.2e}")
    torch.testing.assert_close(out_tri, out_ref, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Performance Benchmark (Triton vs PyTorch)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, H, T, D",
    [
        (2, 4, 1024, 64),  # እውነተኛ (Realistic) የሞዴል ሳይዝ
    ],
)
@pytest.mark.parametrize("BT", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_benchmark(B, H, T, D, BT, dtype, device):
    """
    ይህ ቴስት የ PyTorch እና የ Triton ኮዶችን ፍጥነት ለክቶ ያነፃፅራል (Speedup)።
    """
    q, k, v, beta = _make_inputs(B, H, T, D, device, dtype)

    # የ Warmup ዙሮች (ለማሞቅ)
    for _ in range(3):
        torch_chunk_gated_delta_rule(q, k, v, beta)
        chunk_gated_delta_rule(q, k, v, beta, BT=BT)

    torch.cuda.synchronize()

    # 1. የ PyTorchን ፍጥነት መለካት
    import time

    start_time = time.time()
    for _ in range(10):
        torch_chunk_gated_delta_rule(q, k, v, beta)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / 10 * 1000  # በሚሊ-ሰከንድ (ms)

    # 2. የ Tritonን ፍጥነት መለካት
    start_time = time.time()
    for _ in range(10):
        chunk_gated_delta_rule(q, k, v, beta, BT=BT)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / 10 * 1000  # በሚሊ-ሰከንድ (ms)

    speedup = pytorch_time / triton_time

    print("\n" + "=" * 50)
    print(f"📊 BENCHMARK RESULTS (Shape: {B}x{H}x{T}x{D})")
    print("=" * 50)
    print(f"PyTorch Time: {pytorch_time:.3f} ms")
    print(f"Triton Time : {triton_time:.3f} ms")
    print(f"🚀 Speedup   : {speedup:.2f}x (Triton is {speedup:.2f} times faster!)")
    print("=" * 50)

    # ውድድሩ ቢያንስ የ 0.9x ፍጥነት ይጠይቃል (የኛ በጣም ፈጣን ይሆናል ተብሎ ይጠበቃል)
    assert (
        speedup >= 0.9
    ), f"Performance failed: Speedup is {speedup:.2f}x (Required >= 0.9x)"

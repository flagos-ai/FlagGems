# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Qualified H100 TMA64 prefill GEMM; independent matrix and numeric-scale ABI."""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

QUALIFIED_SHAPES = frozenset(
    (
        (8192, 576, 5120),
        (8192, 1792, 5120),
        (8192, 4096, 1280),
        (8192, 5120, 1024),
        (8192, 25600, 6144),
        (15360, 576, 5120),
        (15360, 1792, 5120),
        (15360, 4096, 1280),
        (15360, 5120, 288),
        (15360, 5120, 1024),
        (15360, 25600, 6144),
        (16384, 576, 5120),
        (16384, 1792, 5120),
        (16384, 4096, 1280),
        (16384, 5120, 288),
        (16384, 5120, 1024),
        (16384, 25600, 6144),
    )
)


@triton.jit
def _tile_ids(M, N, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = pid // (GM * nn)
    first = group * GM
    gm = tl.minimum(nm - first, GM)
    pm = first + pid % gm
    pn = (pid % (GM * nn)) // gm
    return pm, pn


@triton.jit
def _consume_slab(ash, bsh, As, Bs, rm, rn, acc, slab_index, K, SLAB: tl.constexpr):
    for part in tl.static_range(0, SLAB // 32):
        ki = slab_index * (SLAB // 32) + part
        # Do not append any extra zero additions for the K288 tail.
        if ki < K // 32:
            a32 = ash.subslice(part * 32, 32, dim=1)
            b32 = bsh.subslice(part * 32, 32, dim=1)
            product = tle.gpu.wgmma(a32, b32, trans_b=True)
            product = tle.gpu.wgmma_wait(0, product)
            sa = tl.load(As + rm * (K // 32) + ki)
            sb = tl.load(Bs + (rn // 32) * (K // 32) + ki)
            acc += product * sa[:, None] * sb[None, :]
    return acc


@triton.jit
def _tma_copy_slab(
    AD,
    BD,
    ash,
    bsh,
    abar,
    bbar,
    pm,
    pn,
    slab_index,
    SLAB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # Each operand owns one exact-byte-count completion barrier. A barrier
    # is not shared between two copies with ambiguous expected-byte updates.
    tle.gpu.copy(AD, ash, [BM, SLAB], [pm * BM, slab_index * SLAB], barrier=abar)
    tle.gpu.copy(BD, bsh, [BN, SLAB], [pn * BN, slab_index * SLAB], barrier=bbar)


@triton.jit
def _tma_ring_kernel(
    AD,
    BD,
    As,
    Bs,
    C,
    M,
    N,
    K,
    SLAB: tl.constexpr,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 128,
    GM: tl.constexpr = 32,
):
    pm, pn = _tile_ids(M, N, BM, BN, GM)
    # Descriptor OOB reads are zero padded. Invalid N576 columns never reach
    # output; valid columns have exactly the same independent FP8 products.
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    safe_rn = rn % N
    ash = tle.gpu.alloc([2, BM, SLAB], tl.float8e4nv)
    bsh = tle.gpu.alloc([2, BN, SLAB], tl.float8e4nv)
    abars = tle.gpu.alloc_barriers(2, expect_bytes=BM * SLAB)
    bbars = tle.gpu.alloc_barriers(2, expect_bytes=BN * SLAB)
    _tma_copy_slab(
        AD, BD, ash.slot(0), bsh.slot(0), abars[0], bbars[0], pm, pn, 0, SLAB, BM, BN
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for slab_index in range(0, tl.cdiv(K, SLAB)):
        current = slab_index % 2
        following = (slab_index + 1) % 2
        phase = (slab_index // 2) % 2
        tle.gpu.barrier_wait(abars[current], phaseIdx=phase)
        tle.gpu.barrier_wait(bbars[current], phaseIdx=phase)
        if slab_index + 1 < tl.cdiv(K, SLAB):
            _tma_copy_slab(
                AD,
                BD,
                ash.slot(following),
                bsh.slot(following),
                abars[following],
                bbars[following],
                pm,
                pn,
                slab_index + 1,
                SLAB,
                BM,
                BN,
            )
        # The slot will not be overwritten until this iteration has consumed
        # every K32 view and waited for its last WGMMA result.
        acc = _consume_slab(
            ash.slot(current),
            bsh.slot(current),
            As,
            Bs,
            rm,
            safe_rn,
            acc,
            slab_index,
            K,
            SLAB,
        )
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    tl.store(
        C + om[:, None] * N + on[None, :],
        acc.to(tl.bfloat16),
        (om[:, None] < M) & (on[None, :] < N),
    )


def supports(A, B, As, Bs, output_dtype=torch.bfloat16):
    if not all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in (A, B, As, Bs)):
        return False
    m, k = A.shape
    n, bk = B.shape
    return (
        output_dtype == torch.bfloat16
        and bk == k
        and (m, n, k) in QUALIFIED_SHAPES
        and A.is_cuda
        and all(t.device == A.device and t.is_contiguous() for t in (A, B, As, Bs))
        and A.dtype == B.dtype == torch.float8_e4m3fn
        and As.dtype == Bs.dtype == torch.float32
        and As.shape == (m, k // 32)
        and Bs.shape == (n // 32, k // 32)
        and A.data_ptr() % 16 == B.data_ptr() % 16 == 0
        and triton.__version__.split("+")[0] == "3.6.0"
        and torch.cuda.get_device_capability(A.device) == (9, 0)
    )


def block_fp8_prefill(A, B, As, Bs, output_dtype=torch.bfloat16):
    """Exact K32 scaling with fresh descriptors and output on the current stream.

    No descriptors, weights, workspaces or tensor pointers are cached.
    The caller selects its original implementation when supports() is false.
    """
    if not supports(A, B, As, Bs, output_dtype):
        raise ValueError("unsupported H100/FlagTree 3.6 prefill matrix contract")
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, k = A.shape
    n = B.shape[0]
    out = torch.empty((m, n), device=A.device, dtype=output_dtype)
    ad = TensorDescriptor.from_tensor(A, [64, 64])
    bd = TensorDescriptor.from_tensor(B, [128, 64])
    _tma_ring_kernel[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
        ad, bd, As, Bs, out, m, n, k, 64, num_warps=4, num_stages=1
    )
    return out

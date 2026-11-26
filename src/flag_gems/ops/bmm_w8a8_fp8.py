import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def bmm_w8a8_fp8_kernel(
    A,
    B,
    A_SCALE,
    B_SCALE,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_asb: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsb: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    A_SCALE += pid_b * stride_asb
    B_SCALE += pid_b * stride_bsb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_cta_per_group = gridy * GROUP_M
        group_id = pid // num_cta_per_group
        inner_group_id = pid % num_cta_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    if TILE_M > BLOCK_M:
        m_block_idx = offs_m // BLOCK_M
    else:
        m_block_idx = (pid_m * TILE_M) // BLOCK_M
    n_block_idx = (pid_n * TILE_N) // BLOCK_N
    a_scale_ptr = A_SCALE + m_block_idx * stride_asm
    b_scale_ptr = B_SCALE + n_block_idx * stride_bsn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    if BLOCK_K == K:
        if TILE_M > BLOCK_M:
            a_scale = tl.load(a_scale_ptr, mask=offs_m < M, other=0).to(tl.float32)[
                :, None
            ]
        else:
            a_scale = tl.load(a_scale_ptr).to(tl.float32)
        b_scale = tl.load(b_scale_ptr).to(tl.float32)
        for k_tile in range(NUM_K_TILES):
            k_offsets = k_tile * TILE_K + offs_k
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk
        acc = acc * a_scale * b_scale
    else:
        for k_tile in range(NUM_K_TILES):
            k_offsets = k_tile * TILE_K + offs_k
            k_block_idx = (k_tile * TILE_K) // BLOCK_K
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            partial = tl.dot(a, b, allow_tf32=False)
            if TILE_M > BLOCK_M:
                a_scale = tl.load(
                    a_scale_ptr + k_block_idx * stride_ask,
                    mask=offs_m < M,
                    other=0,
                ).to(tl.float32)[:, None]
            else:
                a_scale = tl.load(a_scale_ptr + k_block_idx * stride_ask).to(tl.float32)
            b_scale = tl.load(b_scale_ptr + k_block_idx * stride_bsk).to(tl.float32)
            acc += partial * a_scale * b_scale
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


def _get_bmm_w8a8_fp8_config(M, N, K, block_k):
    if K <= 64:
        if M <= 64:
            return 64, 32, 64, 1, 8, 3
        return 64, 16, 64, 2, 4, 2
    if M <= 64 and N <= 64:
        return 32, 32, 64, 4, 8, 3
    if M <= 128 and N <= 128:
        return 32, 16, 128, 2, 4, 2
    if M <= 256 and N <= 256:
        return 128, 16, 128, 1, 4, 2
    if M >= 2048 and N >= 2048:
        return 256, 32, 128, 2, 4, 2
    if M >= 1024 and N >= 1024:
        return 256, 32, 128, 1, 4, 2
    if M >= 512 and N >= 512:
        return 128, 32, 128, 8, 4, 3
    return 64, 16, 128, 2, 4, 2


@libentry()
@triton.jit
def _bmm_w8a8_fp8_pack_b(B, Packed, K: tl.constexpr, N: tl.constexpr):
    batch = tl.program_id(2)
    kk = tl.program_id(0) * 32 + tl.arange(0, 32)
    nn = tl.program_id(1) * 128 + tl.arange(0, 128)
    mask = (kk[:, None] < K) & (nn[None, :] < N)
    values = tl.load(B + batch * K * N + kk[:, None] * N + nn[None, :], mask, 0.0)
    tl.store(Packed + batch * K * N + nn[None, :] * K + kk[:, None], values, mask)


@libentry()
@triton.jit
def _bmm_w8a8_fp8_hopper_tma(
    A,
    B,
    A_SCALE,
    B_SCALE,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    grid_m: tl.constexpr = triton.cdiv(M, TILE_M)
    grid_n: tl.constexpr = triton.cdiv(N, TILE_N)
    group = pid // (GROUP_M * grid_n)
    group_size = tl.minimum(grid_m - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % group_size
    pn = pid % (GROUP_M * grid_n) // group_size
    acc = tl.zeros((TILE_M, TILE_N), tl.float32)
    # Tiles stay inside the original 128-row/column scale blocks.
    for block in tl.range(triton.cdiv(K, 128), num_stages=STAGES):
        a = A.load([batch, pm * TILE_M, block * 128]).reshape((TILE_M, 128))
        b = B.load([batch, pn * TILE_N, block * 128]).reshape((TILE_N, 128)).T
        sa = tl.load(
            A_SCALE
            + (batch * triton.cdiv(M, 128) + pm * TILE_M // 128) * triton.cdiv(K, 128)
            + block
        )
        sb = tl.load(
            B_SCALE
            + (batch * triton.cdiv(K, 128) + block) * triton.cdiv(N, 128)
            + pn * TILE_N // 128
        )
        partial = tl.dot(a, b, allow_tf32=False)
        acc += partial * (sa * sb)
    Out.store(
        [batch, pm * TILE_M, pn * TILE_N],
        acc.to(Out.dtype).reshape((1, TILE_M, TILE_N)),
    )


def _get_hopper_bmm_w8a8_fp8_config(batch, M, N, K):
    if M <= 512:
        return 64, 128 if batch <= 4 else 64, 8, 4, 2
    if K >= 4096:
        return 128, 128, 8, 8 if batch >= 8 else 4, 3
    if batch >= 8 and M % 128 == 0:
        return 64, 128, 8, 4, 2
    return 128, 128, 8, 4, 3


def _can_use_hopper_bmm_w8a8_fp8(A, B, A_scale, B_scale, out, block_size):
    return (
        TensorDescriptor is not None
        and runtime.device.vendor_name == "nvidia"
        and tuple(block_size) == (128, 128, 128)
        and A.dtype == B.dtype == torch.float8_e4m3fn
        and A_scale.dtype == B_scale.dtype == torch.float32
        and out.dtype == torch.bfloat16
        and A.shape[0] >= 4
        and A.shape[1] >= 256
        and B.shape[2] >= 1024
        and A.shape[2] >= 128
        and A.shape[2] % 128 == 0
        and B.shape[2] % 16 == 0
        and all(t.is_contiguous() for t in (A, B, A_scale, B_scale, out))
        and all(t.data_ptr() % 16 == 0 for t in (A, B, out))
        and torch_device_fn.get_device_capability(A.device)[0] == 9
    )


def _launch_hopper_bmm_w8a8_fp8(A, B, A_scale, B_scale, out):
    batch, M, K = A.shape
    N = B.shape[2]
    tm, tn, group, warps, stages = _get_hopper_bmm_w8a8_fp8_config(batch, M, N, K)
    packed = torch.empty((batch, N, K), dtype=B.dtype, device=B.device)
    a_desc = TensorDescriptor(A, A.shape, A.stride(), [1, tm, 128])
    b_desc = TensorDescriptor(packed, packed.shape, packed.stride(), [1, tn, 128])
    out_desc = TensorDescriptor(out, out.shape, out.stride(), [1, tm, tn])
    # Repack on every call: never cache mutable tensor contents or change the API layout.
    _bmm_w8a8_fp8_pack_b[(triton.cdiv(K, 32), triton.cdiv(N, 128), batch)](
        B, packed, K, N
    )
    _bmm_w8a8_fp8_hopper_tma[(triton.cdiv(M, tm) * triton.cdiv(N, tn), batch)](
        a_desc,
        b_desc,
        A_scale,
        B_scale,
        out_desc,
        M,
        N,
        K,
        tm,
        tn,
        group,
        stages,
        num_warps=warps,
        num_stages=stages,
    )


def bmm_w8a8_fp8(
    A,
    B,
    A_scale,
    B_scale,
    block_size=(128, 128, 128),
    out_dtype=torch.bfloat16,
    out=None,
):
    logger.debug("GEMS BMM_W8A8_FP8_BLOCK_SCALE")
    assert A.ndim == 3 and B.ndim == 3 and A_scale.ndim == 3 and B_scale.ndim == 3
    assert A.shape[0] == B.shape[0] == A_scale.shape[0] == B_scale.shape[0]
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    assert A.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "A must be fp8"
    assert B.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "B must be fp8"

    block_m, block_n, block_k = block_size
    batch, M, K = A.shape
    _, _, N = B.shape
    num_m_blocks = triton.cdiv(M, block_m)
    num_k_blocks = triton.cdiv(K, block_k)
    num_n_blocks = triton.cdiv(N, block_n)
    launch_block_m = M if M < block_m else block_m
    launch_block_k = K if K < block_k else block_k
    launch_block_n = N if N < block_n else block_n
    assert A_scale.shape == (batch, num_m_blocks, num_k_blocks)
    assert B_scale.shape == (batch, num_k_blocks, num_n_blocks)

    if out is None:
        out = torch.empty((batch, M, N), dtype=out_dtype, device=A.device)
    else:
        assert out.shape == (batch, M, N)
        assert out.dtype == out_dtype and out.device == A.device

    if _can_use_hopper_bmm_w8a8_fp8(A, B, A_scale, B_scale, out, block_size):
        with torch_device_fn.device(A.device):
            _launch_hopper_bmm_w8a8_fp8(A, B, A_scale, B_scale, out)
        return out

    def grid_fn(meta):
        return (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(N, meta["TILE_N"]),
            batch,
        )

    (
        tile_m,
        tile_n,
        tile_k,
        group_m,
        num_warps,
        num_stages,
    ) = _get_bmm_w8a8_fp8_config(M, N, K, launch_block_k)
    assert launch_block_m % tile_m == 0 or tile_m % launch_block_m == 0
    assert launch_block_k % tile_k == 0, "block_k must be divisible by TILE_K"
    assert launch_block_n % tile_n == 0, "block_n must be divisible by TILE_N"

    with torch_device_fn.device(A.device):
        bmm_w8a8_fp8_kernel[grid_fn](
            A,
            B,
            A_scale,
            B_scale,
            out,
            M,
            N,
            K,
            NUM_K_TILES=triton.cdiv(K, tile_k),
            BLOCK_M=launch_block_m,
            BLOCK_K=launch_block_k,
            BLOCK_N=launch_block_n,
            stride_ab=A.stride(0),
            stride_am=A.stride(1),
            stride_ak=A.stride(2),
            stride_bb=B.stride(0),
            stride_bk=B.stride(1),
            stride_bn=B.stride(2),
            stride_asb=A_scale.stride(0),
            stride_asm=A_scale.stride(1),
            stride_ask=A_scale.stride(2),
            stride_bsb=B_scale.stride(0),
            stride_bsk=B_scale.stride(1),
            stride_bsn=B_scale.stride(2),
            stride_ob=out.stride(0),
            stride_om=out.stride(1),
            stride_on=out.stride(2),
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out

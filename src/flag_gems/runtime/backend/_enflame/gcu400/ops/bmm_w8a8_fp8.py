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
    EVEN: tl.constexpr,
):
    pid_mn = ext.program_id(0)
    pid_b = ext.program_id(1)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    A_SCALE += pid_b * stride_asb
    B_SCALE += pid_b * stride_bsb
    O += pid_b * stride_ob

    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)
    if GROUP_M == 1:
        pid_m = pid_mn // grid_n
        pid_n = pid_mn % grid_n
    else:
        # Group the linear launch order along M: one GROUP_M-tall band of row tiles
        # keeps the same K band of B resident in cache while walking the N tiles.
        width = GROUP_M * grid_n
        group_id = pid_mn // width
        first_m = group_id * GROUP_M
        group_size = tl.minimum(grid_m - first_m, GROUP_M)
        pid_m = first_m + (pid_mn % group_size)
        pid_n = (pid_mn % width) // group_size

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
            acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk
        acc = acc * a_scale * b_scale
    else:
        if EVEN:
            # All shapes divide the selected tiles exactly: drop every predicate so the
            # K loop stays a pure load/dot/scale pipeline.
            for k_tile in range(NUM_K_TILES):
                k_block_idx = (k_tile * TILE_K) // BLOCK_K
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                partial = tl.dot(a, b, allow_tf32=False)
                if TILE_M > BLOCK_M:
                    a_scale = tl.load(a_scale_ptr + k_block_idx * stride_ask).to(
                        tl.float32
                    )[:, None]
                else:
                    a_scale = tl.load(a_scale_ptr + k_block_idx * stride_ask).to(
                        tl.float32
                    )
                b_scale = tl.load(b_scale_ptr + k_block_idx * stride_bsk).to(tl.float32)
                acc += partial * a_scale * b_scale
                a_ptrs += TILE_K * stride_ak
                b_ptrs += TILE_K * stride_bk
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
                    a_scale = tl.load(a_scale_ptr + k_block_idx * stride_ask).to(
                        tl.float32
                    )
                b_scale = tl.load(b_scale_ptr + k_block_idx * stride_bsk).to(tl.float32)
                acc += partial * a_scale * b_scale
                a_ptrs += TILE_K * stride_ak
                b_ptrs += TILE_K * stride_bk

    if EVEN:
        tl.store(o_ptrs, acc)
    else:
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
        return 32, 64, 128, 2, 4, 2
    if M <= 256 and N <= 256:
        return 128, 128, 128, 1, 4, 2
    # For large shapes, TILE_N covers a full 128-wide scale block: a 32-column block would quadruple the number of CTAs
    # along N
    # and repeatedly move the entire K band of A; at the 2048 size, measurements reached only 15% of bf16 bmm
    #
    # TILE_M is kept at the 128-row scale-block height on purpose: a 256-row tile turns
    # A_SCALE into a 256-wide vector, so every K iteration has to broadcast-multiply the
    # whole 256x128 accumulator (probe: 17 TFLOPS with the vector rescale vs 30 TFLOPS with
    # a scalar one, same shape). GROUP_M=8 keeps a 128-row band of A resident while walking
    # the N tiles, and 3 pipeline stages hide the fp8 load latency.
    if M >= 512 and N >= 512:
        return 128, 128, 128, 8, 4, 3
    return 128, 128, 128, 4, 4, 3


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
    logger.debug("GEMS_ENFLAME BMM_W8A8_FP8_BLOCK_SCALE")
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
        # One flat axis over (m, n) tiles plus one axis over the batch keeps the launch
        # order controllable for the GROUP_M swizzle and stays inside the 65535 / 255
        # hardware grid limits.
        return (
            triton.cdiv(M, meta["TILE_M"]) * triton.cdiv(N, meta["TILE_N"]),
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
    # TILE_N must divide the smaller of 128 and N: otherwise the right-edge tile can cross a scale block,
    # or trigger the divisibility assertion below; here we fall back to a power of 2 so any N is supported
    while launch_block_n % tile_n != 0:
        tile_n //= 2
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
            EVEN=(M % tile_m == 0 and N % tile_n == 0 and K % tile_k == 0),
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


# ---------------------------------------------------------------------------
# Channel-wise INT8 BMM (bmm_w8a8) — GCU400 tuned single-kernel path.
#
# Semantics mirror the generic flag_gems.ops.bmm_w8a8 (per-row A scale,
# per-output-channel B scale, INT32 accumulation, fp32 scale folding, cast to
# out_dtype); only the tiling/launch strategy is vendor-specific.
#
# Config table anchors (L600, triton.testing.do_bench kernel mode w1000/r100,
# cold cache, vs the benchmark's bf16 torch.bmm baseline on pre-converted
# inputs; the baseline leg is measured outside timing):
#   (1,1,1024,4096)   (16,64,4096) nw1 ns1 : 10.5-11.1us vs 19.3us -> 1.75x
#     M=1 pads the MMA M dim to 16; a single 256KB B-column-block load per
#     program beats multi-iteration pipelines (24.2us at BK=512).
#   (1,128,1024,4096) (64,128,2048) nw1 ns1 : 14.7us   vs 14.6us -> ~1.0x
#     GCU400 has no accelerated INT8 MMA (int8 tl.dot runs at the fp32-MMA
#     rate); large-K tiles raise the per-SPC MMA feed rate ~4.5 TOPS and
#     16 programs already saturate it, so skinny-M shapes cap at parity.
#   (2,33,65,129)      (64,128,256)  nw1 ns1 : 6.4us    vs 8.6us  -> 1.36x
#   (1,2048,1024,4096) (128,128,1024) nw2 ns1: 69.9us  vs 66.4us -> 0.95x
#     Large M*N shapes are compute-bound; (128,256,64) collapses (L2 thrash).
# Strides/shapes stay runtime arguments (no per-shape recompile storm); EVEN
# is the only shape-derived constexpr (compile-time mask elision).
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _bmm_w8a8_int8_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_sab,
    stride_sam,
    stride_sbb,
    stride_sbn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_b = pid // (grid_m * grid_n)
    pid_mn = pid % (grid_m * grid_n)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        A
        + pid_b * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0,
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    sa = tl.load(
        SA + pid_b * stride_sab + offs_m * stride_sam, mask=offs_m < M, other=0.0
    ).to(tl.float32)
    sb = tl.load(
        SB + pid_b * stride_sbb + offs_n * stride_sbn, mask=offs_n < N, other=0.0
    ).to(tl.float32)
    val = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = (
        C
        + pid_b * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    if EVEN:
        tl.store(c_ptrs, val.to(C.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            val.to(C.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _w8a8_next_pow2(value, limit):
    if value <= 0:
        return 16
    return max(16, min(limit, triton.next_power_of_2(value)))


def _select_bmm_w8a8_config(batch, M, N, K):
    # Deterministic per-shape tiling (no runtime autotune). See the anchor
    # table above: GEMV-ish / skinny-M / large M*N classes.
    if M <= 16:
        return 16, 64, _w8a8_next_pow2(K, 4096), 1, 1
    if M <= 128:
        return 64, 128, _w8a8_next_pow2(K, 2048), 1, 1
    return 128, 128, _w8a8_next_pow2(K, 1024), 2, 1


def _validate_bmm_w8a8_inputs(A, B, A_scale, B_scale, out_dtype, out):
    if (
        A.ndim != 3
        or B.ndim != 3
        or A.shape[0] != B.shape[0]
        or A.shape[2] != B.shape[1]
    ):
        raise ValueError("Expected compatible batched matrices")
    batch, M, K = A.shape
    N = B.shape[2]
    if A_scale.shape not in ((batch, M), (batch, M, 1)) or B_scale.shape not in (
        (batch, N),
        (batch, 1, N),
    ):
        raise ValueError("Expected per-row A scales and per-output-channel B scales")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("Operands must both have INT8 dtype")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Expected FP16, BF16 or FP32 output")
    if A.device != B.device or A.device != A_scale.device or A.device != B_scale.device:
        raise ValueError("Inputs must share a device")
    if not A_scale.dtype.is_floating_point or not B_scale.dtype.is_floating_point:
        raise TypeError("Scales must be floating point")
    if out is None:
        out = torch.empty((batch, M, N), device=A.device, dtype=out_dtype)
    elif out.shape != (batch, M, N) or out.dtype != out_dtype or out.device != A.device:
        raise ValueError("Invalid output tensor")
    return out


def bmm_w8a8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    *,
    block_size=(128, 128, 128),
    config=None,
) -> torch.Tensor:
    """Channel-wise batched W8A8 matrix multiplication (INT8 operands).

    Same contract as the generic flag_gems.ops.bmm_w8a8: A (batch, M, K),
    B (batch, K, N), INT32 accumulation, per-row A scales and per-output-
    channel B scales folded in FP32 before the out_dtype cast. block_size is
    accepted for API compatibility and unused; config=(BM, BN, BK,
    num_warps, num_stages, group) overrides the deterministic heuristic.
    """
    A, B = activation, weight
    A_scale, B_scale = activation_scale, weight_scale
    out = _validate_bmm_w8a8_inputs(A, B, A_scale, B_scale, out_dtype, out)
    batch, M, K = A.shape
    N = B.shape[2]
    if batch * M * N == 0:
        return out
    A_scale = A_scale[:, :, None] if A_scale.ndim == 2 else A_scale
    B_scale = B_scale[:, None, :] if B_scale.ndim == 2 else B_scale

    if config is None:
        block_m, block_n, block_k, num_warps, num_stages = _select_bmm_w8a8_config(
            batch, M, N, K
        )
    else:
        block_m, block_n, block_k, num_warps, num_stages = config[:5]
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0 and K > 0
    grid = (batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _bmm_w8a8_int8_kernel[grid](
        A,
        B,
        A_scale,
        B_scale,
        out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        A_scale.stride(0),
        A_scale.stride(1),
        B_scale.stride(0),
        B_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN=even,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out

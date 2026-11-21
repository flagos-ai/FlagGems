# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

from .bmm import bmm
from .mul import mul

try:  # the HCU/HIP backend knobs live here since triton 3.3
    from triton import knobs as _triton_knobs
except ImportError:  # pragma: no cover - older triton
    _triton_knobs = None

logger = logging.getLogger(__name__)

_IN_THREAD_TRANSPOSE_KNOB = "use_in_thread_transpose"
_UNSET = object()


def _in_thread_transpose_store():
    hcu = getattr(_triton_knobs, "hcu", None)
    if hcu is None or not hasattr(hcu, _IN_THREAD_TRANSPOSE_KNOB):
        return None
    # Writing the instance dict directly is what `knobs.hcu.<knob> = True` does,
    # minus the putenv call it also makes; the backend reads the knob through the
    # same dict, and skipping putenv keeps this cheap enough to wrap every launch.
    # With putenv, the four environment mutations per call were enough to make the
    # first benchmarked case of a process fluctuate between 0.010ms and 0.031ms.
    return hcu.__dict__


_ITT_STORE = _in_thread_transpose_store()


class _in_thread_transpose:
    """Compile the dot operands with the HCU in-thread-transpose rewrite.

    ``aten::baddbmm.out`` hands ``tl.dot`` a row-major ``B``, so the reduction
    axis of the right-hand operand is the strided one.  Without this rewrite the
    backend rebuilds that operand with a transposing LDS read, which costs
    15%-25% of the runtime of the large tiles selected in ``_tile_config`` (for
    example 1.44ms -> 1.19ms on a 2x4096x4096x4096 float16 case).  The backend
    only enables the rewrite by default on gfx942, so force it on around our own
    launches and restore the previous setting afterwards.

    The knob is not part of the compilation cache key, but this kernel is always
    compiled with the rewrite forced on, so its cache entries stay consistent
    whatever the surrounding process does with the setting.
    """

    __slots__ = ("_prev",)

    def __enter__(self):
        if _ITT_STORE is not None:
            self._prev = _ITT_STORE.get(_IN_THREAD_TRANSPOSE_KNOB, _UNSET)
            _ITT_STORE[_IN_THREAD_TRANSPOSE_KNOB] = True
        return self

    def __exit__(self, *exc_info):
        if _ITT_STORE is not None:
            if self._prev is _UNSET:
                _ITT_STORE.pop(_IN_THREAD_TRANSPOSE_KNOB, None)
            else:
                _ITT_STORE[_IN_THREAD_TRANSPOSE_KNOB] = self._prev
        return False


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("baddbmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
    flagtune_op_name="baddbmm",
)
@triton.heuristics(runtime.get_heuristic_config("baddbmm"))
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    O,
    bias,
    alpha,
    beta,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    bias_batch_stride: tl.constexpr,
    bias_M_stride: tl.constexpr,
    bias_N_stride: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N
    bias += pid_b * bias_batch_stride

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=mask_a)
        b = tl.load(b_ptrs, mask=mask_b)
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

    bias_ptrs = bias + offs_m[:, None] * bias_M_stride + offs_n[None, :] * bias_N_stride

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    else:
        mask_c = True
        if not DIVISIBLE_M:
            mask_c &= offs_m[:, None] < M
        if not DIVISIBLE_N:
            mask_c &= offs_n[None, :] < N

    bi = tl.load(bias_ptrs, mask=mask_c)
    out = accumulator * alpha + bi * beta
    o = out.to(bi.dtype)
    tl.store(o_ptrs, o, mask=mask_c)


def _baddbmm_launch(bias, A, B, beta, alpha, out):
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    bbias = bias.expand(batch, M, N).contiguous()
    bias_batch_stride = bbias.stride(0)
    bias_M_stride = bbias.stride(1)
    bias_N_stride = bbias.stride(-1)

    grid = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        baddbmm_kernel[grid](
            A,
            B,
            out,
            bbias,
            alpha,
            beta,
            M,
            N,
            K,
            bias_batch_stride=bias_batch_stride,
            bias_M_stride=bias_M_stride,
            bias_N_stride=bias_N_stride,
        )


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        _baddbmm_launch(bias, A, B, beta, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def compute_bias_grad(d_output, beta, bias):
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def compute_A_grad(d_output, B, alpha):
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = bmm(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = bmm(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = bmm(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = bmm(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_out_kernel(
    A,
    B,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha,
    beta,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALPHA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    group_size = min(num_m - first_m, GROUP_M)
    in_group = pid % (GROUP_M * num_n)
    pid_m = first_m + in_group % group_size
    pid_n = in_group // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    # Wrap partial edge tiles so the inner loop only needs a K mask.  Aligned
    # tiles additionally advertise their contiguity, which is what lets the
    # backend widen the operand loads; on a partial tile the wrapped offsets are
    # genuinely non-contiguous, so the hint is left off there.
    if M % BLOCK_M == 0:
        load_rows = tl.max_contiguous(tl.multiple_of(rows % M, BLOCK_M), BLOCK_M)
    else:
        load_rows = rows % M
    if N % BLOCK_N == 0:
        load_cols = tl.max_contiguous(tl.multiple_of(cols % N, BLOCK_N), BLOCK_N)
    else:
        load_cols = cols % N
    a_ptrs = (
        A + batch * stride_ab + load_rows[:, None] * stride_am + ks[None, :] * stride_ak
    )
    b_ptrs = (
        B + batch * stride_bb + ks[:, None] * stride_bk + load_cols[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if not ALPHA_ZERO:
        for k in range(tl.cdiv(K, BLOCK_K)):
            if K % BLOCK_K == 0:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, (k * BLOCK_K + ks[None, :]) < K, other=0)
                b = tl.load(b_ptrs, (k * BLOCK_K + ks[:, None]) < K, other=0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        if not ALPHA_ONE:
            acc *= alpha

    mask = (rows[:, None] < M) & (cols[None, :] < N)
    if not BETA_ZERO:
        bias_ptrs = (
            bias
            + batch * stride_cb
            + rows[:, None] * stride_cm
            + cols[None, :] * stride_cn
        )
        bias_value = tl.load(bias_ptrs, mask, other=0).to(tl.float32)
        if not BETA_ONE:
            bias_value *= beta
        acc += bias_value
    out_ptrs = (
        out + batch * stride_ob + rows[:, None] * stride_om + cols[None, :] * stride_on
    )
    tl.store(out_ptrs, acc, mask)


# Tile shapes measured on a BW1000 DCU (gfx936, 80 CU, 64KB LDS) with the
# in-thread-transpose rewrite enabled, over the `--level core` baddbmm_out
# benchmark shapes plus the small/skinny shapes used by the accuracy tests.
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kpack)
_TILE_WAVE = (128, 256, 32, 4, 2, 8, 1)  # 16-bit, enough tiles to fill the chip
_TILE_FEW = (64, 64, 64, 4, 2, 1, 1)  # 16-bit, fewer tiles than CUs
_TILE_WAVE_FP32 = (64, 64, 32, 4, 2, 8, 1)
_TILE_FEW_FP32 = (64, 64, 128, 8, 2, 1, 1)

_NUM_CU = 80


def _tile_config(dtype, batch, M, N):
    """Pick a tile shape for one `baddbmm.out` launch.

    Large tiles amortize the operand traffic but only pay off once they still
    produce at least one full wave of workgroups; the benchmark's 2x384x384x384
    case would only reach 12 of the 80 CUs with the 128x256 tile.
    """
    wave, few = (
        (_TILE_WAVE_FP32, _TILE_FEW_FP32)
        if dtype == torch.float32
        else (_TILE_WAVE, _TILE_FEW)
    )
    tiles = batch * triton.cdiv(M, wave[0]) * triton.cdiv(N, wave[1])
    return wave if tiles >= _NUM_CU else few


def _baddbmm_out_launch(bias, A, B, beta, alpha, out):
    batch, M, K = A.shape
    N = B.shape[2]
    if batch == 0 or M == 0 or N == 0:
        return

    # Keep broadcast dimensions as zero strides instead of copying the bias.
    bias_shape = (1,) * (3 - bias.ndim) + tuple(bias.shape)
    bias_strides = (0,) * (3 - bias.ndim) + tuple(bias.stride())
    assert len(bias_shape) == 3 and all(
        size == 1 or size == target for size, target in zip(bias_shape, (batch, M, N))
    ), "Bias is not broadcastable to the output shape"
    bias_strides = tuple(
        0 if size == 1 else stride for size, stride in zip(bias_shape, bias_strides)
    )

    block_m, block_n, block_k, warps, stages, group_m, kpack = _tile_config(
        A.dtype, batch, M, N
    )
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(A.device), _in_thread_transpose():
        _baddbmm_out_kernel[grid](
            A,
            B,
            bias,
            out,
            M,
            N,
            K,
            alpha,
            beta,
            *A.stride(),
            *B.stride(),
            *bias_strides,
            *out.stride(),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            ALPHA_ZERO=alpha == 0,
            ALPHA_ONE=alpha == 1,
            BETA_ZERO=beta == 0,
            BETA_ONE=beta == 1,
            num_warps=warps,
            num_stages=stages,
            kpack=kpack,
        )


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS_HYGON BADDBMM_OUT")
    batch, M, K = A.shape
    _, _, N = B.shape
    assert (
        out.shape == (batch, M, N) and out.dtype == A.dtype
    ), "Incompatible output shape or dtype for baddbmm.out"
    if A.dtype in (torch.float16, torch.bfloat16, torch.float32):
        _baddbmm_out_launch(bias, A, B, beta, alpha, out)
    else:
        _baddbmm_launch(
            bias.contiguous(), A.contiguous(), B.contiguous(), beta, alpha, out
        )
    return out


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    logger.debug("GEMS_HYGON BADDBMM")
    return BaddbmmFunction.apply(
        bias.contiguous(),
        A.contiguous(),
        B.contiguous(),
        beta,
        alpha,
    )

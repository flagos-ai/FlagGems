import logging
import math

import paddle
import torch
import triton
import triton.language as tl
from paddle.autograd import PyLayer

from flag_gems.runtime import device_guard
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx.to(tl.int32) * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,  # pointer to the input
    elem_cnt: tl.constexpr,  # number of columns in X
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    grad_in,  # pointer to the gradient input
    grad_out,  # pointer to the gradient output
    indices,  # pointer to the input
    padding_idx,  # padding_idx
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        # grad_in may be fp32 while grad_out is a half type: the caller widens the
        # accumulator when several index elements collide on the same row, because
        # half-type atomics are emulated and collapse under that contention.
        embedding_grad = embedding_grad.to(grad_in.dtype.element_ty)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            embedding_grad = embedding_grad.to(grad_in.dtype.element_ty)
            tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx", "M"])
def embedding_backward_gather_kernel(
    grad_in,  # pointer to the gradient input
    grad_out,  # pointer to the gradient output
    indices,  # pointer to the input
    M,  # number of index elements
    padding_idx,  # padding_idx
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # number of columns in X
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per weight row, gathering the rows that select it.

    The scatter form below needs an atomic per index element plus a zeroed output
    buffer. Both hurt at small sizes: the zero-fill is a second launch, and half-type
    atomics are emulated with a CAS loop that collapses under collisions. Here every
    row is written exactly once, so there are no atomics and nothing to pre-zero --
    at the cost of rescanning the indices per row, which only pays off while
    num_weights * M * N stays small.
    """
    row = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        offs = start + tl.arange(0, BLOCK_M)
        m_mask = offs < M
        idx = tl.load(indices + offs, mask=m_mask, other=-1).to(tl.int32)
        match = (idx == row) & m_mask
        # masked-off lanes are not fetched, so only the selected rows are read
        vals = tl.load(
            grad_out + offs[:, None] * N + cols[None, :],
            mask=match[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(vals.to(tl.float32), axis=0)
    if HAS_PADDING_IDX:
        if row == padding_idx:
            acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    tl.store(grad_in + row * N + cols, acc.to(grad_in.dtype.element_ty), mask=col_mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tle.program_id(0)
    row_step = tle.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


def embedding(indices, weight, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logger.debug("GEMS EMBEDDING FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = math.prod(indices.shape)
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    # TODO: remove contiguous enforcement
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    with device_guard(weight):
        embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)

    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    # paddle's Tensor.numel() returns a device Tensor, so using it as a grid size
    # forces a device-to-host sync on every call.
    M = math.prod(indices.shape)
    N = grad_outputs.shape[-1]
    place = grad_outputs.place
    BLOCK_SIZE = triton.next_power_of_2(N)
    HAS_PADDING_IDX = padding_idx is not None

    # The gather form rescans the indices once per weight row, so its cost grows with
    # num_weights * M * N; past a few million lane-slots the atomic scatter below
    # wins. Measured crossover on H800: it is 1.3-4.0x faster up to 4M (the win peaks
    # where half-type atomics collide), and falls off a cliff right after.
    if not scale_grad_by_freq and N <= 1024 and num_weights * M * N <= 4 * 1024 * 1024:
        block_m = max(1, min(64, 8192 // BLOCK_SIZE))
        with device_guard(grad_outputs):
            # every row is written, so the buffer does not need zeroing
            grad_inputs = paddle._C_ops.empty(
                [num_weights, N], grad_outputs.dtype, place
            )
            embedding_backward_gather_kernel[num_weights,](
                grad_inputs,
                grad_outputs,
                indices,
                M,
                padding_idx,
                HAS_PADDING_IDX,
                N,
                block_m,
                BLOCK_SIZE,
            )
        return grad_inputs

    # triton lowers the scatter to native vectorized half atomics
    # (`atom.global.add.noftz.v8.f16`), but only when each lane owns several
    # contiguous columns -- at one column per lane it degenerates and costs 25x more
    # (471us vs 18us at 128x128). Keeping elements-per-lane = BLOCK_SIZE / (32 * warps)
    # at 4 or above was fastest for every benchmarked N, in both dtypes.
    num_warps = 1 if BLOCK_SIZE <= 128 else 2
    # Those half atomics accumulate in half, and so does paddle's own kernel: at 1024
    # collisions per row both land at ~1e-2 relative error against an fp32 reference,
    # so widening the accumulator here would only buy precision the framework does not
    # promise, at 1.7x the time. bfloat16 is different -- 8 mantissa bits do collapse
    # under that many additions -- so it still gets the fp32 buffer plus a cast.
    acc_in_fp32 = grad_outputs.dtype == torch.bfloat16 and M > 32 * num_weights
    # `torch.zeros(..., device=...)` costs ~70us per call through the proxy and
    # dominated this op (the kernel launch itself is 0.11us). `_C_ops.full` skips
    # the python-level shape/dtype/place parsing that `paddle.zeros` does (5.8us vs
    # 10.1us), and allocating inside the device guard keeps it on the right place.
    out_dtype = paddle.float32 if acc_in_fp32 else grad_outputs.dtype
    with device_guard(grad_outputs):
        grad_inputs = paddle._C_ops.full([num_weights, N], 0.0, out_dtype, place)

        if scale_grad_by_freq:
            indice_freq = paddle._C_ops.full([num_weights], 0.0, paddle.int32, place)
            indice_freq.stop_gradient = True
            INDICE_BLOCK_SIZE = 256
            indice_grid = (triton.cdiv(M, INDICE_BLOCK_SIZE),)
            indice_freq_kernel[indice_grid](indice_freq, indices, M, INDICE_BLOCK_SIZE)
        else:
            indice_freq = None

        embedding_backward_kernel[M,](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            HAS_PADDING_IDX,
            N,
            BLOCK_SIZE,
            num_warps=num_warps,
        )

        if scale_grad_by_freq:
            embedding_grad_scale_kernel[M,](
                grad_inputs, indice_freq, num_weights, N, BLOCK_SIZE
            )
    # `Tensor.to` costs 22.9us of arg parsing through the proxy; `_C_ops.cast` is 6.1us.
    if acc_in_fp32:
        return paddle._C_ops.cast(grad_inputs, grad_outputs.dtype)
    return grad_inputs


class Embedding(PyLayer):
    @staticmethod
    def forward(
        ctx, indices, weight, padding_idx, scale_grad_by_freq=False, sparse=False
    ):  
        if weight.requires_grad:
            ctx.save_for_backward(indices, weight)
            ctx.padding_idx = padding_idx
            ctx.scale_grad_by_freq = scale_grad_by_freq
            ctx.sparse = sparse

        return embedding(indices, weight, padding_idx, scale_grad_by_freq, sparse)

    @staticmethod
    def backward(ctx, grad_outputs):
        indices, weight = ctx.saved_tensor()
        padding_idx = ctx.padding_idx
        scale_grad_by_freq = ctx.scale_grad_by_freq
        sparse = ctx.sparse

        grad_input = embedding_backward(
            grad_outputs,
            indices,
            weight.shape[0],
            padding_idx,
            scale_grad_by_freq,
            sparse,
        )

        return None, grad_input


embedding_paddle = Embedding.apply

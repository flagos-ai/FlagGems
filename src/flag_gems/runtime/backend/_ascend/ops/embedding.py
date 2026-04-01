import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

MAX_BLOCK_SIZE = 32768
NUM_VECTOR_CORES = 48


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    M,
    N,
    ROWS_PER_CORE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    for row in range(ROWS_PER_CORE):
        row_idx_in_batch = pid * ROWS_PER_CORE + row
        if row_idx_in_batch < M:
            idx = tl.load(in_ptr + row_idx_in_batch)
            for i in range(NUM_ITERS):
                cols = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                val = tl.load(
                    weight_ptr + idx * N + cols, mask, other=0.0,
                    care_padding=False,
                )
                tl.store(out_ptr + row_idx_in_batch * N + cols, val, mask)


@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,
    elem_cnt,
    num_tasks,
    INDICE_BLOCK_SIZE: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tle.program_id(0)

    for task_id in range(pid, num_tasks, NCORE):
        block_start = task_id * INDICE_BLOCK_SIZE
        for i in range(INDICE_BLOCK_SIZE):
            off = block_start + i
            if off < elem_cnt:
                idx = tl.load(indices + off)
                tl.atomic_add(indices_freq + idx, 1)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    grad_in,
    grad_out,
    indices,
    padding_idx,
    M,
    HAS_PADDING_IDX: tl.constexpr,
    N,
    ROWS_PER_CORE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    for row in range(ROWS_PER_CORE):
        row_idx_in_batch = pid * ROWS_PER_CORE + row
        if row_idx_in_batch < M:
            row_idx = tl.load(indices + row_idx_in_batch).to(tl.int32)

            skip = HAS_PADDING_IDX and (row_idx == padding_idx)
            if not skip:
                for i in range(NUM_ITERS):
                    cols = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = cols < N
                    embedding_grad = tl.load(
                        grad_out + row_idx_in_batch * N + cols, mask, other=0.0,
                        care_padding=False,
                    )
                    if tl.constexpr(embedding_grad.dtype.is_bf16()):
                        embedding_grad = embedding_grad.to(tl.float32)
                    tl.atomic_add(grad_in + row_idx * N + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    ROWS_PER_CORE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    for row in range(ROWS_PER_CORE):
        row_idx = pid * ROWS_PER_CORE + row
        if row_idx < n_rows:
            indice_freq_val = tl.load(indice_freq + row_idx)
            if indice_freq_val > 1:
                embedding_scale = 1.0 / indice_freq_val

                for i in range(NUM_ITERS):
                    cols = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = cols < N
                    embedding_grad = tl.load(
                        grad_out + row_idx * N + cols, mask=mask,
                        care_padding=False,
                    )
                    scaled_embedding_grad = embedding_grad * embedding_scale
                    tl.store(
                        grad_out + row_idx * N + cols,
                        scaled_embedding_grad, mask=mask,
                    )


def _compute_block_sizes(N):
    BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
    NUM_ITERS = triton.cdiv(N, BLOCK_SIZE)
    return BLOCK_SIZE, NUM_ITERS


def _compute_grid(M):
    if M == 0:
        return 0, 0
    ncore = min(M, NUM_VECTOR_CORES)
    rows_per_core = triton.cdiv(M, ncore)
    return ncore, rows_per_core


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
    ):
        logger.debug("GEMS_ASCEND EMBEDDING FORWARD")
        assert not sparse, "Currently do not support sparse format"

        M = math.prod(indices.shape)
        N = weight.shape[-1]

        if M == 0:
            ctx.M = 0
            ctx.N = N
            ctx.num_weights = weight.shape[0]
            ctx.padding_idx = padding_idx
            ctx.scale_grad_by_freq = scale_grad_by_freq
            ctx.sparse = sparse
            ctx.indices = indices
            return torch.empty(
                (*indices.shape, N), device=indices.device, dtype=weight.dtype
            )

        BLOCK_SIZE, NUM_ITERS = _compute_block_sizes(N)
        ncore, rows_per_core = _compute_grid(M)

        indices = indices.contiguous()
        weight = weight.contiguous()
        output = torch.empty(
            (*indices.shape, N), device=indices.device, dtype=weight.dtype
        )

        with torch_device_fn.device(weight.device):
            embedding_kernel[ncore,](
                output, indices, weight, M, N,
                rows_per_core, NUM_ITERS, BLOCK_SIZE,
            )

        ctx.M = M
        ctx.N = N
        ctx.num_weights = weight.shape[0]
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        ctx.indices = indices

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        logger.debug("GEMS_ASCEND EMBEDDING BACKWARD")
        assert not ctx.sparse, "Currently do not support sparse format"

        if ctx.M == 0:
            grad_inputs = torch.zeros(
                (ctx.num_weights, grad_outputs.shape[-1]),
                device=grad_outputs.device,
                dtype=grad_outputs.dtype,
            )
            return grad_inputs, None, None, None, None

        grad_inputs = torch.zeros(
            (ctx.num_weights, grad_outputs.shape[-1]),
            device=grad_outputs.device,
            dtype=(
                torch.float32
                if grad_outputs.dtype is torch.bfloat16
                else grad_outputs.dtype
            ),
        )

        if ctx.scale_grad_by_freq:
            indice_freq = torch.zeros(
                (ctx.num_weights,),
                requires_grad=False,
                device=grad_outputs.device,
                dtype=torch.int32,
            )
            INDICE_BLOCK_SIZE = 256
            indice_num_tasks = triton.cdiv(ctx.M, INDICE_BLOCK_SIZE)
            indice_ncore = min(indice_num_tasks, NUM_VECTOR_CORES)

            with torch_device_fn.device(grad_outputs.device):
                indice_freq_kernel[indice_ncore,](
                    indice_freq, ctx.indices, ctx.M, indice_num_tasks,
                    INDICE_BLOCK_SIZE, indice_ncore
                )
        else:
            indice_freq = None

        BLOCK_SIZE, NUM_ITERS = _compute_block_sizes(ctx.N)
        ncore, rows_per_core = _compute_grid(ctx.M)

        HAS_PADDING_IDX = ctx.padding_idx is not None

        with torch_device_fn.device(grad_outputs.device):
            embedding_backward_kernel[ncore,](
                grad_inputs,
                grad_outputs,
                ctx.indices,
                ctx.padding_idx,
                ctx.M,
                HAS_PADDING_IDX,
                ctx.N,
                rows_per_core,
                NUM_ITERS,
                BLOCK_SIZE,
            )

        if ctx.scale_grad_by_freq:
            ncore_scale, rows_per_core_scale = _compute_grid(ctx.num_weights)
            with torch_device_fn.device(grad_outputs.device):
                embedding_grad_scale_kernel[ncore_scale,](
                    grad_inputs, indice_freq, ctx.num_weights, ctx.N,
                    rows_per_core_scale, NUM_ITERS, BLOCK_SIZE,
                )
        return (
            (
                grad_inputs.to(torch.bfloat16)
                if grad_outputs.dtype is torch.bfloat16
                else grad_inputs
            ),
            None,
            None,
            None,
            None,
        )


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(weight, indices, padding_idx, scale_grad_by_freq, sparse)


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS_ASCEND EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = grad_outputs.shape[-1]

    if M == 0:
        return torch.zeros(
            (num_weights, N),
            device=grad_outputs.device,
            dtype=grad_outputs.dtype,
        )

    grad_inputs = torch.zeros(
        (num_weights, N),
        device=grad_outputs.device,
        dtype=(
            torch.float32
            if grad_outputs.dtype is torch.bfloat16
            else grad_outputs.dtype
        ),
    )

    if scale_grad_by_freq:
        indice_freq = torch.zeros(
            (num_weights,),
            requires_grad=False,
            device=grad_outputs.device,
            dtype=torch.int32,
        )
        INDICE_BLOCK_SIZE = 256
        indice_num_tasks = triton.cdiv(M, INDICE_BLOCK_SIZE)
        indice_ncore = min(indice_num_tasks, NUM_VECTOR_CORES)

        with torch_device_fn.device(grad_outputs.device):
            indice_freq_kernel[indice_ncore,](
                indice_freq, indices, M, indice_num_tasks,
                INDICE_BLOCK_SIZE, indice_ncore
            )
    else:
        indice_freq = None

    BLOCK_SIZE, NUM_ITERS = _compute_block_sizes(N)
    ncore, rows_per_core = _compute_grid(M)

    HAS_PADDING_IDX = padding_idx is not None

    with torch_device_fn.device(grad_outputs.device):
        embedding_backward_kernel[ncore,](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            M,
            HAS_PADDING_IDX,
            N,
            rows_per_core,
            NUM_ITERS,
            BLOCK_SIZE,
        )

    if scale_grad_by_freq:
        ncore_scale, rows_per_core_scale = _compute_grid(num_weights)
        with torch_device_fn.device(grad_outputs.device):
            embedding_grad_scale_kernel[ncore_scale,](
                grad_inputs, indice_freq, num_weights, N,
                rows_per_core_scale, NUM_ITERS, BLOCK_SIZE,
            )

    return (
        grad_inputs.to(torch.bfloat16)
        if grad_outputs.dtype is torch.bfloat16
        else grad_inputs
    )

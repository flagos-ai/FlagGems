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
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.codegen_config_utils import get_codegen_config

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # accumulation dtype
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    # divide by total element count M to get mean
    mean_val = sum_val / M
    tl.store(out, mean_val)


_MEAN_BWD_I64_LIMIT = 2**31
_MEAN_BWD_PLANS = {}


@libentry()
@triton.jit
def _mean_backward_fill_kernel(
    out_ptr,
    grad_ptr,
    inv_n,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
):
    """All dims reduced: one scalar upstream gradient drives the whole output."""
    pid = ext.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(grad_ptr).to(tl.float32) * inv_n
    result = val.to(out_ptr.dtype.element_ty)
    if EVEN:
        tl.store(out_ptr + offs, result)
    else:
        tl.store(out_ptr + offs, result, mask=offs < numel)


@libentry()
@triton.jit
def _mean_backward_row_kernel(
    out_ptr,
    grad_ptr,
    inv_n,
    numel,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
):
    """Reduced dimensions form one contiguous run that ends in a single row."""
    pid = ext.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        grad = tl.load(grad_ptr + offs // N)
    else:
        grad = tl.load(grad_ptr + offs // N, mask=offs < numel, other=0.0)
    result = (grad.to(tl.float32) * inv_n).to(out_ptr.dtype.element_ty)
    if EVEN:
        tl.store(out_ptr + offs, result)
    else:
        tl.store(out_ptr + offs, result, mask=offs < numel)


@libentry()
@triton.jit
def _mean_backward_dim_kernel(
    out_ptr,
    grad_ptr,
    inv_n,
    N,
    K,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    I64: tl.constexpr,
):
    """Contiguous reduction block with a trailing kept run of length K.

    One (ROWS, BLOCK_K) tile of the broadcast gradient is stored per program,
    so the store stays a flat contiguous vector store.
    """
    pid_n = ext.program_id(0)
    pid_m = ext.program_id(1)
    pid_k = ext.program_id(2)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offs = pid_n * ROWS + tl.arange(0, ROWS)
    if I64:
        grad_offs = pid_m.to(tl.int64) * K + k_offs
        out_offs = pid_m.to(tl.int64) * N * K + n_offs[:, None] * K + k_offs[None, :]
    else:
        grad_offs = pid_m * K + k_offs
        out_offs = pid_m * N * K + n_offs[:, None] * K + k_offs[None, :]

    if EVEN_K:
        grad = tl.load(grad_ptr + grad_offs)
    else:
        grad = tl.load(grad_ptr + grad_offs, mask=k_offs < K, other=0.0)
    value = (grad.to(tl.float32) * inv_n).to(out_ptr.dtype.element_ty)
    tile = tl.broadcast_to(value[None, :], (ROWS, BLOCK_K))
    if EVEN_N and EVEN_K:
        tl.store(out_ptr + out_offs, tile)
    elif EVEN_N:
        tl.store(
            out_ptr + out_offs,
            tile,
            mask=tl.broadcast_to((k_offs < K)[None, :], (ROWS, BLOCK_K)),
        )
    elif EVEN_K:
        tl.store(
            out_ptr + out_offs,
            tile,
            mask=tl.broadcast_to((n_offs < N)[:, None], (ROWS, BLOCK_K)),
        )
    else:
        tl.store(
            out_ptr + out_offs, tile, mask=(n_offs < N)[:, None] & (k_offs < K)[None, :]
        )


@libentry()
@triton.jit
def _mean_backward_bcast_kernel(
    out_ptr,
    grad_ptr,
    inv_n,
    numel,
    SHAPE: tl.constexpr,
    SSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
):
    """General case: per-element index decomposition over constexpr shape tables."""
    pid = ext.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    idx = offs
    src = tl.zeros([BLOCK], dtype=offs.dtype)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        size = SHAPE[d]
        coord = idx % size
        idx = idx // size
        if SSTRIDE[d] != 0:
            src += coord * SSTRIDE[d]
    if EVEN:
        grad = tl.load(grad_ptr + src)
    else:
        grad = tl.load(grad_ptr + src, mask=offs < numel, other=0.0)
    result = (grad.to(tl.float32) * inv_n).to(out_ptr.dtype.element_ty)
    if EVEN:
        tl.store(out_ptr + offs, result)
    else:
        tl.store(out_ptr + offs, result, mask=offs < numel)


def _mean_dtype(inp, dtype):
    if dtype is None:
        dtype = inp.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            raise RuntimeError(
                "mean(): could not infer output dtype. Input dtype must be either "
                f"a floating point or complex dtype. Got: {dtype}"
            )
    elif not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "mean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    return dtype


def mean(inp, *, dtype=None):
    if torch.is_grad_enabled() and inp.requires_grad:
        if inp.is_complex() or (dtype is not None and dtype.is_complex):
            raise NotImplementedError(
                "FlagGems mean backward does not support complex tensors"
            )
        return _MeanFunction.apply(inp, dtype)
    logger.debug("GEMS MEAN")
    M = inp.numel()
    dtype = _mean_dtype(inp, dtype)

    if M == 0:
        out = torch.empty([], dtype=dtype, device=inp.device)
        out.fill_(float("nan"))
        return out

    inp = inp.contiguous()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@triton.jit
def mean_dim_kernel_non_inner_vec(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,  # number of threads per block along K
    VEC_SIZE: tl.constexpr,  # elements per thread (1 for FP32, 8 for FP16/BF16)
):
    # Determine accumulation and load behavior
    input_dtype = input_ptr.dtype.element_ty
    if tl.constexpr(input_dtype == tl.float16) or tl.constexpr(
        input_dtype == tl.bfloat16
    ):
        ACC_DTYPE = tl.float32
        # VEC_SIZE should be 4 or 8 for vectorization
    else:
        ACC_DTYPE = input_dtype
        # VEC_SIZE = 1 for FP32

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    # Each thread handles VEC_SIZE consecutive elements
    k_base = pid_k * BLOCK_SIZE_K * VEC_SIZE
    k_offsets = (
        k_base
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * VEC_SIZE
        + tl.arange(0, VEC_SIZE)[None, :]
    )
    # Shape: [BLOCK_SIZE_K, VEC_SIZE]
    k_mask = k_offsets < K

    # Accumulator: [BLOCK_SIZE_K, VEC_SIZE]
    acc = tl.zeros((BLOCK_SIZE_K, VEC_SIZE), dtype=ACC_DTYPE)

    base = pid_m * N * K

    for n in range(N):
        offsets = base + n * K + k_offsets
        # This will trigger vectorized load if VEC_SIZE >= 4 and aligned
        val = tl.load(input_ptr + offsets, mask=k_mask, other=0.0)
        acc += val.to(ACC_DTYPE)

    mean_val = acc / N

    # Store back
    out_offsets = pid_m * K + k_offsets
    tl.store(output_ptr + out_offsets, mean_val, mask=k_mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("mean_non_inner"))
@triton.jit
def mean_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # accumulation dtype
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        # sum along reduction axis (N) -> keep dims so axis 0 corresponds to TILE_K
        summed = tl.sum(inp, axis=0, keep_dims=True)
        # divide by N to get mean
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        sum_tile = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum_tile += inp
        summed = tl.sum(sum_tile, axis=0, keep_dims=True)
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def mean_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        summed = tl.sum(inp, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        sum_vec = tl.zeros(
            [
                TILE_N,
            ],
            dtype=cdtype,
        )
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum_vec += inp
        summed = tl.sum(sum_vec, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    summed = tl.sum(_sum, axis=1)[:, None]
    mean = summed / N
    tl.store(out, mean, row_mask)


def mean_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    """Forward-only mean over ``dim`` (no autograd wiring)."""
    logger.debug("GEMS MEAN_DIM")
    dtype = _mean_dtype(inp, dtype)

    if dim == [] or dim == ():
        # mean over all elements
        if not keepdim:
            return mean(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(mean(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)

    # -------- normalize dim to a list of ints --------
    if isinstance(dim, int):
        dim = [dim]
    else:
        try:
            dim = list(dim)
        except TypeError:
            raise TypeError(
                f"dim must be an int, iterable of ints, or [], got {type(dim)}"
            )

    dim = [d % inp.ndim for d in dim]
    # -------------------------------------------------

    if len(dim) == 1:
        dim0 = dim[0]
        N = inp.shape[dim0]  # reduction length
        # product of dims before dim0; use initializer 1 for empty slice
        M = reduce(lambda x, y: x * y, shape[:dim0], 1)
        K = reduce(lambda x, y: x * y, shape[dim0 + 1 :], 1)
        shape[dim0] = 1
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        if M * K == 0:
            return out if keepdim else out.squeeze(dim=dim0)

        if N == 0:
            out.fill_(float("nan"))
            return out if keepdim else out.squeeze(dim=dim0)

        inp = inp.contiguous()
        with torch_device_fn.device(inp.device):
            if K >= 1024:
                input_dtype = inp.dtype
                if input_dtype in (torch.float16, torch.bfloat16):
                    VEC_SIZE = 8
                    BLOCK_SIZE_K = 128
                else:
                    VEC_SIZE = 1
                    BLOCK_SIZE_K = min(triton.next_power_of_2(K), 512)
                grid = (M, triton.cdiv(K, BLOCK_SIZE_K * VEC_SIZE))
                mean_dim_kernel_non_inner_vec[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    VEC_SIZE=VEC_SIZE,
                    num_warps=(
                        8
                        if BLOCK_SIZE_K <= 128
                        else min(16, get_codegen_config().max_num_warps_per_cta)
                    ),
                )
            elif K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                mean_dim_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                grid = (M, 1, 1)
                mean_dim_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        if not keepdim:
            out = out.squeeze(dim=dim0)
        return out
    else:
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = reduce(lambda x, y: x * y, shape, 1)
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        if M == 0:
            return out if keepdim else out.squeeze(dim=dim)

        if N == 0:
            out.fill_(float("nan"))
            return out if keepdim else out.squeeze(dim=dim)

        inp = dim_compress(inp, dim)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            mean_dim_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def _normalize_mean_dims(dim, ndim):
    """Return the sorted unique reduced dimensions for a backward pass.

    ``None`` and an empty sequence mean "reduce every dimension", matching the
    forward ``mean``/``mean.dim`` contract.
    """
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dims = [dim]
    else:
        try:
            dims = list(dim)
        except TypeError:
            raise TypeError(
                f"dim must be an int, iterable of ints, or [], got {type(dim)}"
            )
    if not dims:
        return list(range(ndim))
    return sorted({d % ndim for d in dims})


def _mean_bwd_prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def _mean_bwd_pow2(value):
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _mean_bwd_flat_block(numel):
    """Pick a flat block size that keeps the grid wide enough to saturate."""
    block = 4096
    while block > 128 and block > numel:
        block //= 2
    while block > 128 and numel // block < 4:
        block //= 2
    return block


def _mean_bwd_contiguous_run(dims):
    """Return [start, stop) when ``dims`` is one contiguous run, else None."""
    start, stop = dims[0], dims[0] + 1
    for dim in dims[1:]:
        if dim == stop:
            stop = dim + 1
        else:
            return None
    return [start, stop]


def _mean_bwd_plan(shape, grad_numel, dim):
    """Build the launch plan for one backward signature.

    The plan depends only on the input shape and the reduced dimensions: the
    input dtype is carried by the output pointer and the upstream gradient is
    always contiguous when a plan is built.  Host-side work is therefore done
    once per signature, so launch-bound shapes do not pay Python setup cost.
    """
    shape = tuple(shape)
    ndim = len(shape)
    numel = _mean_bwd_prod(shape)
    dims = _normalize_mean_dims(dim, ndim)
    n_reduced = _mean_bwd_prod([shape[d] for d in dims])
    inv_n = 1.0 / n_reduced
    i64 = numel >= _MEAN_BWD_I64_LIMIT

    # All dims reduced (or nothing to reduce): a single scalar source.
    if len(dims) == ndim and grad_numel == 1:
        block = _mean_bwd_flat_block(numel)
        return (
            _mean_backward_fill_kernel,
            (triton.cdiv(numel, block),),
            (inv_n, numel, block, numel % block == 0, i64),
            8,
        )

    run = _mean_bwd_contiguous_run(dims) if dims else None
    if run is not None:
        rows = _mean_bwd_prod(shape[: run[0]])
        inner = _mean_bwd_prod(shape[run[1] :])
        if inner == 1:
            block = _mean_bwd_flat_block(numel)
            return (
                _mean_backward_row_kernel,
                (triton.cdiv(numel, block),),
                (inv_n, numel, n_reduced, block, numel % block == 0, i64),
                8,
            )
        block_k = min(4096, _mean_bwd_pow2(inner))
        tile_rows = min(max(1, 8192 // block_k), _mean_bwd_pow2(n_reduced))
        return (
            _mean_backward_dim_kernel,
            (triton.cdiv(n_reduced, tile_rows), rows, triton.cdiv(inner, block_k)),
            (
                inv_n,
                n_reduced,
                inner,
                tile_rows,
                block_k,
                block_k == inner,
                n_reduced % tile_rows == 0,
                i64,
            ),
            8 if tile_rows * block_k >= 4096 else 4,
        )

    # Generic broadcast: the source strides come from the input shape, so this
    # plan is one kernel over the flat output and is also cached by signature.
    src_stride = [0] * ndim
    acc = 1
    for d in range(ndim - 1, -1, -1):
        if d not in dims:
            src_stride[d] = acc
            acc *= shape[d]
    block = _mean_bwd_flat_block(numel)
    return (
        _mean_backward_bcast_kernel,
        (triton.cdiv(numel, block),),
        (inv_n, numel, shape, tuple(src_stride), block, numel % block == 0, i64),
        8,
    )


def _mean_backward(grad_output, input_shape, input_dtype, dim):
    """Broadcast ``grad_output / N`` over the reduced dimensions of the input.

    ``grad_output`` is the upstream gradient of the mean output; ``N`` is the
    number of reduced elements.  The input values and strides are never read,
    so an arbitrary layout of the forward input is supported.
    """
    grad = grad_output
    if not grad.is_contiguous():
        grad = grad.contiguous()
    grad_input = torch.empty(input_shape, dtype=input_dtype, device=grad.device)
    numel = _mean_bwd_prod(tuple(input_shape))
    if numel == 0:
        # Empty input or empty reduction: ATen also returns an empty gradient,
        # so there is no value to write and no NaN/inf can be observed.
        return grad_input

    key = (tuple(input_shape), input_dtype, grad.shape, grad.dtype, tuple(dim))
    plan = _MEAN_BWD_PLANS.get(key)
    if plan is None:
        plan = _mean_bwd_plan(input_shape, grad.numel(), dim)
        _MEAN_BWD_PLANS[key] = plan
    kernel, grid, args, num_warps = plan
    with torch_device_fn.device(grad_input.device):
        kernel[grid](grad_input, grad, *args, num_warps=num_warps)
    return grad_input


class _MeanFunction(torch.autograd.Function):
    """autograd bridge for ``torch.mean`` (all-reduce overload)."""

    @staticmethod
    def forward(ctx, inp, dtype):
        ctx.input_shape = tuple(inp.shape)
        ctx.input_dtype = inp.dtype
        with torch.no_grad():
            return mean(inp, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None
        dims = list(range(len(ctx.input_shape)))
        grad_input = _mean_backward(grad_output, ctx.input_shape, ctx.input_dtype, dims)
        return grad_input, None


class _MeanDimFunction(torch.autograd.Function):
    """autograd bridge for ``torch.mean(input, dim, keepdim)``."""

    @staticmethod
    def forward(ctx, inp, dim, keepdim, dtype):
        ctx.input_shape = tuple(inp.shape)
        ctx.input_dtype = inp.dtype
        ctx.dims = _normalize_mean_dims(dim, inp.ndim)
        with torch.no_grad():
            return mean_dim_comm(inp, dim, keepdim, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        grad_input = _mean_backward(
            grad_output, ctx.input_shape, ctx.input_dtype, ctx.dims
        )
        return grad_input, None, None, None


def mean_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS MEAN_DIM (wrapper)")

    if torch.is_grad_enabled() and inp.requires_grad:
        if inp.is_complex() or (dtype is not None and dtype.is_complex):
            raise NotImplementedError(
                "FlagGems mean backward does not support complex tensors"
            )
        return _MeanDimFunction.apply(inp, dim, keepdim, dtype)
    return mean_dim_comm(inp, dim, keepdim, dtype=dtype)


def mean_backward(grad_output, input_shape, input_dtype, dim=None):
    """Return the input gradient of mean, using the input shape and reduced axes."""
    input_shape = tuple(input_shape)
    dims = _normalize_mean_dims(dim, len(input_shape))
    return _mean_backward(grad_output, input_shape, input_dtype, dims)

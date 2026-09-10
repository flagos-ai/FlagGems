import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device_guard
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    amax_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amax_val)


@libentry()
@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    # `rows` enumerates the (outer, inner) pairs of the axes that are kept, where
    # K is the number of trailing elements after the reduced axis. K == 1 is the
    # innermost-reduction case and degenerates to the original `rows * N` layout,
    # so a non-innermost axis no longer needs a materialized transpose.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    inp_base = (rows // K) * N * K + rows % K
    out = out + rows

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _all = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp + inp_base + cols * K, mask, other=min_value)
        _all = tl.maximum(_all, a)
    all = tl.max(_all, axis=1)[:, None]
    tl.store(out, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMAX")
    if dim is None or len(dim) == 0:
        M = inp.size
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with device_guard(inp):
            amax_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
            )
            amax_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )  # max block size is 128k, so mid does not requires int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]

        if len(dim) == 1:
            # Reduce a single axis in place: strided addressing instead of
            # dim_compress, which would copy the whole tensor to move the axis last.
            d = dim[0]
            inp = inp.contiguous()
            N = shape[d]
            K = math.prod(shape[d + 1 :])
            M = math.prod(shape[:d]) * K
        else:
            inp = dim_compress(inp, dim)
            N = 1
            for i in dim:
                N *= shape[i]
            K = 1
            M = math.prod(inp.shape) // N
        for i in dim:
            shape[i] = 1

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with device_guard(inp):
            amax_kernel[grid](inp, out, M, N, K)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out

def amax_paddle(x: 'Tensor', dim: 'int | Sequence[int] | None' = None, keepdim: 'bool' = False,   name: 'str | None' = None, *, out: 'Tensor | None' = None) -> 'Tensor':
    if isinstance(dim, int):
        dim = [dim]
    return amax(x, dim, keepdim)
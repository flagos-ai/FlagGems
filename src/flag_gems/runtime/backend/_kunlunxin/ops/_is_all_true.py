import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

_GLOBAL_2D_MIN = 1 << 23
_TILE_BUDGET = 64 * 512


def _pick_2d_cols(n_elements):
    for block_n in (65536, 32768, 16384, 8192):
        if n_elements % block_n == 0:
            return block_n
    return 0


def _heur_block_n(args):
    n = args["N"]
    block_n = min(triton.next_power_of_2(n), 512)
    if n > 512:
        block_n = min(triton.next_power_of_2(n), 4096)
    return triton.next_power_of_2(max(block_n, 1))


def _heur_block_m(args):
    block_n = _heur_block_n(args)
    block_m = min(triton.cdiv(args["M"], 12), 64)
    block_m = min(block_m, max(_TILE_BUDGET // block_n, 1))
    return triton.next_power_of_2(max(block_m, 1))

# _is_all_true: tests if all elements of a bool tensor are True (a specialized
# torch.all that only accepts bool tensors and returns a scalar bool tensor).
#
# The generic ops/_is_all_true.py sizes stage-1 with
#   block_size = next_power_of_2(ceil(sqrt(n_elements)))
# which is the UNBOUNDED-BLOCK anti-pattern: block_size grows with sqrt(N) (N=1G
# -> 32768), so stage-1 launches a giant constexpr tile (the IR dump
# ir-is_all_true-dev1.log shows tensor<32768x...> materialized 2039x, 814 modules
# / 1001 kernel recompiles, 113MB) that ConvertTritonXPUToLLVM expands per
# element. Reuse the tuned `all` recipe: a BOUNDED block_size from
# get_block_size_1d + buffer_size_limit=2048 so the tile is chunked instead of
# materialized whole, plus the mid_size==1 early return.


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def is_all_true_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=1)
    # masked-out lanes must be True (identity for AND); do not rely on `other`.
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


@libentry()
@triton.jit
def is_all_true_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


@libentry()
@triton.jit
def is_all_true_empty_kernel(out):
    tl.store(out, True)


@libentry()
@triton.heuristics(values={"BLOCK_M": _heur_block_m, "BLOCK_N": _heur_block_n})
@triton.jit
def is_all_true_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    inp = inp + rows * N
    out = out + rows

    all_true = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        values = tl.load(inp + cols, mask=mask, other=1)
        all_true = all_true and (values != 0)
    tl.store(out, tl.reduce(all_true, axis=1, combine_fn=reduce_all)[:, None], row_mask)


def _is_all_true(inp):
    logger.debug("GEMS_KUNLUNXIN _IS_ALL_TRUE")
    assert inp.dtype == torch.bool, "Input tensor must be of type bool"

    n_elements = inp.numel()

    # all() of the empty set is True (vacuous truth).
    if n_elements == 0:
        out = torch.empty([], dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            is_all_true_empty_kernel[(1, 1, 1)](out, buffer_size_limit=2048)
        return out

    if n_elements >= _GLOBAL_2D_MIN and inp.is_contiguous():
        block_n = _pick_2d_cols(n_elements)
        if block_n:
            block_m_count = n_elements // block_n
            mid = torch.empty((block_m_count,), dtype=torch.bool, device=inp.device)
            out = torch.empty([], dtype=torch.bool, device=inp.device)
            block_mid = triton.next_power_of_2(block_m_count)
            def grid(meta):
                return (max(triton.cdiv(block_m_count, meta["BLOCK_M"]), 1),)

            with torch_device_fn.device(inp.device):
                is_all_true_kernel_2d[grid](
                    inp, mid, block_m_count, block_n, buffer_size_limit=2048
                )
                is_all_true_kernel_2[(1, 1, 1)](
                    mid, out, block_m_count, block_mid, buffer_size_limit=2048
                )
            return out

    block_size = get_block_size_1d(n_elements, inp.element_size())
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        is_all_true_kernel_1[(mid_size, 1, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        is_all_true_kernel_2[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )

    return out

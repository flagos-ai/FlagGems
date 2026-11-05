import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# Below this numel the single-pass reversed flip beats the two-pass
# materialization (ported unchanged from the pre-rewrite implementation,
# used for everything the flat 2-D kernel below does not cover).
_SMALL_NUMEL = 200000

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _rot90_copy_pw(src):
    return src


# NOTE (kunlunxin/XPU): the generic rot90 kernel is decorated with
# `@triton.autotune(configs=get_tuned_config("rot90"), key=["n_elements"])`.
# On XPU triton, autotune re-benchmarks ALL configs for every distinct
# n_elements (benchmark has many shapes/dtypes -> many distinct n) -> the launch
# path recompiles per (config, n) -> IR explosion (196MB / 10512 modules, see
# ir-rot90-dev5.log). Same family as the bernoulli_/uniform_ "don't let
# autotune/heuristics supply launch params on XPU" lesson. Fix: drop autotune,
# compute BLOCK_SIZE/num_warps in the Python wrapper (size-banded) and pass them
# explicitly. Kernel body is byte-for-byte identical to generic -> zero numeric
# change.
#
# 2026-09-14: an earlier revision replaced this kernel with a captured native
# ``aten::flip`` (``torch.library.get_kernel`` + ``call_boxed``), which made the
# measured gem *be* the reference implementation (rot90 is flip + a free view),
# so its benchmark ratio was ~1.0 by construction. That is banned for metric
# integrity: an operator implementation may not run the vendor implementation
# of its own computation. This is the honest Triton path; until the backend
# grows a vectorised reverse-lane load (Vgather negative stride, tracked in
# analysis/14 P6) it is expected to be several times slower than the vendor
# flip -- that gap is real and belongs in the numbers, not hidden.
@triton.jit
def rot90_kernel_2d(
    in_ptr,
    out_ptr,
    n_elements,
    M,
    N,
    k_norm,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m_minus_1 = M - 1
    n_minus_1 = N - 1

    if k_norm == 0:
        stride_0 = n_elements // M
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % N

        in_dim0 = out_dim0
        in_dim1 = out_dim1

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    elif k_norm == 1:
        stride_0 = n_elements // N
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % M

        in_dim0 = out_dim1
        in_dim1 = n_minus_1 - out_dim0

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    elif k_norm == 2:
        stride_0 = n_elements // M
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % N

        in_dim0 = m_minus_1 - out_dim0
        in_dim1 = n_minus_1 - out_dim1

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    else:  # k_norm == 3
        stride_0 = n_elements // N
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % M

        in_dim0 = m_minus_1 - out_dim1
        in_dim1 = out_dim0

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _launch_config(n_elements):
    # Size-banded BLOCK_SIZE / num_warps (mirrors the nvidia rot90 tune configs)
    # computed in Python and passed explicitly, so no autotune recompiles on XPU.
    if n_elements <= 4096:
        return 512, 2
    elif n_elements <= 65536:
        return 1024, 4
    elif n_elements <= 1048576:
        return 2048, 8
    else:
        return 4096, 16


def rot90_2d(inp, k, dims, out):
    """Handle the case when dims = [0, 1] using optimized Triton kernel."""
    M = inp.shape[dims[0]]
    N = inp.shape[dims[1]]
    n_elements = out.numel()
    if n_elements == 0:
        return

    k_norm = ((k % 4) + 4) % 4

    BLOCK_SIZE, num_warps = _launch_config(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(inp.device):
        rot90_kernel_2d[grid](
            inp,
            out,
            n_elements,
            M,
            N,
            k_norm,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )


def _rot90_generic(x, k_norm, dim0, dim1):
    """Pre-rewrite view-based implementation (flip/transpose + tle materialization).

    Used for ndim != 2 and for explicit non-(0, 1) dims: the flat 2-D kernel
    below assumes the MxN matrix spans the whole contiguous tensor, which only
    holds for a 2-D input.
    """
    if k_norm == 0:
        return x.clone()
    if k_norm == 1:
        if x.numel() <= _SMALL_NUMEL:
            return x.flip([dim1]).transpose(dim0, dim1)
        out_shape = list(x.shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        # Materialize the transposed view with the copy-family recipe (same as
        # permute_copy): tle takes the whole transfer, the pointwise kernel
        # keeps the rest. No `torch.ops.aten._copy_from` -- it dispatches to
        # the XPU fallback.
        transposed = x.transpose(dim0, dim1)
        if not tle_copy(transposed, out):
            _rot90_copy_pw(transposed, out0=out)
        return out.flip([dim0])
    if k_norm == 2:
        return x.flip([dim0, dim1])
    return x.flip([dim0]).transpose(dim0, dim1)


def rot90(input, k=1, dims=[0, 1]):
    logger.debug("GEMS_KUNLUNXIN ROT90")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    dim0, dim1 = dims[0], dims[1]
    k_norm = ((k % 4) + 4) % 4

    if dim0 != 0 or dim1 != 1 or x.ndim != 2:
        # Anything but a 2-D default-dims input takes the view-based path: the
        # flat kernel only models an MxN matrix spanning the whole tensor.
        return _rot90_generic(x, k_norm, dim0, dim1)

    M = x.shape[dim0]
    N = x.shape[dim1]

    # Large power-of-two numels: materialise the transpose with tle and flip the
    # *outer* axis. The flat kernel below reads its input with a reversed lane
    # order (D-017: no vectorised path on this backend), while this ordering
    # lands flip's block path -- contiguous inner run, and a grid that covers the
    # task space exactly, which is what keeps flip's index clamp out of the
    # kernel (an inexact grid costs ~10x on a block copy, e.g. 400x800). Measured
    # 2.6x-9x faster than the flat kernel at 512^2 / 1024^2 / 2048^2 (2026-09-18).
    if k_norm == 1 and x.numel() > _SMALL_NUMEL and (x.numel() & (x.numel() - 1)) == 0:
        wide = torch.empty([N, M], device=x.device, dtype=x.dtype)
        if tle_copy(x.transpose(dim0, dim1), wide):
            return wide.flip([dim0])
        # tle cannot express this transfer -> fall through to the flat kernel.

    if k_norm == 0 or k_norm == 2:
        out_shape = list(x.shape)
    else:
        out_shape = list(x.shape)
        out_shape[dim0] = N
        out_shape[dim1] = M

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    rot90_2d(x, k, dims, out)
    return out

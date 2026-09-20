import logging

import torch
import triton
from triton import language as tl

from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import c_contiguous_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_KUNLUNXIN REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    if not inp.is_contiguous():
        inp = inp.contiguous()
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    out_view_stride = c_contiguous_stride(out_view_shape)

    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(output, out_view_shape, out_view_stride)
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask, other=0)
    out_offset = cumsum - repeats

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS_KUNLUNXIN REPEAT_INTERLEAVE_TENSOR")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"

    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()

    assert result_size >= 0, "repeats can not be negative"

    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    grid = (size,)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


@triton.jit
def repeat_interleave_self_tensor_kernel(
    inp,
    out,
    cumsum,
    repeats,
    D,
    rsum,
    inner,
    BLOCK_I: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # 3D grid = (D, outer, chunks): parallelize the inner (contiguous) dim across
    # programs instead of serial-looping it inside one program. Separate program_id
    # axes also avoid the flat-grid pid//D / pid%D in-kernel division, which the XPU
    # OffsetAnalysis miscompiles for pid>=12 (COMPILER_ISSUES.md #5).
    j = ext.program_id(axis=0)  # index along the repeat dim (0..D-1)
    o = ext.program_id(axis=1)  # index along the flattened outer dims
    ck = ext.program_id(axis=2)  # chunk index over the inner dim
    r = tl.load(repeats + j)
    tl.device_assert(r >= 0, "repeats can not be negative")
    start = tl.load(cumsum + j) - r
    base_in = (o * D + j) * inner
    base_out = (o * rsum + start) * inner
    cols = ck * BLOCK_I + tl.arange(0, BLOCK_I)
    # load the input row chunk once, then broadcast-store it r times to the
    # consecutive output rows (read traffic = input body, not k*input body).
    if NEED_MASK:
        m = cols < inner
        v = tl.load(inp + base_in + cols, mask=m, other=0)
        for rep in range(0, r):
            tl.store(out + base_out + rep * inner + cols, v, mask=m)
    else:
        v = tl.load(inp + base_in + cols)
        for rep in range(0, r):
            tl.store(out + base_out + rep * inner + cols, v)


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_KUNLUNXIN REPEAT_INTERLEAVE_SELF_TENSOR")

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim, but got \
                repeats.size(0) = {} and input.size({}) = {}".format(
                repeats.size(0), dim, inp_shape[dim]
            )
        )

    repeats = repeats.contiguous()
    inp = inp.contiguous()
    D = inp_shape[dim]
    outer = 1
    inner = 1
    for s in inp_shape[:dim]:
        outer *= s
    for s in inp_shape[dim + 1 :]:
        inner *= s

    if inner == 1:
        indices = repeat_interleave_tensor(repeats)
        return torch.index_select(inp, dim, indices)

    cumsum = repeats.cumsum(axis=0)
    rsum = int(cumsum[-1].item())
    out_shape = inp_shape[:dim] + [rsum] + inp_shape[dim + 1 :]
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    block_i = min(max(triton.next_power_of_2(inner), 64), 32768)
    need_mask = inner % block_i != 0
    chunks = triton.cdiv(inner, block_i)
    grid = (D, outer, chunks)
    repeat_interleave_self_tensor_kernel[grid](
        inp,
        out,
        cumsum,
        repeats,
        D,
        rsum,
        inner,
        BLOCK_I=block_i,
        NEED_MASK=need_mask,
        num_warps=8,
        buffer_size_limit=8192,
    )
    return out

import math
import logging

import torch
import triton
from triton import language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

log = logging.getLogger(__name__)

# PPL hardware limit: 4 cores × 65536 CTAs/core × 1024 elements/CTA = 2^28
_MAX_ELEMENTS_PER_LAUNCH = 268435456  # 256 MiElements (1 GiB for f32)

_REPEAT_CODEGEN_CONFIG = CodeGenConfig(
    max_tile_size=1024,
    max_grid_size=(2147483647, 1, 1),
    max_num_warps_per_cta=32,
    prefer_block_pointer=False,
    prefer_1d_tile=True,
)


@pointwise_dynamic(
    num_inputs=1,
    promotion_methods=[(0, "DEFAULT")],
    config=_REPEAT_CODEGEN_CONFIG,
)
@triton.jit
def _repeat_copy(x):
    return x


def _launch_chunked(inp, out, shape, in_strides, out_strides,
                    in_offset, out_offset, max_elem, kernel):
    """Recursively split the interleaved task-space so each launch fits
    within the PPL per-launch element limit.

    * Round 1: split repeat dimensions (even indices).  Their input strides
      are 0 so only the output offset changes — the kernel sees identical
      memory-access patterns across chunks.

    * Round 2: split data dimensions (odd indices) as a last resort.  Both
      input and output offsets change.

    Chunks that share the same shape reuse the same compiled kernel.  At
    most two distinct shapes appear per split dimension (full-sized chunks
    + one ragged tail).
    """
    total = math.prod(shape)
    ndim = len(shape)

    if total <= max_elem:
        in_view = StridedBuffer(inp, shape, in_strides, offset=in_offset)
        out_view = StridedBuffer(out, shape, out_strides, offset=out_offset)
        kernel(in_view, out0=out_view)
        return

    # Prefer repeat dimensions (even indices, 0 input stride) —
    # only the output offset advances, keeping the kernel's memory-access
    # pattern identical across chunks.  Fall back to data dims (odd indices).
    for i in list(range(0, ndim, 2)) + list(range(1, ndim, 2)):
        ri = shape[i]
        if ri <= 1:
            continue
        per_unit = total // ri
        if per_unit == 0:
            continue
        chunk_ri = min(max(1, max_elem // per_unit), ri)

        for start in range(0, ri, chunk_ri):
            size = min(chunk_ri, ri - start)
            chunk_shape = list(shape)
            chunk_shape[i] = size
            chunk_in_offset = in_offset + start * in_strides[i]
            chunk_out_offset = out_offset + start * out_strides[i]
            _launch_chunked(inp, out, chunk_shape, in_strides, out_strides,
                            chunk_in_offset, chunk_out_offset, max_elem, kernel)
        return

    raise RuntimeError(
        f"repeat: cannot split shape {shape} to fit max_elem={max_elem}"
    )


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    """repeat via StridedBuffer + pointwise copy.

    Uses a 0-stride interleaved view so the kernel is a plain copy — no %
    or // arithmetic, avoiding PPL arithmetic conversion issues on sophgo.

    The oversized max_grid forces monolithic kernel mode (one_tile_per_cta=1),
    sidestepping PPL's grid-stride-loop truncation at 8 iterations.
    Tensors exceeding the PPL per-launch limit (~256M elements) are
    automatically split into multiple launches.
    """
    log.debug("SOPHGO GEMS REPEAT")

    in_shape = list(inp.shape)
    sizes = list(sizes)

    # Align ranks: pad with 1s on the left for the shorter one
    if len(sizes) > len(in_shape):
        in_shape = [1] * (len(sizes) - len(in_shape)) + in_shape
    elif len(in_shape) > len(sizes):
        sizes = [1] * (len(in_shape) - len(sizes)) + sizes

    rank = len(in_shape)
    inp = inp.reshape(in_shape)
    inp_stride = list(inp.stride())
    out_shape = [in_shape[i] * sizes[i] for i in range(rank)]

    # Empty output
    if any(s == 0 for s in sizes):
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Build interleaved task space: [s0, a0, s1, a1, ..., sn, an]
    interleaved_shape = []
    in_view_strides = []
    out_view_strides = []
    for i in range(rank):
        interleaved_shape.append(sizes[i])
        interleaved_shape.append(in_shape[i])
        in_view_strides.append(0)                      # repeat dim: 0-stride
        in_view_strides.append(inp_stride[i])           # data dim: input stride
        out_view_strides.append(in_shape[i] * out.stride(i))  # repeat dim
        out_view_strides.append(out.stride(i))                # data dim

    # Pre-instantiate the kernel once so all chunks share the same compiled
    # binary (same ndim → same kernel; uniform chunk shapes → same grid).
    ndim = len(interleaved_shape)
    kernel = _repeat_copy.instantiate(ndim)
    _launch_chunked(inp, out, interleaved_shape, in_view_strides,
                    out_view_strides, 0, 0, _MAX_ELEMENTS_PER_LAUNCH, kernel)
    return out

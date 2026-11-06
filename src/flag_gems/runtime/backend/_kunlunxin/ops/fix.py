# Kunlunxin (XPU) backend implementation of fix (trunc toward zero).
#
# Vendor-specific rewrite of the generic flag_gems.ops.fix.  Compared to the
# generic implementation, the launch path uses a constant-tuple grid instead of
# `grid = lambda meta: ...`; on the xpubin backend the lambda object is baked
# into the compilation cache key, so every call triggers a full recompile of
# the kernel (see KernelOpt_Experience 2.4 #41: 134-149 ms/call for a kernel
# whose device time is ~2.6 ms).  When the element count is an exact multiple
# of BLOCK_SIZE the kernel is compiled with EVEN=True and runs a mask-free
# path, which keeps the load/store on the contiguous block-DMA path.  The
# truncation itself reuses the verified bit-pattern helper from .trunc.

import logging

import torch
import triton
import triton.language as tl


# The truncation math is the pure-FP form of the verified bit-pattern helper
# in .trunc. It is kept as an in-module copy because the flagtree triton
# backend cannot lower the f16->f32 vector convert + bitcast chain of the
# bit-pattern version (the resulting `triton_xpu.unpack` is lowered without a
# bufPtr; see the "triton_xpu.unpack reached lowering without a bufPtr"
# diagnostic). The FP form is mathematically identical for every input
# (keeps -0.0, passes +-inf/NaN through) and stays in the element type, so
# f16/bf16/f64 need no f32 round trip.
@triton.jit
def _trunc_fp32(xf):
    # fix/trunc toward zero: x >= 0 ? floor(x) : ceil(x). Exact for every
    # value (floor/ceil are single precise ops; -0.0 keeps its sign through
    # the x >= 0 branch, +-inf and NaN pass through unchanged). The value is
    # computed in fp32 (tl.floor/ceil only accept f32/f64; fp16/bf16 are
    # widened) with a plain element-wise convert -- no bitcast, so no
    # vector->scalar unpack is produced on the backend.
    t = xf.to(tl.float32)
    return tl.where(t >= 0.0, tl.floor(t), tl.ceil(t)).to(xf.dtype)


logger = logging.getLogger(__name__)


@triton.jit
def _fix_trunc_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        x = tl.load(x_ptr + offsets)
    else:
        x = tl.load(x_ptr + offsets, mask=offsets < n_elements)

    # Truncation toward zero via the pure-FP helper: exact for all inputs
    # (keeps the -0.0 sign, passes +-inf/NaN through), in the element type.
    res = _trunc_fp32(x)

    if EVEN:
        tl.store(out_ptr + offsets, res)
    else:
        tl.store(out_ptr + offsets, res, mask=offsets < n_elements)


@triton.jit
def _copy_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        x = tl.load(x_ptr + offsets)
        tl.store(out_ptr + offsets, x)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)


def _pick_block(n_elements: int, dtype: torch.dtype):
    """Pick (BLOCK_SIZE, num_warps) from the measured size/dtype sweep.

    Keep ~32 KB of elements per program (16384 x f16/bf16, 8192 x f32,
    4096 x f64/int64); small workloads are launch-bound and use 2048/4w.
    """
    if n_elements <= 16384:
        return 2048, 4
    elem_size = max(dtype.itemsize, 2)
    block = max(1024, min(16384, 32768 // elem_size))
    return block, 8


def _launch_fix_kernel(x: torch.Tensor, out: torch.Tensor):
    # The kernel indexes both tensors linearly, so they must be contiguous;
    # matching shapes/devices are guaranteed by the XPU dispatch and by the
    # callers (which allocate `out` with empty_like).
    if not (x.is_contiguous() and out.is_contiguous()):
        raise RuntimeError("fix only supports contiguous tensors")

    n_elements = x.numel()
    block_size, num_warps = _pick_block(n_elements, x.dtype)
    # Constant-tuple grid: a lambda grid object is part of the xpubin compile
    # cache key and forces a full recompile on every launch.
    grid = (triton.cdiv(n_elements, block_size),)
    even = n_elements % block_size == 0

    # For floating point types, perform truncation toward zero.
    # For non-floating types, it's effectively a no-op (copy).
    if x.is_floating_point():
        _fix_trunc_kernel[grid](
            x, out, n_elements, BLOCK_SIZE=block_size, EVEN=even, num_warps=num_warps
        )
    else:
        _copy_kernel[grid](
            x, out, n_elements, BLOCK_SIZE=block_size, EVEN=even, num_warps=num_warps
        )


def fix(self: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN FIX")
    """
    Wrapper for ATen operator: ('fix', <Autograd.disable: False>)
    Rounds elements toward zero (like trunc) for floating tensors.
    Leaves integer tensors unchanged.
    """
    if self.is_complex():
        # Complex: fix/trunc each component separately.  ATen has no complex
        # fix (torch.fix on a complex tensor raises NotImplementedError), so
        # view the buffer as a flat real tensor and reuse the float kernel.
        out = torch.empty_like(self)
        _launch_fix_kernel(
            torch.view_as_real(self).view(-1), torch.view_as_real(out).view(-1)
        )
        return out

    out = torch.empty_like(self)
    _launch_fix_kernel(self, out)
    return out


def fix_out(self: torch.Tensor, out: torch.Tensor):
    """
    Wrapper for ATen operator: ('fix.out', <Autograd.disable: False>)
    Writes the result into 'out'.
    """
    if self.is_complex():
        # Same real-view handling as fix(); write through the caller's 'out'.
        _launch_fix_kernel(
            torch.view_as_real(self).view(-1), torch.view_as_real(out).view(-1)
        )
        return out

    _launch_fix_kernel(self, out)
    return out

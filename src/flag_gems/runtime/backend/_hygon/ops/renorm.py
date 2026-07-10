import logging
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)
you

@libentry()
@triton.jit
def renorm_fused_kernel(
    X,
    Y,
    num_slices,
    outer_size,
    inner_size,
    p_val,
    maxnorm,
    stride_slice,
    stride_outer,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid >= num_slices:
        return

    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    base = X + pid * stride_slice
    out_base = Y + pid * stride_slice
    offs = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=cdtype)

    for o in range(outer_size):
        block_base = base + o * stride_outer
        for off in range(0, inner_size, BLOCK_SIZE):
            cols = off + offs
            mask = cols < inner_size
            x_vals = tl.load(block_base + cols, mask=mask, other=0.0).to(cdtype)
            if p_val == 2.0:
                powered = x_vals * x_vals
            else:
                powered = tl_extra_shim.pow(tl.abs(x_vals), p_val)
            acc += powered

    sum_val = tl.sum(acc)
    if p_val == 2.0:
        norm = tl_extra_shim.sqrt(sum_val)
    else:
        norm = tl_extra_shim.pow(sum_val, 1.0 / p_val)

    eps = 1e-12
    scale = maxnorm / tl.maximum(norm, eps)
    scale = tl.minimum(scale, 1.0)

    for o in range(outer_size):
        block_base = base + o * stride_outer
        out_block_base = out_base + o * stride_outer
        for off in range(0, inner_size, BLOCK_SIZE):
            cols = off + offs
            mask = cols < inner_size
            x_vals = tl.load(block_base + cols, mask=mask, other=0.0).to(cdtype)
            y_vals = x_vals * scale
            tl.store(out_block_base + cols, y_vals.to(X.dtype.element_ty), mask=mask)


def _compute_block_size(inner_size):
    return min(triton.next_power_of_2(inner_size), 128)


def renorm(input, p, dim, maxnorm):
    logger.debug("GEMS RENORM")

    if dim < 0:
        dim = input.ndim + dim

    if input.is_contiguous():
        num_slices = input.shape[dim]
        outer_size = input.shape[:dim].numel() if dim > 0 else 1
        inner_size = input.shape[dim+1:].numel() if dim < input.ndim - 1 else 1

        stride_slice = input.stride(dim)
        stride_outer = input.stride(dim - 1) if dim > 0 else 0

        output = torch.empty_like(input)
        BLOCK = _compute_block_size(inner_size)
        grid = (num_slices,)

        with torch_device_fn.device(input.device):
            renorm_fused_kernel[grid](
                input, output,
                num_slices, outer_size, inner_size,
                p, maxnorm,
                stride_slice, stride_outer,
                BLOCK_SIZE=BLOCK,
            )
        return output
    else:
        ndim = input.ndim
        perm = list(range(ndim))
        perm.remove(dim)
        perm.insert(0, dim)
        inv_perm = [perm.index(i) for i in range(ndim)]
        x_perm = input.permute(perm).contiguous()
        result = renorm(x_perm, p, 0, maxnorm)
        return result.permute(inv_perm)


def renorm_(input, p, dim, maxnorm):
    logger.debug("GEMS RENORM_")

    if dim < 0:
        dim = input.ndim + dim

    if input.is_contiguous():
        num_slices = input.shape[dim]
        outer_size = input.shape[:dim].numel() if dim > 0 else 1
        inner_size = input.shape[dim+1:].numel() if dim < input.ndim - 1 else 1

        stride_slice = input.stride(dim)
        stride_outer = input.stride(dim - 1) if dim > 0 else 0

        BLOCK = _compute_block_size(inner_size)
        grid = (num_slices,)

        with torch_device_fn.device(input.device):
            renorm_fused_kernel[grid](
                input, input,
                num_slices, outer_size, inner_size,
                p, maxnorm,
                stride_slice, stride_outer,
                BLOCK_SIZE=BLOCK,
            )
        return input
    else:
        ndim = input.ndim
        perm = list(range(ndim))
        perm.remove(dim)
        perm.insert(0, dim)
        inv_perm = [perm.index(i) for i in range(ndim)]
        x_perm = input.permute(perm).contiguous()
        renorm_(x_perm, p, 0, maxnorm)
        input.copy_(x_perm.permute(inv_perm))
        return input
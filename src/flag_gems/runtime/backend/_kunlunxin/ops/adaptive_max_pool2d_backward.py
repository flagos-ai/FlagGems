import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_max_pool2d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_elems,
    in_h,
    in_w,
    out_h,
    out_w,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    in_hw = in_h * in_w
    nc = safe_offsets // in_hw
    rem = safe_offsets % in_hw
    h = rem // in_w
    w = rem % in_w

    my_flat = h * in_w + w

    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    out_per_nc = out_h * out_w
    gop = grad_output_ptr + nc * out_per_nc
    iop = indices_ptr + nc * out_per_nc

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for oh in range(0, MAX_H):
        o_h = h_min + oh
        h_ok = o_h < h_max
        c_h = tl.minimum(o_h, out_h - 1)
        for ow in range(0, MAX_W):
            o_w = w_min + ow
            w_ok = o_w < w_max
            c_w = tl.minimum(o_w, out_w - 1)
            active = mask & h_ok & w_ok
            o_off = c_h * out_w + c_w
            idx = tl.load(iop + o_off).to(tl.int32)
            val = tl.load(gop + o_off).to(tl.float32)
            acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _adaptive_max_pool2d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_out,
    out_per_nc,
    in_hw,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_out
    idx = tl.load(indices_ptr + offsets).to(tl.int32)
    val = tl.load(grad_output_ptr + offsets).to(tl.float32)
    nc = offsets // out_per_nc
    tl.store(
        grad_input_ptr + nc * in_hw + idx,
        val.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def adaptive_max_pool2d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL2D_BACKWARD")

    input_is_3d = self.dim() == 3
    if input_is_3d:
        self = self.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        indices = indices.unsqueeze(0)

    self = self.contiguous()
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    n_in = in_n * in_c * in_h * in_w
    if n_in == 0 or grad_output.numel() == 0:
        grad_input = torch.zeros_like(self)
        return grad_input.squeeze(0) if input_is_3d else grad_input

    exact = (in_h % out_h == 0) and (in_w % out_w == 0)

    with torch_device_fn.device(self.device):
        if exact:
            grad_input = torch.zeros_like(self)
            n_out = grad_output.numel()
            _adaptive_max_pool2d_backward_scatter_kernel[(triton.cdiv(n_out, 256),)](
                grad_output,
                indices,
                grad_input,
                n_out,
                out_h * out_w,
                in_h * in_w,
                BLOCK=256,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        else:
            grad_input = torch.empty_like(self)
            max_h = 1 if in_h % out_h == 0 else (out_h // in_h + 2)
            max_w = 1 if in_w % out_w == 0 else (out_w // in_w + 2)
            _adaptive_max_pool2d_backward_gather_kernel[(triton.cdiv(n_in, 128),)](
                grad_output,
                indices,
                grad_input,
                n_in,
                in_h,
                in_w,
                out_h,
                out_w,
                MAX_H=max_h,
                MAX_W=max_w,
                BLOCK=128,
                num_warps=2,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    return grad_input.squeeze(0) if input_is_3d else grad_input

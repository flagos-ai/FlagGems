import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_max_pool3d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_elems,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    in_hw = in_h * in_w
    in_spatial = in_d * in_hw
    nc = safe_offsets // in_spatial
    rem = safe_offsets % in_spatial
    d = rem // in_hw
    rem2 = rem % in_hw
    h = rem2 // in_w
    w = rem2 % in_w

    my_flat = d * in_hw + h * in_w + w

    d_min = (d * out_d) // in_d
    d_max = tl.minimum(((d + 1) * out_d + in_d - 1) // in_d, out_d)
    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    out_per_nc = out_d * out_h * out_w
    gop = grad_output_ptr + nc * out_per_nc
    iop = indices_ptr + nc * out_per_nc

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for od in tl.static_range(0, MAX_D):
        o_d = d_min + od
        d_ok = o_d < d_max
        c_d = tl.minimum(o_d, out_d - 1)
        for oh in tl.static_range(0, MAX_H):
            o_h = h_min + oh
            h_ok = o_h < h_max
            c_h = tl.minimum(o_h, out_h - 1)
            for ow in tl.static_range(0, MAX_W):
                o_w = w_min + ow
                w_ok = o_w < w_max
                active = mask & d_ok & h_ok & w_ok
                c_w = tl.minimum(o_w, out_w - 1)
                o_off = c_d * (out_h * out_w) + c_h * out_w + c_w
                idx = tl.load(iop + o_off).to(tl.int32)
                val = tl.load(gop + o_off).to(tl.float32)
                acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def adaptive_max_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D_BACKWARD")

    grad_output = grad_output.contiguous()
    self = self.contiguous()
    in_n, in_c, in_d, in_h, in_w = self.shape
    out_d, out_h, out_w = grad_output.shape[-3:]

    grad_input = torch.empty_like(self)

    n_in = grad_input.numel()
    if n_in == 0 or grad_output.numel() == 0:
        return grad_input

    max_d = 1 if in_d % out_d == 0 else (out_d // in_d + 2)
    max_h = 1 if in_h % out_h == 0 else (out_h // in_h + 2)
    max_w = 1 if in_w % out_w == 0 else (out_w // in_w + 2)

    # Use the indices produced by the forward pass. Recomputing them here
    # breaks ties differently than ATen (the forward selects the argmax with
    # its own ordering), which moves gradient with grad_output=ones to the
    # wrong input elements. The indices are int64 linear indices, so narrow
    # to int32 for the gather kernel.
    indices = indices.contiguous().to(torch.int32)
    with torch_device_fn.device(self.device):
        _adaptive_max_pool3d_backward_gather_kernel[(triton.cdiv(n_in, 128),)](
            grad_output,
            indices,
            grad_input,
            n_in,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            MAX_D=max_d,
            MAX_H=max_h,
            MAX_W=max_w,
            BLOCK=128,
            num_warps=2,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return grad_input

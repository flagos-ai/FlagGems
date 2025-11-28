import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def trace_kernel(
    inp_ptr,
    out_ptr,
    num_diag,
    stride0,
    stride1,
    BLOCK_SIZE: tl.constexpr,
):
    inp_dtype = inp_ptr.type.element_ty
    if inp_dtype.is_int():
        acc_dtype = tl.int64
        other_val = 0
    elif inp_dtype == tl.float64:
        acc_dtype = tl.float64
        other_val = 0.0
    else:
        acc_dtype = tl.float32
        other_val = 0.0

    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)

    diag_stride = stride0 + stride1

    for i in range(0, tl.cdiv(num_diag, BLOCK_SIZE)):
        block_start = i * BLOCK_SIZE
        current_indices = block_start + tl.arange(0, BLOCK_SIZE)

        mask = current_indices < num_diag

        ptr_offsets = current_indices * diag_stride
        current_ptrs = inp_ptr + ptr_offsets

        vals = tl.load(current_ptrs, mask=mask, other=other_val)

        acc += vals.to(acc_dtype)

    final_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr, final_sum.to(out_ptr.type.element_ty))


def trace(self):
    logger.debug("GEMS TRACE")

    if self.ndim != 2:
        raise RuntimeError(
            f"trace: expected a 2D tensor, but got a {self.ndim}D tensor"
        )

    M, N = self.shape
    stride0, stride1 = self.stride()
    num_diag = min(M, N)
    if num_diag == 0:
        if self.dtype.is_floating_point:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        else:
            return torch.tensor(0, dtype=torch.int64, device=self.device)

    if self.dtype.is_floating_point:
        output_dtype = self.dtype
    else:
        output_dtype = torch.int64
    out = torch.empty((), dtype=output_dtype, device=self.device)

    grid = (1,)
    BLOCK_SIZE = 1024
    if num_diag < BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(num_diag)
        if BLOCK_SIZE == 0:
            BLOCK_SIZE = 1

    with torch_device_fn.device(self.device):
        trace_kernel[grid](
            self,
            out,
            num_diag,
            stride0,
            stride1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out

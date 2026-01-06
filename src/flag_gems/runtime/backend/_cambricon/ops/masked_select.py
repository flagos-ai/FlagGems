import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_select"), key=["n_elements"])
@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_p = tl.num_programs(axis=0)
    step = BLOCK_SIZE * num_p

    offset_start = pid * BLOCK_SIZE

    d_width = (
        inp_ptr.dtype.element_ty.primitive_bitwidth
    )  # floor_div(8) in one line throws compiling error
    small_out = n_elements.to(tl.int64) * (d_width // 8) <= 2**31

    for offset in tl.range(offset_start, n_elements, step):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(
            tl.int1
        )
        out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0) - 1

        # not contiguous, the underlying discrete offset is in byte
        if small_out:
            # prefix_sum may be in float32 type
            out_offset_int32 = out_offset.to(tl.int32)
            tl.store(out_ptr + out_offset_int32, inp, mask=(select_mask and mask))
        else:
            out_offset_int64 = out_offset.to(tl.int64)
            tl.store(out_ptr + out_offset_int64, inp, mask=(select_mask and mask))


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_select"), key=["n_elements"])
@triton.jit
def cast_bool2float_kernel(
    select_mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_p = tl.num_programs(axis=0)
    step = BLOCK_SIZE * num_p

    offset_start = pid * BLOCK_SIZE

    for offset in tl.range(offset_start, n_elements, step):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        tl.store(out_ptr + offsets, select_mask, mask=mask)


def masked_select(inp, mask):
    logger.debug("GEMS_CAMBRICON MASKED SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.contiguous()
    mask = mask.contiguous()

    mask_flattened = mask.ravel()

    n_elements = inp.numel()
    mask_faster = mask_flattened
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(inp.device):
        if n_elements <= 2**24:
            # faster cumsum in float type
            mask_faster = torch.empty_like(mask_flattened, dtype=torch.float32)
            cast_bool2float_kernel[grid](mask, mask_faster, n_elements)

        prefix_sum = mask_faster.cumsum(axis=0)
        out = torch.empty(
            int(prefix_sum[-1].item()), dtype=inp.dtype, device=inp.device
        )

        masked_select_kernel[grid](inp, mask_flattened, prefix_sum, out, n_elements)
    return out

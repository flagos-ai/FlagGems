import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest1d"), key=["N", "C", "OL"]
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest1d"))
@triton.jit
def upsample_nearest1d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OL,
    IL,
    reciprocal_scale_l,
    BLOCK_SIZE: tl.constexpr,
    SAME_L: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ol = idx % OL
    if SAME_L:
        il = ol
    else:
        il = tl.minimum(
            tl.math.floor(ol.to(tl.float32) * reciprocal_scale_l).to(tl.int32), IL - 1
        )

    offset_o = nc_iter * OL + ol
    offset_i = nc_iter * IL + il
    src_index_stride = nc_stride * IL
    dst_index_stride = nc_stride * OL

    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride


def upsample_nearest1d(
    input: torch.Tensor,
    output_size: Optional[Tuple[int]] = None,
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST1D")
    assert input.device.type == device
    assert input.ndim == 3, "The ndim of input must be 3"
    assert (
        output_size is not None or scales is not None
    ), "Either output_size or scales should be defined."

    OL = output_size[0] if output_size is not None else int(input.shape[2] * scales)
    N, C, IL = input.shape

    if scales is not None:
        reciprocal_scale_l = float(
            torch.tensor(1.0 / scales, dtype=torch.float32).item()
        )
    else:
        # Use float32 division to match PyTorch's behavior
        reciprocal_scale_l = float(
            (
                torch.tensor(IL, dtype=torch.float32)
                / torch.tensor(OL, dtype=torch.float32)
            ).item()
        )

    # allocate output
    output = torch.empty((N, C, OL), device=input.device, dtype=input.dtype)
    total_threads = OL
    grid = lambda meta: (
        triton.cdiv(total_threads, meta["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

    with torch_device_fn.device(input.device):
        upsample_nearest1d_kernel[grid](
            output,
            input,
            N,
            C,
            OL,
            IL,
            reciprocal_scale_l,
        )
    return output

from typing import Optional

import torch


def mm_w8a8_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Hopper FP8 W8A8 matrix multiplication implementation."""
    from flag_gems.runtime.backend._nvidia.hopper.ops.mm_w8a8 import (
        mm_w8a8_fp8 as hopper_mm_w8a8_fp8,
    )

    return hopper_mm_w8a8_fp8(a, b, out_dtype=out_dtype)

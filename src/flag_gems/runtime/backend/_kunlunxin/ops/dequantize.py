# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _dequantize_kernel(x, scale, zero_point, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(output + offsets, (values - zero_point) * scale, mask=mask)


def dequantize(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DEQUANTIZE")
    if not input.is_quantized:
        raise RuntimeError("dequantize expects a quantized tensor")
    if input.qscheme() not in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        raise NotImplementedError(
            "Kunlunxin dequantize supports per-tensor quantization only."
        )

    int_repr = input.int_repr().to(input.device)
    output = torch.empty(int_repr.shape, dtype=torch.float32, device=input.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    with torch_device_fn.device(input.device):
        _dequantize_kernel[(triton.cdiv(n_elements, 1024),)](
            int_repr,
            float(input.q_scale()),
            int(input.q_zero_point()),
            output,
            n_elements,
            BLOCK=1024,
        )
    return output

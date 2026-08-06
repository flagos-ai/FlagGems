# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch


_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)


def avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    result = torch.ops.aten.avg_pool3d_backward.default.redispatch(
        _CPU_KEYSET,
        grad_output.to(device="cpu", dtype=torch.float32),
        torch.empty(input.shape, device="cpu", dtype=torch.float32),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    return result.to(device=input.device, dtype=input.dtype)

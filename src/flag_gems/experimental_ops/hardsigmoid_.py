# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = x + 3.0
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 6.0)
    y = y / 6.0
    tl.store(x_ptr + offsets, y, mask=mask)


_hardsigmoid_triton = hardsigmoid_


def hardsigmoid_(*args, **kwargs):
    # Extract input tensor (supports positional or keyword: 'input' or 'self')
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if x is None:
        raise ValueError("hardsigmoid_ expects a tensor as the first argument.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("hardsigmoid_ expects a torch.Tensor as input.")
    if not x.is_floating_point():
        raise TypeError("hardsigmoid_ only supports floating point tensors.")
    if x.device.type != "cuda":
        raise RuntimeError("hardsigmoid_ Triton kernel requires a CUDA tensor.")

    BLOCK_SIZE = 1024

    def launch(t: torch.Tensor):
        n_elements = t.numel()
        if n_elements == 0:
            return
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _hardsigmoid_triton[grid](t, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if not x.is_contiguous():
        tmp = x.contiguous()
        launch(tmp)
        x.copy_(tmp)
    else:
        launch(x)

    return x

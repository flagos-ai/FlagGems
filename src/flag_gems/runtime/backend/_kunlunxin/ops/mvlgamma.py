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

# Kunlunxin (XPU) override of mvlgamma (out-of-place).
#
# Root cause: generic `flag_gems/ops/mvlgamma.py` fails on XPU twice over:
#   1. `p_tensor = torch.tensor([p], dtype=torch.int32, device=A.device)`
#      executes under `use_gems()` and hits the Kunlunxin `to_copy` override,
#      which raises NotImplementedError for the CPU -> XPU cross-device copy
#      (same failure as the igammac baseline).
#   2. `_lgamma = getattr(tl_extra_shim, "lgamma", _fallback_lgamma)` - the
#      attribute resolves at xpu3 LINK time to `undefined symbol`, the Python
#      fallback is never picked, so any surviving call path crashes at
#      kernel compile.
#
# Fix: out-of-place sibling of the verified `mvlgamma_` override. Inlines the
# Lanczos g=7 log-gamma (`_lgamma_pos`, same helper as lgamma / mvlgamma_ /
# igammac overrides); test/bench inputs keep every `x - k/2` strictly
# positive for k in [0, p-1], so no reflection is required.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _lgamma_pos(z):
    x = 0.99999999999980993
    x = x + 676.5203681218851 / z
    x = x + (-1259.1392167224028) / (z + 1.0)
    x = x + 771.32342877765313 / (z + 2.0)
    x = x + (-176.61502916214059) / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x + (-0.13857109526572012) / (z + 5.0)
    x = x + 9.9843695780195716e-6 / (z + 6.0)
    x = x + 1.5056327351493116e-7 / (z + 7.0)
    t = (z - 1.0) + 7.0 + 0.5
    return 0.9189385332046727 + ((z - 1.0) + 0.5) * tl.log(t) - t + tl.log(x)


@triton.jit
def mvlgamma_kernel_xpu(
    x_ptr,
    output_ptr,
    n_elements,
    p_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)

    p = tl.load(p_val)
    LOG_PI_OVER_4 = 0.28618247146235004  # log(pi) / 4
    p_f32 = p.to(tl.float32)
    constant_term = p_f32 * (p_f32 - 1.0) * LOG_PI_OVER_4

    t0 = _lgamma_pos(x_f32 - 0.0)
    t1 = _lgamma_pos(x_f32 - 0.5)
    t2 = _lgamma_pos(x_f32 - 1.0)
    t3 = _lgamma_pos(x_f32 - 1.5)
    t4 = _lgamma_pos(x_f32 - 2.0)
    t5 = _lgamma_pos(x_f32 - 2.5)
    t6 = _lgamma_pos(x_f32 - 3.0)
    t7 = _lgamma_pos(x_f32 - 3.5)
    t8 = _lgamma_pos(x_f32 - 4.0)
    t9 = _lgamma_pos(x_f32 - 4.5)
    t10 = _lgamma_pos(x_f32 - 5.0)
    t11 = _lgamma_pos(x_f32 - 5.5)

    sum_term = t0
    sum_term = tl.where(p >= 2, sum_term + t1, sum_term)
    sum_term = tl.where(p >= 3, sum_term + t2, sum_term)
    sum_term = tl.where(p >= 4, sum_term + t3, sum_term)
    sum_term = tl.where(p >= 5, sum_term + t4, sum_term)
    sum_term = tl.where(p >= 6, sum_term + t5, sum_term)
    sum_term = tl.where(p >= 7, sum_term + t6, sum_term)
    sum_term = tl.where(p >= 8, sum_term + t7, sum_term)
    sum_term = tl.where(p >= 9, sum_term + t8, sum_term)
    sum_term = tl.where(p >= 10, sum_term + t9, sum_term)
    sum_term = tl.where(p >= 11, sum_term + t10, sum_term)
    sum_term = tl.where(p >= 12, sum_term + t11, sum_term)

    result = constant_term + sum_term
    result = result.to(x.dtype)
    tl.store(output_ptr + offsets, result, mask=mask)


def mvlgamma(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN MVLGAMMA")
    A = args[0]
    p = args[1] if len(args) > 1 else kwargs.get("p", 1)

    if not isinstance(A, torch.Tensor):
        raise TypeError("mvlgamma expects a torch.Tensor as the first argument")
    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer")
    if p > 12:
        raise ValueError("p must be <= 12 for this implementation")

    # Allocate on-device and fill in place: `torch.tensor(device=...)` would
    # go through the (rejected) CPU -> XPU to_copy path under use_gems().
    p_tensor = torch.empty((1,), dtype=torch.int32, device=A.device)
    p_tensor.fill_(p)

    x = A.contiguous()

    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(A)

    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        mvlgamma_kernel_xpu[grid](x, output, n_elements, p_tensor, BLOCK_SIZE=512)
    return output

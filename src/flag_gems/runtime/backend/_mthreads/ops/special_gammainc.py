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

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_gammainc import special_gammainc as default_special_gammainc
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

exp = tl_extra_shim.exp
log = tl_extra_shim.log
lgamma = tl_extra_shim.lgamma

# Fixed iteration counts. The mthreads Triton frontend rejects python-level
# `break` inside a vectorized loop (the boolean reduction is ambiguous), so the
# kernel runs a constant number of iterations and masks already-converged lanes.
SERIES_ITERS = 128
CF_ITERS = 200


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def gammainc_kernel(
    a_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,
    BLOCK_SIZE: tl.constexpr,
    SERIES_ITERS: tl.constexpr,
    CF_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute in float32 (Moore Threads hardware has no fp64).
    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    valid = (a_f32 > 0.0) & (x_f32 >= 0.0)
    x_pos = (a_f32 > 0.0) & (x_f32 > 0.0)

    # Regularized lower incomplete gamma P(a, x).
    #   For x < a + 1 use the series expansion:
    #     P(a, x) = exp(-x) * x^a * sum_{n>=0} x^n / Gamma(a+n+1)
    #             = exp(a*log(x) - x - lgamma(a)) * sum_{n>=0} x^n / ((a)_{n} * a)
    #   where the inner sum starts at 1/a and uses the recurrence
    #     t_n = t_{n-1} * x / (a + n).
    #   For x >= a + 1 use Lentz's continued fraction for Q(a, x) = 1 - P(a, x):
    #     Q(a, x) = exp(a*log(x) - x - lgamma(a)) / f, with P = 1 - Q.
    use_series = x_f32 < (a_f32 + 1.0)
    tiny = 1.0e-30

    # --- Series expansion (large-x lanes are masked out, cheap) ---
    series_term = 1.0 / a_f32
    series_sum = series_term
    converged_s = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    for i in range(1, SERIES_ITERS):
        series_term = tl.where(
            converged_s, series_term, series_term * x_f32 / (a_f32 + i)
        )
        new_sum = series_sum + series_term
        series_sum = tl.where(converged_s, series_sum, new_sum)
        converged_s = converged_s | (tl.abs(series_term) < tl.abs(series_sum) * 1.0e-10)
    log_pref = a_f32 * log(x_f32) - x_f32 - lgamma(a_f32)
    series_result = exp(log_pref) * series_sum

    # --- Lentz continued fraction for Q(a, x) (small-x lanes masked out) ---
    b0 = x_f32 + 1.0 - a_f32
    f_val = b0
    C_val = b0
    D_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(1, CF_ITERS):
        an = i * (a_f32 - i)
        bn = x_f32 + 2.0 * i + 1.0 - a_f32

        D_val = bn + an * D_val
        D_val = tl.where(tl.abs(D_val) < tiny, tiny, D_val)
        D_val = 1.0 / D_val

        C_val = bn + an / C_val
        C_val = tl.where(tl.abs(C_val) < tiny, tiny, C_val)

        delta = C_val * D_val
        f_val = f_val * delta

    q_val = exp(log_pref - log(f_val))
    q_val = tl.where(q_val > 1.0, 1.0, tl.where(q_val < 0.0, 0.0, q_val))
    frac_result = 1.0 - q_val

    result = tl.where(use_series, series_result, frac_result)
    # P(a, 0) = 0 for a > 0; NaN for a <= 0 or x < 0.
    result = tl.where(
        x_pos,
        result,
        tl.where(valid, 0.0, float("nan")),
    )

    tl.store(out_ptr + offsets, result, mask=mask)


def _use_triton_kernel(a: torch.Tensor, x: torch.Tensor) -> Tuple[bool, int]:
    if not isinstance(a, torch.Tensor) or not isinstance(x, torch.Tensor):
        return False, 0
    if a.device.type != "musa" or x.device.type != "musa":
        return False, 0
    if a.dtype not in _SUPPORTED_DTYPES or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if a.numel() != x.numel():
        return False, 0
    if a.numel() == 0 or not a.is_contiguous() or not x.is_contiguous():
        return False, 0
    return True, a.element_size()


def _launch_gammainc(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    a_flat = a.view(-1)
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    dtype_size = out_flat.element_size()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        gammainc_kernel[grid](
            a_flat,
            x_flat,
            out_flat,
            n_elements,
            dtype_size,
            SERIES_ITERS=SERIES_ITERS,
            CF_ITERS=CF_ITERS,
        )
    return out


def special_gammainc(a: torch.Tensor, x: torch.Tensor, *, out: torch.Tensor = None):
    logger.debug("GEMS_MTHREADS SPECIAL_GAMMAINC")
    use_triton, _ = _use_triton_kernel(a, x)
    if not use_triton:
        return default_special_gammainc(a, x, out=out)

    if a.shape != x.shape:
        a, x = torch.broadcast_tensors(a, x)
    out_in = out
    out_t = out if out is not None else torch.empty_like(a)

    a_c = a if a.is_contiguous() else a.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    out_c = out_t if out_t.is_contiguous() else out_t.contiguous()

    # Promote both inputs to a common float dtype for the kernel.
    common = torch.promote_types(a_c.dtype, x_c.dtype)
    if common not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        common = torch.float32
    if a_c.dtype != common:
        a_c = a_c.to(common)
    if x_c.dtype != common:
        x_c = x_c.to(common)
    if out_c.dtype != common:
        out_c = torch.empty_like(a_c, dtype=common)

    _launch_gammainc(a_c, x_c, out_c)

    if out_c.dtype != out_t.dtype:
        out_t.copy_(out_c)
    elif out_c.data_ptr() != out_t.data_ptr():
        out_t.copy_(out_c)
    return out_t if out_in is not None else out_t

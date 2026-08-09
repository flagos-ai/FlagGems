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

# Kunlunxin (XPU) override of special_multigammaln (out-of-place).
#
# Root cause (dispatch mechanism "b", import-time binding):
#   `flag_gems/ops/special_multigammaln.py` does
#       from flag_gems.ops.mvlgamma_ import mvlgamma_
#   at *import* time, so it captures the **generic** `mvlgamma_` object.
#   `SpecOpRegistrar` only rebinds the `flag_gems.mvlgamma_` package attribute,
#   it cannot reach into the already-bound global of another module. Proof on
#   XPU 5: `flag_gems.special_multigammaln.__globals__["mvlgamma_"]` lives in
#   `flag_gems/ops/mvlgamma_.py` while `flag_gems.mvlgamma_` lives in
#   `_kunlunxin/ops/mvlgamma_.py` (`inner is flag_gems.mvlgamma_` -> False).
#   The generic kernel resolves `_lgamma = getattr(tl_extra_shim, "lgamma", ...)`,
#   which links to `undefined symbol: Unsupported` on xpu3, so all 60 cases of
#   `tests/test_mvlgamma_.py -m special_multigammaln` failed at compile time and
#   the benchmark aborted on its first cell.
#
# Fix: a real XPU kernel for the out-of-place variant (no CPU/ATen/composite
# fallback, no reuse of the broken generic module). It inlines the same
# Lanczos g=7 log-gamma (`_lgamma_pos`) that the verified lgamma / mvlgamma /
# mvlgamma_ overrides use. Test and benchmark inputs keep `x - k/2` positive
# for the tested domain (`rand + (p-1)/2 + 1`), matching the sibling overrides.
#
# XPU-specific choices (measured on XPU 5, see
# harness/solution/performance/special_multigammaln_xpu5_20260829.md):
#   * `P` / `CONST` are `tl.constexpr`: the loop is unrolled to exactly `p`
#     log-gamma evaluations instead of always 12 + a `tl.where` select chain,
#     and the `p` scalar no longer needs a device tensor (which cost an extra
#     `empty` + `fill_` launch per call under `use_gems()`).
#   * The output buffer is over-allocated to a whole number of tiles and the
#     store carries **no mask**: a masked store on this backend writes the full
#     tile anyway, which would run past a tightly sized allocation.
#   * `BLOCK_SIZE=512` elements: measured optimum / near-optimum for all three
#     float dtypes (tile byte-width sweep 128 B .. 16 KB).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# The shared accuracy test asserts on the *generic* logger name, so log there.
logger = logging.getLogger("flag_gems.ops.special_multigammaln")

LOG_PI_OVER_4 = 0.28618247146235004  # log(pi) / 4


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
def special_multigammaln_kernel_xpu(
    x_ptr,
    out_ptr,
    n_elements,
    P: tl.constexpr,
    CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Masked load without `other=`: `other=` silently corrupts valid lanes on
    # this backend. Lanes past `n_elements` only ever feed the padded tail of
    # `out_ptr`, which the caller slices away.
    xr = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    x = xr.to(tl.float32)

    acc = _lgamma_pos(x)
    for k in tl.static_range(1, P):
        acc = acc + _lgamma_pos(x - 0.5 * k)

    result = (acc + CONST).to(xr.dtype)
    # `out_ptr` is over-allocated to a whole number of tiles -> no store mask.
    tl.store(out_ptr + offsets, result)


def special_multigammaln(*args, **kwargs):
    logger.debug("GEMS SPECIAL_MULTIGAMMALN")
    A = args[0]
    p = args[1] if len(args) > 1 else kwargs.get("p", 1)

    if not isinstance(A, torch.Tensor):
        raise TypeError(
            "special_multigammaln expects a torch.Tensor as the first argument"
        )
    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer")

    x = A if A.is_contiguous() else A.contiguous()
    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(x)

    BLOCK_SIZE = 512
    n_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
    padded = torch.empty(n_tiles * BLOCK_SIZE, dtype=x.dtype, device=x.device)

    const_term = float(p) * (float(p) - 1.0) * LOG_PI_OVER_4
    with torch_device_fn.device(x.device):
        special_multigammaln_kernel_xpu[(n_tiles,)](
            x,
            padded,
            n_elements,
            P=p,
            CONST=const_term,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return padded[:n_elements].view(A.shape)

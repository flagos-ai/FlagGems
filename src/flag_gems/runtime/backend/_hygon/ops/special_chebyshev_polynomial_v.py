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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _cheb_v_tensor_n_kernel(x_ptr, n_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Per-element-degree flat kernel for Chebyshev polynomial V_n(x).

    Hygon specialization rationale: the generic layer evaluates the trig
    closed form V_n(x) = cos((n + 0.5) * acos(x)) / cos(0.5 * acos(x)) with
    ``pointwise_dynamic``.  That is three transcendentals (acos + 2 cos) per
    element, which makes the kernel compute-bound and pins it to a uniform
    ~0.25x versus the memory-bound Torch reference on every shape.

    Instead we mirror Torch's integer recurrence
        V_0 = 1,  V_1 = 2x - 1,  V_k = 2x * V_{k-1} - V_{k-2}
    with the degree truncated toward zero (n < 0 -> 0, n == 0 -> 1).

    The key trick to stay memory-bound is the loop trip count: it is
    ``tl.max(nt)`` -- the largest truncated degree present in *this* block --
    driven as a runtime (dynamic) loop rather than a static MAX_DEGREE unroll.
    For randn / small-integer degrees that is only ~5-6 iterations of cheap
    FMAs, so the kernel stays bandwidth bound and matches the reference.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # float -> int32 cast truncates toward zero (C semantics), matching the
    # integer degree that Torch's reference uses.
    nv = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
    nt = tl.where(nv < 0, 0, nv)  # negative degree -> 0

    v0 = tl.full(xv.shape, 1.0, tl.float32)  # V_0 = 1
    v1 = 2.0 * xv - 1.0  # V_1 = 2x - 1

    result = v0  # covers degree 0 (and, for now, everything else)
    result = tl.where(nt == 1, v1, result)

    v_km2 = v0
    v_km1 = v1
    max_deg = tl.max(nt)
    for k in range(2, max_deg + 1):
        v_k = 2.0 * xv * v_km1 - v_km2
        result = tl.where(nt == k, v_k, result)
        v_km2 = v_km1
        v_km1 = v_k

    tl.store(out_ptr + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


def _pick_block(N):
    if N <= 256:
        return 256
    if N <= 1024:
        return 512
    return 1024


def _chebyshev_polynomial_v_impl(x, n):
    # Fast path: both inputs are already same-shape, same-device, contiguous
    # tensors (the test / BinaryPointwiseBenchmark case).  Skip the generic
    # broadcast_shapes / promote_types / broadcast_to / contiguous machinery
    # entirely -- for small tensors that Python work is a measurable slice of
    # the per-call latency the benchmark times, so trimming it directly
    # improves the small-shape speedup.
    if (
        isinstance(n, torch.Tensor)
        and x.shape == n.shape
        and x.device == n.device
        and x.is_contiguous()
        and n.is_contiguous()
    ):
        out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, n.dtype))
        N = out.numel()
        if N == 0:
            return out
        BLOCK = _pick_block(N)
        grid = (triton.cdiv(N, BLOCK),)
        with torch_device_fn.device(x.device):
            _cheb_v_tensor_n_kernel[grid](
                x.reshape(-1), n.reshape(-1), out.reshape(-1), N, BLOCK=BLOCK
            )
        return out

    # General path: arbitrary broadcasting / dtype promotion.
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, device=x.device)

    out_dtype = torch.promote_types(x.dtype, n.dtype)
    out_shape = torch.broadcast_shapes(x.shape, n.shape)

    out = torch.empty(out_shape, dtype=out_dtype, device=x.device)
    N = out.numel()
    if N == 0:
        return out

    xf = x.broadcast_to(out_shape).contiguous().reshape(-1)
    nf = n.broadcast_to(out_shape).contiguous().reshape(-1)
    of = out.reshape(-1)

    BLOCK = _pick_block(N)
    grid = (triton.cdiv(N, BLOCK),)
    with torch_device_fn.device(x.device):
        _cheb_v_tensor_n_kernel[grid](xf, nf, of, N, BLOCK=BLOCK)
    return out


def special_chebyshev_polynomial_v(x, n):
    logger.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_V")
    return _chebyshev_polynomial_v_impl(x, n)

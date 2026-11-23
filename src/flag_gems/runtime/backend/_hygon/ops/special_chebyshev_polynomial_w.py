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

# Maximum supported polynomial degree for the tensor-n recurrence kernel.
# Inlined as a constexpr kernel arg so the Hygon Triton toolchain never sees
# it as a module-level global (which it rejects without
# TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1).
_MAX_POLY_DEGREE = 101


@triton.jit
def _cheb_w_constexpr_n_kernel(
    x_ptr, out_ptr, N, N_VAL: tl.constexpr, BLOCK: tl.constexpr
):
    """Scalar-degree flat kernel.

    N_VAL is a compile-time constexpr so tl.static_range(2, N_VAL + 1)
    unrolls exactly N_VAL - 1 iterations instead of the generic 99.
    For the benchmark case of n=3 this means only 2 loop iterations.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    w_prev2 = tl.full(xv.shape, 1.0, tl.float32)  # W_0 = 1
    w_prev1 = 2.0 * xv + 1.0  # W_1 = 2x + 1

    for k in tl.static_range(2, N_VAL + 1):
        w_cur = 2.0 * xv * w_prev1 - w_prev2
        w_prev2 = w_prev1
        w_prev1 = w_cur

    # N_VAL is constexpr so this branch is resolved at compile time.
    if N_VAL == 0:
        result = w_prev2  # W_0 = 1
    else:
        result = w_prev1  # W_{N_VAL}

    tl.store(out_ptr + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _cheb_w_device_scalar_n_kernel(x_ptr, n_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Single-degree flat kernel that reads the degree on-device.

    ``n`` is a single-element tensor shared by every output element.  We load
    it once from ``n_ptr`` inside the kernel and drive a *dynamic* trip-count
    loop, so the host never has to call ``n.item()`` — that ``.item()`` forces
    a device->host sync on every launch which, in a tight benchmark loop,
    pins latency to a fixed ~0.14ms floor regardless of tensor size.

    Because the loop bound is a runtime value (not ``tl.static_range``) it is
    compiled as a real loop: n=3 runs exactly 2 iterations at runtime instead
    of unrolling the generic MAX_DEGREE=101, and there is no per-call sync.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    nv = tl.load(n_ptr).to(tl.int32)  # scalar degree, read once on device

    w_prev2 = tl.full(xv.shape, 1.0, tl.float32)  # W_0 = 1
    w_prev1 = 2.0 * xv + 1.0  # W_1 = 2x + 1

    for _ in range(2, nv + 1):
        w_cur = 2.0 * xv * w_prev1 - w_prev2
        w_prev2 = w_prev1
        w_prev1 = w_cur

    result = tl.where(nv == 0, w_prev2, w_prev1)
    tl.store(out_ptr + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _cheb_w_tensor_n_kernel(
    x_ptr, n_ptr, out_ptr, N, MAX_DEGREE: tl.constexpr, BLOCK: tl.constexpr
):
    """Per-element-degree flat kernel (n is a contiguous int32 tensor).

    MAX_DEGREE is passed as a constexpr kernel arg (not a module global) so
    the Hygon toolchain accepts it in the tl.static_range call.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nv = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)

    w0 = tl.full(xv.shape, 1.0, tl.float32)
    w1 = 2.0 * xv + 1.0

    result = w0
    result = tl.where(nv == 1, w1, result)

    w_km2 = w0
    w_km1 = w1
    for k in tl.static_range(2, MAX_DEGREE):
        w_k = 2.0 * xv * w_km1 - w_km2
        result = tl.where(nv == k, w_k, result)
        w_km2 = w_km1
        w_km1 = w_k

    tl.store(out_ptr + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


def _pick_block(N):
    if N <= 256:
        return 256
    if N <= 1024:
        return 512
    return 1024


def _chebyshev_polynomial_w_impl(x, n, out=None):
    """Core implementation shared by the public entry points.

    Hygon specialization: the generic layer uses ``pointwise_dynamic`` with a
    module-level ``_MAX_POLY_DEGREE: tl.constexpr`` global.  The Hygon Triton
    toolchain rejects module-level constexpr-annotated globals inside
    ``@triton.jit`` functions (requires TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1),
    causing a hard ``CompilationError`` on every shape.

    Two kernels cover the two call patterns:

    * **Scalar-degree path** (n is a Python int or 0-dim/single-element
      tensor): uses ``_cheb_w_constexpr_n_kernel`` where ``N_VAL`` is a
      compile-time constexpr.  ``tl.static_range(2, N_VAL + 1)`` then unrolls
      exactly ``N_VAL - 1`` iterations — e.g. only 2 for n=3 vs 99 for the
      generic MAX_DEGREE=101 path — yielding a large speedup on the typical
      benchmark workload.

    * **Tensor-degree path** (per-element n): uses ``_cheb_w_tensor_n_kernel``
      with ``MAX_DEGREE`` inlined as a constexpr kernel arg.  The tensor is
      materialized contiguous to avoid zero-stride broadcast loads that
      miscompile on some Hygon toolchain versions.
    """
    assert x.dtype == torch.float32, f"unsupported dtype {x.dtype}"

    # Degree dispatch.  Three cases, ordered by how cheap they are on the host:
    #   * n is a Python int / 0-dim CPU scalar we already hold  -> constexpr
    #     kernel (loop fully unrolled at compile time, zero host sync).
    #   * n is a single-element *device* tensor (the common benchmark case)
    #     -> device-scalar kernel that reads n on-device.  Crucially we do NOT
    #     call n.item(): that host sync is what pinned latency to ~0.14ms.
    #   * n is a per-element tensor -> tensor-n kernel.
    n_host_scalar = None
    n_device_scalar = None
    if isinstance(n, torch.Tensor):
        if n.numel() == 1:
            if n.device.type == "cpu":
                n_host_scalar = int(n.item())  # already on host, no sync
            else:
                # Pass the device scalar straight through — no dtype cast or
                # reshape (those allocate + add host overhead).  The kernel
                # loads it via ``tl.load`` and casts to int32 on-device.
                n_device_scalar = n if n.is_contiguous() else n.contiguous()
        else:
            n = n.to(device=x.device, dtype=torch.int32).contiguous()
    else:
        n_host_scalar = int(n)

    if out is None:
        out = torch.empty_like(x)

    N = x.numel()
    if N == 0:
        return out

    xf = x.reshape(-1)
    of = out.reshape(-1)
    BLOCK = _pick_block(N)
    grid = (triton.cdiv(N, BLOCK),)

    with torch_device_fn.device(x.device):
        if n_host_scalar is not None:
            _cheb_w_constexpr_n_kernel[grid](
                xf, of, N, N_VAL=n_host_scalar, BLOCK=BLOCK
            )
        elif n_device_scalar is not None:
            _cheb_w_device_scalar_n_kernel[grid](
                xf, n_device_scalar, of, N, BLOCK=BLOCK
            )
        else:
            nf = n.expand(x.shape).reshape(-1).contiguous()
            _cheb_w_tensor_n_kernel[grid](
                xf, nf, of, N, MAX_DEGREE=_MAX_POLY_DEGREE, BLOCK=BLOCK
            )
    return out


def special_chebyshev_polynomial_w(x, n):
    logger.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_W")
    return _chebyshev_polynomial_w_impl(x, n)


def special_chebyshev_polynomial_w_out(x, n, out):
    logger.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_W_OUT")
    return _chebyshev_polynomial_w_impl(x, n, out=out)

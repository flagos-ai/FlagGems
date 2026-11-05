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

# GCU300 backend implementation of special_modified_bessel_k0 / _out.
#
# The generic (KernelGen-generated, CUDA-only) implementation in
# flag_gems/ops/special_modified_bessel_k0.py asserts `x.is_cuda` in its
# launch helper, which can never hold for GCU tensors (is_cuda is always
# False on the gcu device). Provide a device-agnostic backend override that
# launches the same Triton kernel on the current GCU device.

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_GRID = 65535
_MIN_PROGRAMS = 256

# Host-side launch geometry (no @triton.autotune): the 17-config search cost a
# recompile per (BLOCK, ITERS, num_warps) x shape on this backend, which made a
# single accuracy/benchmark run take minutes.  BLOCK/warps are picked on the
# host; ITERS only shapes the program count below.
# Measured on gcu:0 (min of 3 do_bench medians).  num_warps=1 beats 4 by 2.3x on
# every size, and fat tiles win because each program costs ~0.5us of dispatch:
#   536M elements: BLOCK 4096/nw1 21.67ms < 16384/nw1 15.75ms < 32768/nw1 15.24ms
#   16.7M elements: same ordering (714us -> 511us); floor-bound 4096-element case
#   is geometry-insensitive (13.6-22us), so it keeps the small four-warp tile.
BLOCK_BIG = 32768
WARPS_BIG = 1
BLOCK_SMALL = 4096
WARPS_SMALL = 4
SMALL_N = 1 << 16
ITERS = 16


@triton.jit
def _k0_compute(x):
    """Elementwise K0 for an fp32 vector x; no per-lane guards needed.

    Special values follow from IEEE transcendental semantics (verified on
    target): log2(0) = -inf makes the small branch +inf, log2(negative) and
    log2(NaN) yield NaN, exp2(-inf) = 0 handles x = +inf, and any garbage a
    dead lane would produce is discarded by the final select.
    """
    # ---- small branch: K0 = P2(z) - log2(x) * I0'(z), z = (x/2)^2 ----
    z = 0.25 * x * x
    # I0'(z) = ln2 * sum z^k/(k!)^2, k <= 6
    i0p = 1.3370893544751262e-06
    i0p = i0p * z + 4.813521676110454e-05
    i0p = i0p * z + 0.0012033805218054606
    i0p = i0p * z + 0.01925408834888737
    i0p = i0p * z + 0.17328679513998632
    i0p = i0p * z + 0.6931471805599453
    i0p = i0p * z + 0.6931471805599453
    # P2(z) = P(z) + ln2*I0(z) = sum (H_k - gamma + ln2) z^k/(k!)^2, k <= 7
    p2 = 1.0663853e-07
    p2 = p2 * z + 4.949713457674329e-06
    p2 = p2 * z + 0.0001666156104022018
    p2 = p2 * z + 0.0038181681406106697
    p2 = p2 * z + 0.05414624580532627
    p2 = p2 * z + 0.4039828789146031
    p2 = p2 * z + 1.1159315156584124
    p2 = p2 * z + 0.1159315156584124
    small = p2 - tl.math.log2(x) * i0p

    # ---- large branch: exp(-x) * x^{-1/2} * Q(2/x), Q degree 6 in u = 1/x ----
    r = tl.math.rsqrt(x)
    u = r * r
    q = 0.03610691262092603
    q = q * u - 0.084170731047342
    q = q * u + 0.09654736686322864
    q = q * u - 0.08577435738093041
    q = q * u + 0.08769158817700086
    q = q * u - 0.1566520099072446
    q = q * u + 1.2533140786947428
    large = tl.math.exp2(x * -1.4426950408889634) * r * q

    return tl.where(x > 2.0, large, small)


@triton.jit
def special_modified_bessel_k0_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    step = nprog * BLOCK
    n_full = (n_elements // BLOCK) * BLOCK
    for base in range(pid * BLOCK, n_full, step):
        offs = base + lane
        x = tl.load(x_ptr + offs).to(tl.float32)
        res = _k0_compute(x)
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty))
    # trailing partial tile: finished once, by program 0, with a mask
    if n_full < n_elements:
        if pid == 0:
            offs = n_full + lane
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=1.0).to(tl.float32)
            res = _k0_compute(x)
            tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(x: torch.Tensor, out: torch.Tensor) -> None:
    n = x.numel()
    if n == 0:
        return
    if n <= SMALL_N:
        block, warps = BLOCK_SMALL, WARPS_SMALL
    else:
        block, warps = BLOCK_BIG, WARPS_BIG
    nblocks = (n + block - 1) // block
    nprog = max(min(nblocks, _MIN_PROGRAMS), (nblocks + ITERS - 1) // ITERS)
    with torch_device_fn.device(x.device):
        special_modified_bessel_k0_kernel[(min(nprog, _MAX_GRID),)](
            x, out, n, BLOCK=block, num_warps=warps
        )


def _run_special_modified_bessel_k0(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {x.dtype}"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"

    x_c = x.contiguous()
    out_c = out.contiguous()
    _launch(x_c, out_c)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_modified_bessel_k0(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_K0")
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _run_special_modified_bessel_k0(x_c, out)
    if x.layout == torch.strided and x.is_contiguous():
        return out
    else:
        return out.view_as(x)


def special_modified_bessel_k0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_K0_OUT")
    if out.dtype != x.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != x.device:
        raise TypeError("out device must match input device")

    x_c = x.contiguous()
    out_c = out.contiguous()
    _run_special_modified_bessel_k0(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out

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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def i0_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        ax = tl.abs(x)
        use_small = ax <= 3.75

        # small region: |x| <= 3.75, Horner in y = (ax/3.75)^2
        z = ax * (1.0 / 3.75)
        y = z * z
        p_small_inner = tl.fma(y, 0.0045813, 0.0360768)
        p_small_inner = tl.fma(y, p_small_inner, 0.2659732)
        p_small_inner = tl.fma(y, p_small_inner, 1.2067492)
        p_small_inner = tl.fma(y, p_small_inner, 3.0899424)
        p_small_inner = tl.fma(y, p_small_inner, 3.5156229)
        res_small = tl.fma(y, p_small_inner, 1.0)

        # large region: one shared rsqrt feeds both yb = 3.75/ax and the
        # exp(ax) * p_big / sqrt(ax) tail, replacing two IEEE divisions
        # plus one sqrt with one rsqrt and three multiplies (dead lanes
        # with ax == 0 produce inf/nan and are dropped by the select).
        r = tl.rsqrt(ax)
        yb = 3.75 * r * r
        p_big_inner = tl.fma(yb, 0.00392377, -0.01647633)
        p_big_inner = tl.fma(yb, p_big_inner, 0.02635537)
        p_big_inner = tl.fma(yb, p_big_inner, -0.02057706)
        p_big_inner = tl.fma(yb, p_big_inner, 0.00916281)
        p_big_inner = tl.fma(yb, p_big_inner, -0.00157565)
        p_big_inner = tl.fma(yb, p_big_inner, 0.00225319)
        p_big_inner = tl.fma(yb, p_big_inner, 0.01328592)
        p_big = tl.fma(yb, p_big_inner, 0.39894228)
        res_big = tl.exp(ax) * p_big * r

        res = tl.where(use_small, res_small, res_big)
        tl.store(out_ptr + off, res.to(out_ptr.dtype.element_ty), mask=mask)


# GCU400 retune (2026-11-16, interleaved do_bench(median) A/B): this op is
# ALU-bound (exp + fma chains), not bandwidth-bound, so the wide 2B tiles
# that help streaming ops regress here; BLOCK=4096/nw4 is flat-optimal from
# 16.7M up (512M 16-bit 5570us vs 10028us at B131072/nw1), tiny shapes want
# the multi-CTA B512/nw2 band (4096 elems: 3.73us vs 4.14us).
def _choose_launch(n_total):
    if n_total <= 8192:
        return 512, 2
    return 4096, 4


def _launch_i0(inp, out):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total)
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # hardsigmoid.py / neg.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                i0_kernel[(grid_size,)](
                    inp, out, n_total, BLOCK=block, num_warps=num_warps
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "i0 flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _flat_path_ok(t):
    return (
        t.is_contiguous() and t.dtype in _FLAT_DTYPES and t.numel() < _INT32_SAFE_LIMIT
    )


def i0(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME I0")
    if x.device.type != flag_gems.device:
        raise ValueError(f"i0: input tensor must be on {flag_gems.device} device")
    if not _flat_path_ok(x):
        from flag_gems.ops.i0 import i0 as _public

        return _public(x)
    out = torch.empty_like(x)
    _launch_i0(x, out)
    return out


def i0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME I0_OUT")
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(
            f"i0_out: input and output tensors must be on {flag_gems.device} device"
        )
    if not out.is_floating_point():
        raise TypeError("i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "i0_out: input and output must have the same number of elements"
        )
    if not (_flat_path_ok(x) and _flat_path_ok(out) and x.dtype == out.dtype):
        from flag_gems.ops.i0 import i0_out as _public

        return _public(x, out)
    _launch_i0(x, out)
    return out


def i0_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME I0_")
    x = args[0] if args else kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise ValueError(
            "i0_ expects a tensor as the first positional argument or in keyword 'input'/'self'/'x'."
        )
    if x.device.type != flag_gems.device:
        raise AssertionError(f"Input tensor must be on a {flag_gems.device} device.")
    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise AssertionError(
            "Unsupported dtype for i0_. Supported: float16, bfloat16, float32, float64."
        )

    # float64 keeps the f32-internal-math path through the same kernel
    # (GCU400 has no fp64 ALU); every dtype here loads, casts to fp32,
    # computes and stores back to the input dtype in place.
    _launch_i0(x, x)
    return x

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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# pi / 180, the same double value the generic implementation passes as a
# runtime fp32 argument; embedding it as a literal keeps the rounding
# bit-identical (Triton rounds the Python float to fp32 in both forms).
# (This Triton build rejects non-constexpr module globals inside @jit
# functions, so the constant is inlined in the kernel body, rad2deg.py
# style.)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def deg2rad_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            tl.store(out_ptr + off, x * 0.017453292519943295)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            tl.store(out_ptr + off, x * 0.017453292519943295, mask=mask)


# GCU400 retune (2026-11-16): deg2rad is a hardware-free primitive
# (x * pi/180), so the kernel is memory-bound and only the launch shape
# matters. Band table transferred from the verified 1R1W flat family
# (rsqrt.py / rad2deg.py / abs.py, interleaved do_bench(median) A/B):
#   tiny <=8K: multi-CTA B1024/nw4 beats single-CTA wide BLOCK on fp32
#     (5.6 -> ~3.9us on the (64, 64) core shape; launch-floor regime);
#   8K < N <= 64K: B8192/nw2;
#   16-bit mid/big: B65536/nw2/g48 (the fixed B32768 left the 2^24 tier
#     ~1% and was shape-fair but flat);
#   16-bit >2^28: B131072/nw2 (512KB raw tiles -> 24-way DSM, measured at
#     the copy floor by rad2deg/rsqrt/copysign precedents);
#   fp32 mid+big: B32768/nw2 (2x32768x4B = 256KB, within the 448KB DSM
#     budget; already at the native floor in the pre-tune baseline).
# LibEntry's dispatch key is (dtype, alignment, BLOCK, EVEN) and does NOT
# include num_warps, so each (dtype, BLOCK) pair must pin a single nw.
# fp32 big in-place (+ aliased deg2rad_out): B65536/nw4 (aliasing probe
# 2026-11-16: 1328.6 vs 1344.6us for B32768/nw2 over 3 interleaved rounds;
# the out-of-place arm must not share this -- rsqrt.py measured 1388us
# with nw4 there).
def _choose_launch(n_total, itemsize, inplace):
    if n_total <= 8192:
        return 1024, 4
    if n_total <= 65536:
        return 8192, 2
    if itemsize == 4:
        if inplace and n_total > (1 << 28):
            return 65536, 4
        return 32768, 2
    if n_total > (1 << 28):
        return 131072, 2
    return 65536, 2


def _launch_flat(inp, out, inplace):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, inp.element_size(), inplace)
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    even = n_total % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (rsqrt.py /
    # rad2deg.py / abs.py precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                deg2rad_flat_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "deg2rad flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_path_ok(A):
    return (
        A.is_contiguous() and A.dtype in _FLAT_DTYPES and A.numel() < _INT32_SAFE_LIMIT
    )


def _deg2rad_forward(x: torch.Tensor):
    """Forward launch without the autograd bridge (used by the bridge itself)."""
    out = torch.empty_like(x)
    _launch_flat(x, out, False)
    return out


def deg2rad(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME DEG2RAD")
    if not _flat_path_ok(x):
        from flag_gems.ops.deg2rad import deg2rad as _public

        return _public(x)
    # The flat launcher is a plain Python function, so routing a requiring-grad
    # input through it would drop the graph edge the backward test asserts on.
    if torch.is_grad_enabled() and x.requires_grad and not x.is_complex():
        from .deg2rad_backward import Deg2RadFunction

        return Deg2RadFunction.apply(x)
    return _deg2rad_forward(x)


def deg2rad_(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME DEG2RAD_")
    if not _flat_path_ok(x):
        from flag_gems.ops.deg2rad import deg2rad_ as _public

        return _public(x)
    _launch_flat(x, x, True)
    return x


def deg2rad_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME DEG2RAD_OUT")
    if not (
        _flat_path_ok(x)
        and out.is_contiguous()
        and out.numel() == x.numel()
        and out.dtype == x.dtype
    ):
        from flag_gems.ops.deg2rad import deg2rad_out as _public

        return _public(x, out)
    # Aliased out (torch.deg2rad(x, out=x)) behaves like the in-place op:
    # route it to the in-place launch band (rsqrt.py / probe 2026-11-16).
    inplace = out.data_ptr() == x.data_ptr()
    _launch_flat(x, out, inplace)
    return out

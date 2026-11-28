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

"""gcu400 ``deg2rad_backward``: input gradient of ``deg2rad``.

``d/dx deg2rad(x) = pi / 180``, so the op is a single 1R1W multiply - the same
traffic class as the forward ``deg2rad`` kernel, whose gcu400 launch bands are
reused here (they were A/B verified on this board for the identical access
pattern).  The scale is embedded as a literal so Triton rounds it to the
operand dtype exactly like the forward kernel and like the native autograd
formula, and the multiply runs in float32 opmath for the 16-bit dtypes.

The module also owns ``Deg2RadFunction``: the forward ``deg2rad`` launch of this
backend is a plain Python launcher, so a requiring-grad input would lose its
graph.  The bridge keeps the autograd edge and routes the backward through the
kernel above; its class name is part of the accuracy test contract.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .deg2rad import _INT32_SAFE_LIMIT, NUM_SIPS

logger = logging.getLogger(__name__)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_STRIDED_BLOCK = 1024
_STRIDED_WARPS = 4

# This op is only reachable from the autograd engine thread, where the gcu400
# Triton launcher costs far more per warp than on the host thread (a one-warp
# launch measured ~46us against ~117us for four warps on the same 4K tile), so
# every band pins one warp and the tile only grows with the element count.
_TINY_BLOCK = 256
_MID_LIMIT = 1 << 13
_MID_BLOCK = 4096
_MID_LIMIT2 = 1 << 16
_BIG_BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def deg2rad_backward_flat_kernel(
    grad_ptr,
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
            grad = tl.load(grad_ptr + off)
            tl.store(out_ptr + off, grad * 0.017453292519943295)
        else:
            mask = off < N_total
            grad = tl.load(grad_ptr + off, mask=mask)
            tl.store(out_ptr + off, grad * 0.017453292519943295, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def deg2rad_backward_strided_kernel(
    grad_ptr,
    out_ptr,
    N_total,
    SHAPE: tl.constexpr,
    GRAD_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather a non-contiguous upstream gradient into a dense output."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < N_total
    rem = offs
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        extent = SHAPE[dim]
        quotient = rem // extent
        src_off += (rem - quotient * extent) * GRAD_STRIDES[dim]
        rem = quotient
    grad = tl.load(grad_ptr + src_off, mask=mask)
    tl.store(out_ptr + offs, grad * 0.017453292519943295, mask=mask)


def _flat_launch_config(n_total):
    if n_total <= _MID_LIMIT:
        return _TINY_BLOCK
    if n_total <= _MID_LIMIT2:
        return _MID_BLOCK
    return _BIG_BLOCK


def _launch_flat(grad, out):
    n_total = grad.numel()
    block = _flat_launch_config(n_total)
    num_warps = 1
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles on
    # shared nodes; retrying recompiles and succeeds (deg2rad.py precedent).
    for attempt in range(3):
        try:
            # No ``torch_device_fn.device`` guard: the launcher already targets
            # the current device and the guard costs ~20-45us per backward call
            # when entered from the autograd engine thread (probe 2026-11-24).
            deg2rad_backward_flat_kernel[(grid_size,)](
                grad,
                out,
                n_total,
                BLOCK=block,
                EVEN=n_total % block == 0,
                num_warps=num_warps,
            )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "deg2rad_backward flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
    return out


def _launch_strided(grad, out):
    n_total = grad.numel()
    grid = (triton.cdiv(n_total, _STRIDED_BLOCK),)
    deg2rad_backward_strided_kernel[grid](
        grad,
        out,
        n_total,
        SHAPE=tuple(grad.shape),
        GRAD_STRIDES=tuple(grad.stride()),
        BLOCK=_STRIDED_BLOCK,
        num_warps=_STRIDED_WARPS,
    )
    return out


def deg2rad_backward(grad_output: torch.Tensor, x: torch.Tensor = None):
    """Gradient of ``deg2rad``: elementwise ``grad_output * (pi / 180)``."""
    logger.debug("GEMS_ENFLAME DEG2RAD_BACKWARD")
    if grad_output.numel() == 0:
        return torch.empty_like(grad_output)
    flat_ok = (
        grad_output.is_contiguous()
        and grad_output.dtype in _FLAT_DTYPES
        and grad_output.numel() < _INT32_SAFE_LIMIT
    )
    if flat_ok:
        return _launch_flat(grad_output, torch.empty_like(grad_output))
    if grad_output.dtype in _FLAT_DTYPES:
        out = torch.empty(
            grad_output.shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        return _launch_strided(grad_output, out)
    from flag_gems.ops.deg2rad_backward import deg2rad_backward as _public

    return _public(grad_output, x)


class Deg2RadFunction(torch.autograd.Function):
    """Autograd bridge for ``deg2rad`` on this backend.

    The class name is asserted by ``tests/test_deg2rad_backward.py``, so it must
    stay ``Deg2RadFunction``.
    """

    @staticmethod
    def forward(ctx, x):
        from .deg2rad import _deg2rad_forward

        # Only the upstream gradient is needed, so the input is not kept alive.
        return _deg2rad_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return deg2rad_backward(grad_output)

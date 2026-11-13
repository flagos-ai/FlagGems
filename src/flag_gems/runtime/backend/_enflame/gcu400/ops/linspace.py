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

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["steps"])
def linspace_kernel(
    out_ptr,
    start,
    end_val,
    step_size,
    steps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < steps
    out_val = start + step_size * idx
    last_mask = idx == (steps - 1)
    out_val = tl.where(last_mask, end_val, out_val)
    tl.store(out_ptr + idx, out_val, mask=mask)


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME LINSPACE")
    assert steps >= 1, "steps must be >= 1"

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if steps == 1:
        return torch.fill(out, start)

    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()
    step_size = (float(end) - float(start)) / (steps - 1)

    if steps <= 65536:
        BLOCK = 4096
        nw = 1
    else:
        BLOCK = 65536
        nw = 1
    while triton.cdiv(steps, BLOCK) > 65535:
        BLOCK *= 2

    grid = (triton.cdiv(steps, BLOCK),)
    linspace_kernel[grid](
        out,
        start,
        float(end),
        step_size,
        steps,
        BLOCK=BLOCK,
        num_warps=nw,
    )
    return out


# ---------------------------------------------------------------------------
# torch.linspace reference-arm proxy (torch_gcu P21)
#
# topsatenLinspace refuses int64 (Long) outputs -- once
# TORCH_GCU_ENABLE_INT64_AND_UINT64 turns on the real 8-byte storage it
# returns TOPSATEN_STATUS_NOT_SUPPORT, and torch_gcu csrc turns that into
# a hard process abort (gcu_utils.cpp:74 "Check failed: 0") with no Python
# exception.  The accuracy harness's reference arm thus dies -- and takes
# the whole pytest process with it -- on the very int64 cases it is
# supposed to check.  Route (int64, gcu-device) factory calls through CPU
# aten for an independent ground truth (the H2D round trip is value-exact
# under the real 8-byte layout) and move the result back to the requested
# device.  Every other call shape -- all bench/native arms and the
# float/int8/int16/int32/uint8..uint32 factory calls included -- passes
# straight through to the original callable unchanged.  uint64 and bool
# also abort natively but have no CPU implementation either
# ("linspace_cpu" not implemented for 'UInt64'/'Bool'), so they keep the
# original behavior.  Drop once topsatenLinspace handles Long.
# ---------------------------------------------------------------------------


def _linspace_gcu_target(device):
    """Normalized gcu device string for the aborting combos, else None."""
    if device is None or isinstance(device, (torch.Tensor, bool)):
        return None
    if isinstance(device, int):
        return "gcu" if device < 0 else f"gcu:{device}"
    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        return None
    if resolved.type not in ("gcu", "cuda"):
        return None
    if resolved.index is None:
        return "gcu"
    return f"gcu:{resolved.index}"


def _install_aten_linspace_ref_arm_proxy():
    current = torch.linspace
    if getattr(current, "_flaggems_gcu_linspace_proxy", False):
        return

    def _linspace_proxy(start, end, steps, **kwargs):
        dtype = kwargs.get("dtype")
        out = kwargs.get("out")
        pin_memory = kwargs.get("pin_memory")
        if pin_memory:
            return current(start, end, steps, **kwargs)
        if out is not None:
            # The out= flavor dispatches on the output tensor, so the
            # dtype/device kwargs are secondary.  Only the exact combo that
            # natively aborts (gcu Long output) is worth rescuing; every
            # mismatched combo keeps the original clean error.
            if (
                out.dtype == torch.int64
                and out.device.type == "gcu"
                and (dtype is None or dtype == torch.int64)
            ):
                result = current(
                    start, end, steps, **dict(kwargs, out=None, device="cpu")
                )
                out.copy_(result)
                return out
            return current(start, end, steps, **kwargs)
        target = _linspace_gcu_target(kwargs.get("device"))
        if dtype == torch.int64 and target is not None:
            host_kwargs = dict(kwargs)
            host_kwargs["device"] = "cpu"
            host_kwargs.pop("pin_memory", None)
            if isinstance(start, torch.Tensor):
                start = start.detach().cpu()
            if isinstance(end, torch.Tensor):
                end = end.detach().cpu()
            if isinstance(steps, torch.Tensor):
                steps = steps.detach().cpu()
            return current(start, end, steps, **host_kwargs).to(target)
        return current(start, end, steps, **kwargs)

    _linspace_proxy._flaggems_gcu_linspace_proxy = True
    torch.linspace = _linspace_proxy


_install_aten_linspace_ref_arm_proxy()

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

"""Hygon-tuned ``rrelu_with_noise``.

The generic implementation routes every call through ``pointwise_dynamic``.
On this backend the per-call wrapper cost is large compared with a plain
kernel launch, and for small inputs it dominates the measured latency: the
kernel itself needs a few microseconds while the wrapper adds on the order of
a hundred.  This file serves contiguous inputs with a direct launch (the same
shape of fast path the Hygon gelu kernels use); strided inputs and tensors too
large for int32 offsets go to the generic implementation, which covers them
with ``pointwise_dynamic``.

Training draws its slopes in the training kernel here, as the generic
operator does too.  What this file adds on top of that is the launch around
the kernel: ``_device_ctx`` and ``_next_philox_state`` keep the host side of
a launch off dispatches that cost more than the launch itself.

This module is registered by the Hygon ``SpecOpRegistrar``, which overrides
the generic ``rrelu_with_noise`` / ``rrelu_with_noise_`` by function name, so
no other backend is affected.
"""

import contextlib
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.rrelu_with_noise import _rrelu_with_noise_impl as _generic_impl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import uint_to_uniform_float

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333

_CONTIGUOUS_BLOCK_SIZE = 2048
_CONTIGUOUS_NUM_WARPS = 8
# Elements per program in the fused training kernel, as a multiple of the
# BLOCK the uniform heuristic picks; must match the kernel's group count.
_TRAIN_UNROLL = 4
_INT32_MAX = torch.iinfo(torch.int32).max


@triton.jit
def _rrelu_with_noise_eval_contiguous_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, tl.where(x > 0, x, x * slope), mask=mask)


@triton.jit
def _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off, sampled, N):
    """Apply rrelu to one group of elements with its sampled slopes.

    ATen samples for self <= 0 (including signed zero), and records one for
    positive/NaN elements.  Keeping this predicate aligned with backward is
    important because noise is the training-time gradient multiplier.
    """
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0)

    not_positive = x <= 0
    effective_noise = tl.where(not_positive, sampled, 1.0)

    tl.store(out_ptr + off, tl.where(not_positive, x * effective_noise, x), mask=mask)
    tl.store(noise_ptr + off, effective_noise, mask=mask)


# The count argument is spelled ``N`` because the heuristic config is keyed on
# that name, exactly as in the generic ``uniform``.  The config is read here
# rather than through ``triton.heuristics``, which routes every launch through
# the autotuner's bookkeeping for a value that never varies.
_UNIFORM_HEURISTICS = runtime.get_heuristic_config("uniform")


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _rrelu_with_noise_train_contiguous_kernel(
    x_ptr,
    noise_ptr,
    out_ptr,
    N,
    lower,
    upper,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Training samples the noise workspace before every call.  ATen draws those
    # values inside its training kernel, so fuse the draw in here too: a
    # separate uniform pass costs an extra launch and another full write over
    # the workspace.  The counter layout is the generic ``uniform`` one, a
    # single draw per BLOCK lanes spread over four groups, so the generator
    # advances by exactly as much as a ``uniform_`` over the same number of
    # elements would.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(axis=0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += lane
    zero = c0 * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    scale = upper - lower
    r0 = uint_to_uniform_float(r0) * scale + lower
    r1 = uint_to_uniform_float(r1) * scale + lower
    r2 = uint_to_uniform_float(r2) * scale + lower
    r3 = uint_to_uniform_float(r3) * scale + lower

    off_0 = pid * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_0, r0, N)
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_1, r1, N)
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_2, r2, N)
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_3, r3, N)


def _can_use_contiguous_path(self, noise):
    return self.is_contiguous() and noise.is_contiguous() and self.numel() <= _INT32_MAX


def _device_ctx(device):
    """``torch_device_fn.device``, skipped when it would be a no-op.

    Switching the device costs about 9 us per launch on this backend's host,
    and the contiguous paths only ever see tensors that are already on the
    current device.  The guard is still taken when they are not.
    """
    index = device.index
    if index is None or index == torch_device_fn.current_device():
        return contextlib.nullcontext()
    return torch_device_fn.device(device)


def _next_philox_state(increment, generator=None):
    """``philox_backend_seed_offset`` with the state arithmetic kept here.

    Same seed, same offset, and the same advance of ``generator`` as the
    shared helper -- but that helper walks a per-vendor branch table and
    round-trips the state through more dispatches, which on this backend's
    host costs more than the kernel launch it feeds.  The generator state is
    two int64s at this vendor, which is what the shared helper's own
    unpacking assumes, so unpacking the same two values here is equivalent.
    """
    if generator is None:
        generator = torch_device_fn.default_generators[torch_device_fn.current_device()]
    state = generator.get_state()
    state_view = state.view(torch.int64)
    seed, offset = state_view.tolist()
    # Four lanes share one counter, so the offset advances in whole groups.
    state_view[1] = offset + (increment + 3) // 4 * 4
    generator.set_state(state)
    return seed, offset


def _launch_contiguous_eval(self, out, slope):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, _CONTIGUOUS_BLOCK_SIZE),)
    with _device_ctx(self.device):
        _rrelu_with_noise_eval_contiguous_kernel[grid](
            self,
            out,
            n_elements,
            slope,
            BLOCK_SIZE=_CONTIGUOUS_BLOCK_SIZE,
            num_warps=_CONTIGUOUS_NUM_WARPS,
        )
    return out


def _launch_contiguous_train(self, noise, out, lower, upper, generator):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    heuristics = {"N": n_elements}
    block = _UNIFORM_HEURISTICS["BLOCK"](heuristics)
    num_warps = _UNIFORM_HEURISTICS["num_warps"](heuristics)
    grid = (triton.cdiv(n_elements, block * _TRAIN_UNROLL),)
    philox_seed, philox_offset = _next_philox_state(
        triton.cdiv(n_elements, _TRAIN_UNROLL), generator=generator
    )
    with _device_ctx(self.device):
        _rrelu_with_noise_train_contiguous_kernel[grid](
            self,
            noise,
            out,
            n_elements,
            float(lower),
            float(upper),
            philox_seed,
            philox_offset,
            BLOCK=block,
            num_warps=num_warps,
        )
    return out


def _check_rrelu_with_noise_args(self, noise, lower, upper):
    if self.shape != noise.shape:
        raise RuntimeError(
            "noise tensor must have the same shape as self. "
            f"Got self.shape = {tuple(self.shape)} "
            f"and noise.shape = {tuple(noise.shape)}"
        )
    if self.device != noise.device:
        raise RuntimeError(
            f"self and noise must be on the same device, got "
            f"{self.device} and {noise.device}"
        )
    if self.dtype != noise.dtype:
        raise RuntimeError(
            f"self and noise must have the same dtype, got "
            f"{self.dtype} and {noise.dtype}"
        )
    if not self.is_floating_point():
        raise RuntimeError(
            f"rrelu_with_noise is not implemented for dtype {self.dtype}"
        )
    if not math.isfinite(float(lower)):
        raise RuntimeError(f"rrelu: lower bound must be finite, got {lower}")
    if not math.isfinite(float(upper)):
        raise RuntimeError(f"rrelu: upper bound must be finite, got {upper}")
    if float(lower) > float(upper):
        raise RuntimeError(
            f"Lower bound should be less than or equal to the upper bound, "
            f"got lower={lower} and upper={upper}"
        )


def _new_output(self):
    # ``aten::rrelu_with_noise`` returns the out-of-place result in legacy
    # contiguous layout, whatever the input layout is, so the allocation cannot
    # keep ``empty_like``'s preserve_format default: a channels-last or strided
    # input would otherwise come back in its own layout.
    return torch.empty_like(self, memory_format=torch.contiguous_format)


def _rrelu_with_noise_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
    out=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper)

    if self.numel() == 0:
        return _new_output(self) if out is None else out

    # `out` is either None (allocate) or `self` (in-place variant); anything
    # else is not reachable through the public API and takes the generic path.
    inplace = out is not None and out is self
    allocate = out is None
    fast_path = (inplace or allocate) and _can_use_contiguous_path(self, noise)

    if not training:
        slope = (float(lower) + float(upper)) * 0.5
        if fast_path:
            return _launch_contiguous_eval(
                self, self if inplace else _new_output(self), slope
            )
    elif fast_path:
        # Training draws the noise in the kernel, so nothing has to be filled
        # beforehand here.
        return _launch_contiguous_train(
            self, noise, self if inplace else _new_output(self), lower, upper, generator
        )

    # Strided inputs and tensors too large for an int32 flat offset take the
    # generic implementation, in both modes.  Only strided training still fills
    # the workspace up front -- the one path where the draw cannot move into
    # the kernel.
    return _generic_impl(self, noise, lower, upper, training, generator, out=out)


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise (Hygon backend)."""
    logger.debug("GEMS_HYGON RRELU_WITH_NOISE")
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise_ (Hygon backend)."""
    logger.debug("GEMS_HYGON RRELU_WITH_NOISE_")
    return _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator, out=self
    )

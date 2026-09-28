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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333

# One Philox call yields four values, so a program of the fused training kernel
# covers ``BLOCK * _UNROLL`` elements; the kernel calls its group helper once
# per value, and spells that 4 out because a kernel cannot read this module
# constant.  This is the tiling the ``uniform`` kernel uses, and it has to stay
# that way: the counter of a value follows from its element index, so the same
# generator state drawn on a different tiling gives different numbers.
_UNROLL = 4

# The fused training kernel addresses elements by a flat offset, which the
# generated kernel carries as an int32.
_INT32_MAX = torch.iinfo(torch.int32).max


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_train(self, noise):
    # ATen samples for self <= 0 (including signed zero), and records one for
    # positive/NaN elements. Keeping this predicate aligned with backward is
    # important because noise is the training-time gradient multiplier.
    not_positive = self <= 0
    effective_noise = tl.where(not_positive, noise, 1.0)
    output = tl.where(not_positive, self * effective_noise, self)
    return output, effective_noise


@pointwise_dynamic(
    is_tensor=[True, False],
    num_outputs=1,
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_eval(self, slope):
    return tl.where(self > 0, self, self * slope)


@triton.jit
def _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off, sampled, N):
    """Apply rrelu to one group of elements with its sampled slopes."""
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0)

    not_positive = x <= 0
    effective_noise = tl.where(not_positive, sampled, 1.0)

    tl.store(out_ptr + off, tl.where(not_positive, x * effective_noise, x), mask=mask)
    tl.store(noise_ptr + off, effective_noise, mask=mask)


@triton.heuristics(runtime.get_heuristic_config("uniform"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _rrelu_with_noise_train_kernel(
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
    """Training over a contiguous tensor, drawing the slopes in this kernel.

    ATen samples the training slopes inside its training kernel, and drawing
    them here too is what keeps this operator at three passes over the data.
    Filling the workspace with ``uniform_`` beforehand costs a separate pass
    over ``n`` elements, and the training kernel reads that workspace straight
    back, so the same work costs five passes.

    The counter arithmetic below is ``uniform_kernel``'s.  It has to be: a
    value's counter follows from its element index, so the same generator state
    has to be consumed on the same tiling to draw the same values.
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    zero = c0 * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    scale = upper - lower
    r0 = uint_to_uniform_float(r0) * scale + lower
    r1 = uint_to_uniform_float(r1) * scale + lower
    r2 = uint_to_uniform_float(r2) * scale + lower
    r3 = uint_to_uniform_float(r3) * scale + lower

    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_0, r0, N)
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_1, r1, N)
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_2, r2, N)
    _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_3, r3, N)


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


def _fill_training_noise(noise, lower, upper, generator):
    # For a strided workspace, sample contiguously and let the training kernel
    # scatter effective noise into the caller's layout while producing output.
    if noise.is_contiguous():
        noise.uniform_(float(lower), float(upper), generator=generator)
        return noise

    sampled = torch.empty_like(noise, memory_format=torch.contiguous_format)
    sampled.uniform_(float(lower), float(upper), generator=generator)
    return sampled


def _new_output(self):
    # ``aten::rrelu_with_noise`` returns the out-of-place result in legacy
    # contiguous layout, whatever the input layout is, so the allocation cannot
    # keep ``empty_like``'s preserve_format default: a channels-last or strided
    # input would otherwise come back in its own layout.
    return torch.empty_like(self, memory_format=torch.contiguous_format)


def _launch_contiguous_train(self, noise, out, lower, upper, generator):
    """Run the training kernel that draws its own slopes.

    The generator has to advance by as much as a ``uniform_`` over the same
    number of elements would move it, so the increment is the number of
    counters the kernel consumes, one per four elements.
    """
    n_elements = self.numel()
    grid_fn = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"] * _UNROLL),)
    increment = triton.cdiv(n_elements, _UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        _rrelu_with_noise_train_kernel[grid_fn](
            self,
            noise,
            out,
            n_elements,
            float(lower),
            float(upper),
            philox_seed,
            philox_offset,
        )
    return out


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

    if training:
        # The fused kernel addresses elements by flat offset, so it covers
        # contiguous inputs whose element count fits an int32 index.  Strided
        # inputs and larger tensors keep the workspace-filling path, which is
        # what makes the workspace necessary in the first place.
        if (
            self.is_contiguous()
            and noise.is_contiguous()
            and self.numel() <= _INT32_MAX
        ):
            output = out if out is not None else _new_output(self)
            return _launch_contiguous_train(
                self, noise, output, lower, upper, generator
            )
        sampled_noise = _fill_training_noise(noise, lower, upper, generator)
        output = out if out is not None else _new_output(self)
        _rrelu_with_noise_train(self, sampled_noise, out0=output, out1=noise)
        return output
    else:
        slope = (float(lower) + float(upper)) * 0.5
        output = out if out is not None else _new_output(self)
        _rrelu_with_noise_eval(self, slope, out0=output)
        return output


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise.

    Backward is not built here.  This operator is registered on the device
    dispatch key, so ``aten::rrelu_with_noise`` keeps the autograd kernel PyTorch
    generates from its derivative formula, and that kernel calls the separately
    registered ``aten::rrelu_with_noise_backward`` op.  Calling this Python API
    outside the dispatcher therefore returns a forward-only result.
    """
    logger.debug("GEMS RRELU_WITH_NOISE")
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise_.

    Backward is provided the same way as ``rrelu_with_noise``.
    """
    logger.debug("GEMS RRELU_WITH_NOISE_")
    return _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator, out=self
    )

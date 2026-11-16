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

# Kunlunxin (XPU) override of rrelu_with_noise / rrelu_with_noise_.
#
# The generic training kernel carries
# ``@triton.heuristics(runtime.get_heuristic_config("uniform"))``, letting the
# heuristic supply BLOCK and num_warps at launch time.  On the XPU triton fork
# that heuristic-supplied launch path is pathologically slow / intermittent (see
# the ``uniform_`` / ``bernoulli_`` overrides in this tree): launches stall for
# seconds, which for the in-place training path shows up as multi-second cases
# and device event-wait timeouts.
#
# Launching the *same* kernel with BLOCK and num_warps passed explicitly from
# Python (heuristic decorator removed) is robustly fast.  This override drops the
# decorator and computes the launch config in the Python wrapper, mirroring the
# "uniform" heuristic values exactly.  The training/eval kernel bodies, the
# argument checks and the eval pointwise path are reused from the generic module,
# so correctness cannot drift.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.rrelu_with_noise import (
    _INT32_MAX,
    _UNROLL,
    DEFAULT_LOWER,
    DEFAULT_UPPER,
    _check_rrelu_with_noise_args,
    _new_output,
    _rrelu_with_noise_eval,
    _rrelu_with_noise_train,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@triton.jit
def _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off, sampled, N):
    """Apply rrelu to one group of elements with its sampled slopes."""
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0)

    not_positive = x <= 0
    effective_noise = tl.where(not_positive, sampled, 1.0)

    tl.store(out_ptr + off, tl.where(not_positive, x * effective_noise, x), mask=mask)
    tl.store(noise_ptr + off, effective_noise, mask=mask)


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

    Identical to the generic kernel except that the ``uniform`` heuristic
    decorator is intentionally absent: BLOCK and num_warps are passed explicitly
    from the Python wrapper so the slow heuristic-supplied launch path is never
    taken.  The counter arithmetic is ``uniform_kernel``'s, so the same generator
    state is consumed on the same tiling and the drawn values match.
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


def _launch_config(N):
    # Mirrors the "uniform" heuristic values (uniform_heur_block /
    # uniform_heur_num_warps), but computed in Python and passed explicitly so
    # the slow heuristic-supplied launch path is never taken.
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    else:
        return 1024, 16


def _launch_contiguous_train(self, noise, out, lower, upper, generator):
    """Run the training kernel that draws its own slopes.

    The generator has to advance by as much as a ``uniform_`` over the same
    number of elements would move it, so the increment is the number of
    counters the kernel consumes, one per four elements.
    """
    n_elements = self.numel()
    block, num_warps = _launch_config(n_elements)
    grid = (triton.cdiv(n_elements, block * _UNROLL),)
    increment = triton.cdiv(n_elements, _UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        _rrelu_with_noise_train_kernel[grid](
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


def _fill_training_noise(noise, lower, upper, generator):
    """Sample the training workspace, handling equal bounds.

    The XPU ``uniform_`` rejects a degenerate interval (``from == to``) with an
    INVALID PARAMETER, while CPU aten fills a constant.  The training tests use
    equal bounds to make the slope deterministic, and they reach this path for
    strided / oversized inputs, so a degenerate interval is filled directly.
    """
    lo, up = float(lower), float(upper)
    if noise.is_contiguous():
        if lo == up:
            noise.fill_(lo)
        else:
            noise.uniform_(lo, up, generator=generator)
        return noise

    sampled = torch.empty_like(noise, memory_format=torch.contiguous_format)
    if lo == up:
        sampled.fill_(lo)
    else:
        sampled.uniform_(lo, up, generator=generator)
    return sampled


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
        # inputs and larger tensors keep the workspace-filling path.
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
        # The caller's noise buffer and (for the in-place variant) the output
        # may be strided views; the XPU pointwise kernel rejects a strided
        # output, so compute on contiguous buffers and copy the results back.
        self_c = self.contiguous()
        out_c = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        noise_c = torch.empty(
            sampled_noise.shape,
            dtype=sampled_noise.dtype,
            device=sampled_noise.device,
        )
        _rrelu_with_noise_train(self_c, sampled_noise, out0=out_c, out1=noise_c)
        noise.copy_(noise_c)
        if out is not None:
            out.copy_(out_c)
            return out
        return out_c
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
    """FlagGems implementation of aten.rrelu_with_noise (Kunlunxin XPU)."""
    logger.debug("GEMS_KUNLUNXIN RRELU_WITH_NOISE")
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise_ (Kunlunxin XPU)."""
    logger.debug("GEMS_KUNLUNXIN RRELU_WITH_NOISE_")
    return _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator, out=self
    )

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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils.random_utils import philox_backend_seed_offset

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

# Hash-RNG grid cap: GCU400 ships 24 SIPs and 48 CTAs (2 per SIP) is the
# measured concurrency sweet spot for streaming kernels (dropout/xlogy/
# rrelu_with_noise_backward precedent).
_NUM_SLOTS = 48

# 2^-24 / 2^-16: the fp32 sampling grid of one hash word, full and the
# 16-bit split used for reduced dtypes (uniform_ precedent).
_U24 = tl.constexpr(5.960464477539063e-8)
_U16 = tl.constexpr(1.52587890625e-5)

# Non-contiguous in-place calls are computed on contiguous copies and written
# back through the native copy; ``copy_`` has no private-use kernel for this
# keyset yet, so redispatch it past the GCU backend.
_NATIVE_COPY_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _copy_back(dst, src):
    torch.ops.aten.copy_.default.redispatch(_NATIVE_COPY_KEYSET, dst, src, False)


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


@libentry()
@triton.jit(do_not_specialize=["slope", "N"])
def _rrelu_with_noise_eval_kernel(
    x_ptr,
    out_ptr,
    N,
    slope,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Eval over a contiguous tensor: the flat grid-stride streaming form.

    The generic pointwise body measures ~0.7 TB/s on the core large shapes
    (11-12 bench log) while this backend's flat 1R/1W streaming kernels reach
    the ~2.9 TB/s DRAM floor (rrelu_with_noise_backward/xlogy precedent), so
    contiguous eval calls take the flat kernel; strided layouts keep the
    pointwise path.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + lane
        if EVEN:
            x = tl.load(x_ptr + off)
            tl.store(out_ptr + off, tl.where(x > 0, x, x * slope))
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask, other=0.0)
            tl.store(out_ptr + off, tl.where(x > 0, x, x * slope), mask=mask)


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer: 8 cheap integer ops per uniform,
    # full avalanche (same hash family as this backend's dropout / randn /
    # uniform_ kernels).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@triton.jit
def _rrelu_with_noise_train_group(
    x_ptr, noise_ptr, out_ptr, off, sampled, N, EVEN: tl.constexpr
):
    """Apply rrelu to one group of elements with its sampled slopes."""
    # ATen samples for self <= 0 (including signed zero), and records one for
    # positive/NaN elements. Keeping this predicate aligned with backward is
    # important because noise is the training-time gradient multiplier.
    # ``out = x * effective_noise`` is written unconditionally: on the
    # positive arm it multiplies by the recorded unit slope, and
    # ``x * 1.0`` is bit-exact through the fp32 opmath round trip for every
    # input including NaN/Inf/signed zero. One vector multiply replaces a
    # multiply-plus-select per element.
    if EVEN:
        x = tl.load(x_ptr + off)
        effective_noise = tl.where(x <= 0, sampled, 1.0)
        tl.store(out_ptr + off, x * effective_noise)
        tl.store(noise_ptr + off, effective_noise)
    else:
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        effective_noise = tl.where(x <= 0, sampled, 1.0)
        tl.store(out_ptr + off, x * effective_noise, mask=mask)
        tl.store(noise_ptr + off, effective_noise, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "lower", "upper", "N"])
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
    REDUCED: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Training over a contiguous tensor, drawing the slopes in this kernel.

    RNG note: the aten ``rrelu_with_noise`` accuracy contract in this suite is
    statistical (slope coverage of ``[lower, upper]``, per-generator-state
    reproducibility, stream advance) -- see tests/test_rrelu_with_noise.py.
    The philox4x32-10 chain costs ~15 dependent integer vector ops per element
    and dominated the wall of the previous philox version of this kernel
    (measured on uniform_: philox+conversion bucket 5.3 ms of a 6.07 ms total
    on 1 Gi fp16 vs a 0.72 ms store floor). This kernel therefore hashes the
    global element offset with the same xxhash32-style avalanche finalizer
    used by this backend's dropout / randn / uniform_ kernels. The per-call
    state is still drawn from / advances the torch philox generator via
    philox_backend_seed_offset (byte-identical increment accounting), so
    successive calls yield fresh independent samples and stay reproducible
    for a fixed generator state. The raw per-call state is avalanched into
    s0 first, otherwise two same-shape calls would return the same multiset
    of samples, merely permuted (dropout precedent, device-verified). 16-bit
    dtypes (REDUCED=True) split one 32-bit hash into a hi/lo pair of 16-bit
    uniforms, halving the hash budget; the fp32-domain slope arithmetic and
    the single rounding store match the aten opmath contract.
    """
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32(rng_state.to(tl.uint32)) ^ _fmix32((rng_state >> 32).to(tl.uint32))
    scale = upper - lower
    if REDUCED:
        s_u = scale * _U16
    else:
        s_u = scale * _U24
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)

    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * (BLOCK * 4) + lane
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        if REDUCED:
            # one hash feeds a pair of BLOCK lanes (groups 0/1 and 2/3)
            ha = _fmix32(off_0.to(tl.uint32) + s0)
            hb = _fmix32(off_2.to(tl.uint32) + s0)
            u0 = ha >> 16
            u1 = ha & 0xFFFF
            u2 = hb >> 16
            u3 = hb & 0xFFFF
        else:
            u0 = _fmix32(off_0.to(tl.uint32) + s0) >> 8
            u1 = _fmix32(off_1.to(tl.uint32) + s0) >> 8
            u2 = _fmix32(off_2.to(tl.uint32) + s0) >> 8
            u3 = _fmix32(off_3.to(tl.uint32) + s0) >> 8
        # slopes stay in fp32: RN(u * s_u + lower) with the grid scale
        # prefolded, matching the aten opmath (fp32 accumulate, one rounding
        # store into the output/noise dtype).
        r0 = tl.fma(u0.to(tl.float32), s_u, lower)
        r1 = tl.fma(u1.to(tl.float32), s_u, lower)
        r2 = tl.fma(u2.to(tl.float32), s_u, lower)
        r3 = tl.fma(u3.to(tl.float32), s_u, lower)
        _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_0, r0, N, EVEN)
        _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_1, r1, N, EVEN)
        _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_2, r2, N, EVEN)
        _rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_3, r3, N, EVEN)


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


def _choose_train_block_and_warps(n, itemsize):
    # BLOCK/num_warps bands (calibrated by probes on 2026-11-17, do_bench median w1000/r100,
    # full sweep of 11 configs × 3 dtypes × 3 bands, probe table in job tmp/probe_sweep2.log):
    # - tiny hits the launch floor: at n=4096, B2048/nw4 = 4.1-4.5us (from B4096, a 4x
    #   overhang cliff at 15.4us+, larger tiles 47-183us);
    # - 16-bit single band B4096/nw2: best at both 16.7M/512Mi points (from B8192, the 16-bit cliff
    #   is 5.3x, B65536 worst at 0.3x); nw2/nw4 differ by <0.3%;
    # - fp32 large band B32768/nw2 (best at both 16.7M 121.5us / 512Mi 3099.9us points),
    #   B65536 exceeds the DSM limit (Required 1048576 > 917504, confirmed by probes).
    if n <= 2048:
        return 512, 4
    if n <= 4096:
        return 2048, 4
    if itemsize == 2:
        return 4096, 2
    if n <= (1 << 21):
        return 4096, 4
    return 32768, 2


def _choose_eval_block_and_warps(n, itemsize):
    # flat streaming bands (calibrated by probes the same day): eval is 1R/1W with no 4-unroll.
    # - 16-bit large band B65536/nw2: best at both mid 28.9-29.0us / big 689.9-697.9us points;
    # - fp32 large band B65536/nw2 hits the DRAM floor (512Mi 1398.1us = 2.93TB/s),
    #   at the 16.7M band B32768/nw2 is slightly better (47.8us), so the cutoff is 2^24.
    if n <= 4096:
        return max(triton.next_power_of_2(n) // 2, 256), 2
    if n <= 65536:
        return 2048, 4
    if n <= (1 << 21):
        return 8192, 4
    if itemsize == 2:
        return 65536, 2
    if n <= (1 << 24):
        return 32768, 2
    return 65536, 2


def _launch_contiguous_eval(self, out, slope):
    n_elements = self.numel()
    block, num_warps = _choose_eval_block_and_warps(n_elements, self.element_size())
    grid = (min(triton.cdiv(n_elements, block), _NUM_SLOTS),)
    even = n_elements % block == 0
    with torch_device_fn.device(self.device):
        # Retry on occasional P02 cold-compilation failures (same as _launch_contiguous_train).
        for attempt in range(3):
            try:
                _rrelu_with_noise_eval_kernel[grid](
                    self,
                    out,
                    n_elements,
                    float(slope),
                    BLOCK=block,
                    EVEN=even,
                    num_warps=num_warps,
                    num_stages=1,
                )
                return out
            except Exception:
                if attempt == 2:
                    raise
    return out


def _launch_contiguous_train(self, noise, out, lower, upper, generator):
    """Run the training kernel that draws its own slopes.

    The generator has to advance by as much as a ``uniform_`` over the same
    number of elements would move it, so the increment is the number of
    philox counters ``uniform_`` would consume, one per four elements.
    """
    n_elements = self.numel()
    increment = triton.cdiv(n_elements, _UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    block, num_warps = _choose_train_block_and_warps(n_elements, self.element_size())
    grid = (min(triton.cdiv(n_elements, block * _UNROLL), _NUM_SLOTS),)
    even = n_elements % (block * _UNROLL) == 0
    with torch_device_fn.device(self.device):
        # P02: the external llvm converter occasionally fails on cold compilation (converterResult=-2 family),
        # retrying the launch to trigger recompilation recovers (precedent: xlogy/rrelu_with_noise_backward).
        for attempt in range(3):
            try:
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
                    REDUCED=self.element_size() == 2,
                    EVEN=even,
                    num_warps=num_warps,
                    num_stages=1,
                )
                return out
            except Exception:
                if attempt == 2:
                    raise
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
        # Contiguous inputs use the flat streaming kernel (1R/1W hits the DRAM floor); inputs with a layout
        # still take the pointwise path.
        if (
            self.is_contiguous()
            and output.is_contiguous()
            and self.numel() <= _INT32_MAX
        ):
            return _launch_contiguous_eval(self, output, slope)
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

    The training kernel writes the effective noise back through a strided
    store, which mis-addresses the tail on this backend (measured with a
    channels-last workspace: the result was right, the noise buffer was not).
    Sample into a contiguous copy and write it back with the native copy.
    """
    logger.debug("GEMS_ENFLAME RRELU_WITH_NOISE")
    if training and not noise.is_contiguous():
        noise_c = noise.contiguous()
        output = _rrelu_with_noise_impl(
            self, noise_c, lower, upper, training, generator
        )
        _copy_back(noise, noise_c)
        return output
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

    A strided ``self``/``noise`` cannot be the kernel's output on this backend:
    the pointwise store mis-addresses the tail when ``stride(-1) != 1``
    (measured with a ``[:, ::2]`` view -- both tensors came back wrong).  Run
    the implementation on contiguous copies and copy the results back.
    """
    logger.debug("GEMS_ENFLAME RRELU_WITH_NOISE_")
    if self.is_contiguous() and noise.is_contiguous():
        return _rrelu_with_noise_impl(
            self, noise, lower, upper, training, generator, out=self
        )

    self_c = self.contiguous()
    noise_c = noise.contiguous()
    result = _rrelu_with_noise_impl(
        self_c, noise_c, lower, upper, training, generator, out=self_c
    )
    _copy_back(self, result)
    if training:
        # Training consumes/updates the noise workspace; eval leaves it alone.
        _copy_back(noise, noise_c)
    return self

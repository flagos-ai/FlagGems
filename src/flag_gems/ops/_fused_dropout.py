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
import threading

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import uint_to_uniform_float

logger = logging.getLogger(__name__)

_rng_bridge = None
_rng_bridge_lock = threading.Lock()

# A Python lock cannot serialize a get_state/set_state transaction with native
# random kernels. Use the Generator's own mutex and virtual offset interface;
# no assumptions about the backend's serialized state layout are needed.
_RNG_SOURCE = r"""
#include <ATen/core/Generator.h>
#include <torch/csrc/utils/pybind.h>
#include <limits>
#include <mutex>

std::pair<uint64_t, uint64_t> reserve(
    at::Generator generator, c10::Device device, uint64_t words) {
  TORCH_CHECK(generator.defined(), "_fused_dropout requires a Generator");
  TORCH_CHECK(generator.device() == device,
              "_fused_dropout generator device does not match input device");
  constexpr auto limit = std::numeric_limits<uint64_t>::max();
  TORCH_CHECK(words <= limit - 3, "_fused_dropout RNG increment overflow");
  const auto increment = ((words + 3) / 4) * 4;
  std::lock_guard<std::mutex> lock(generator.mutex());
  const auto seed = generator.current_seed();
  const auto offset = generator.get_offset();
  TORCH_CHECK(offset % 4 == 0, "_fused_dropout requires a word-aligned Philox offset");
  TORCH_CHECK(increment <= limit - offset, "_fused_dropout RNG offset overflow");
  generator.set_offset(offset + increment);
  return {seed, offset};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("reserve", &reserve, pybind11::call_guard<pybind11::gil_scoped_release>());
}
"""


class _PythonRngBridge:
    # This lock only coordinates this operator's Python fallback calls. Native
    # random operations do not acquire it, even when they share the Generator.
    uses_native_mutex = False

    @staticmethod
    def reserve(generator, device, words):
        if generator.device != device:
            raise RuntimeError(
                "_fused_dropout generator device does not match input device"
            )
        limit = (1 << 64) - 1
        if words < 0 or words > limit - 3:
            raise RuntimeError("_fused_dropout RNG increment overflow")
        increment = (words + 3) // 4 * 4
        with _rng_bridge_lock:
            seed = generator.initial_seed()
            offset = generator.get_offset()
            if offset % 4:
                raise RuntimeError(
                    "_fused_dropout requires a word-aligned Philox offset"
                )
            if increment > limit - offset:
                raise RuntimeError("_fused_dropout RNG offset overflow")
            generator.set_offset(offset + increment)
        return seed, offset


def _get_rng_bridge():
    global _rng_bridge
    if _rng_bridge is None:
        with _rng_bridge_lock:
            if _rng_bridge is None:
                build_options = dict(
                    name="flag_gems_fused_dropout_rng",
                    cpp_sources=_RNG_SOURCE,
                    extra_cflags=["-O2"],
                    with_cuda=False,
                    verbose=False,
                )
                try:
                    from torch.utils.cpp_extension import load_inline

                    _rng_bridge = load_inline(**build_options)
                except (RuntimeError, OSError, ImportError) as error:
                    logger.warning(
                        "_fused_dropout RNG bridge unavailable (%s); using Triton "
                        "with Python-serialized Generator offsets. Do not share "
                        "this Generator with concurrent native random operations "
                        "or other RNG helpers: the native mutex is unavailable.",
                        (
                            str(error).splitlines()[0]
                            if str(error)
                            else type(error).__name__
                        ),
                    )
                    _rng_bridge = _PythonRngBridge()
    return _rng_bridge


@triton.jit(do_not_specialize=["seed", "offset"])
def _fused_dropout_kernel(
    X,
    Y,
    M,
    N,
    seed,
    offset,
    P: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
    LEGACY_CONSTEXPR: tl.constexpr = False,
):
    group = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # Generator offsets count uint32 words, whereas a Philox counter produces
    # four words. Add in 64 bits before splitting, including across 2**32.
    counter = (offset.to(tl.uint64) >> 2) + group.to(tl.uint64)
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    zero = tl.full((BLOCK,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed.to(tl.uint64), c0, c1, zero, zero)
    component = tl.arange(0, 4)[None, :]
    bits = tl.where(
        component == 0,
        r0[:, None],
        tl.where(
            component == 1,
            r1[:, None],
            tl.where(component == 2, r2[:, None], r3[:, None]),
        ),
    )
    random = uint_to_uniform_float(bits)
    index = group[:, None] * 4 + component
    valid = index < N
    if FLAT:
        x_index = index
        y_index = index
    else:
        remaining = index
        x_index = tl.full((BLOCK, 4), 0, tl.int64)
        y_index = tl.full((BLOCK, 4), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            # CoreX 3.1 leaves constexpr tuples wrapped at subscripting.
            if LEGACY_CONSTEXPR:
                coordinate = remaining % SHAPE.value[axis]
                remaining = remaining // SHAPE.value[axis]
                x_index += coordinate * X_STRIDES.value[axis]
                y_index += coordinate * Y_STRIDES.value[axis]
            else:
                coordinate = remaining % SHAPE[axis]
                remaining = remaining // SHAPE[axis]
                x_index += coordinate * X_STRIDES[axis]
                y_index += coordinate * Y_STRIDES[axis]
    compute_type: tl.constexpr = (
        tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    )
    probability = tl.full((), P, compute_type)
    keep = random.to(compute_type) < probability
    scale = 1.0 / probability
    x = tl.load(X + x_index, valid, other=0).to(compute_type)
    # Multiplication (rather than where) preserves NaN/Inf and p=0 semantics.
    y = x * keep.to(compute_type) * scale
    tl.store(Y + y_index, y, valid)
    tl.store(M + y_index, keep.to(tl.uint8), valid)


# Triton Ascend 3.2 crashes when lowering the (BLOCK, 4) broadcast above.
# Contiguous 1D word lanes also avoid its strided-store UB addressing bug.
# Repeating each Philox group four times preserves the generic word stream.
@triton.jit(do_not_specialize=["seed", "offset"])
def _fused_dropout_kernel_ascend_32(
    X,
    Y,
    M,
    N,
    seed,
    offset,
    P: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * (BLOCK * 4) + tl.arange(0, BLOCK * 4)
    group = index // 4
    counter = (offset.to(tl.uint64) >> 2) + group.to(tl.uint64)
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    zero = tl.full((BLOCK * 4,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed.to(tl.uint64), c0, c1, zero, zero)
    component = index % 4
    bits = tl.where(
        component == 0,
        r0,
        tl.where(component == 1, r1, tl.where(component == 2, r2, r3)),
    )
    random = uint_to_uniform_float(bits)
    valid = index < N
    if FLAT:
        x_index = index
        y_index = index
    else:
        remaining = index
        x_index = tl.full((BLOCK * 4,), 0, tl.int64)
        y_index = tl.full((BLOCK * 4,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE.value[axis]
            remaining = remaining // SHAPE.value[axis]
            x_index += coordinate * X_STRIDES.value[axis]
            y_index += coordinate * Y_STRIDES.value[axis]
    compute_type: tl.constexpr = (
        tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    )
    probability = tl.full((), P, compute_type)
    keep = random.to(compute_type) < probability
    scale = 1.0 / probability
    x = tl.load(X + x_index, valid, other=0).to(compute_type)
    y = x * keep.to(compute_type) * scale
    tl.store(Y + y_index, y, valid)
    tl.store(M + y_index, keep.to(tl.uint8), valid)


@torch.compiler.disable
def _fused_dropout(self, p, generator=None):
    """Dropout with keep probability and a Generator-backed Philox stream.

    Eager replay uses the backend Generator. Default-generator checkpoint
    replay also requires a working backend checkpoint adapter; explicit
    generators require caller-managed state restoration. CUDA graph capture
    is rejected; device graph replay is unsupported.
    CUDA bitwise equivalence and its launch-dependent counter consumption are
    not promised. The native mutex bridge requires a C++ compiler and Ninja.
    If it cannot be built, Python-serialized reservations support eager replay
    but must not share a Generator with concurrent native/other RNG operations.
    """
    logger.debug("GEMS _FUSED_DROPOUT")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError("_fused_dropout supports floating-point inputs only")
    if self.dtype == torch.float64 and flag_gems.vendor_name in (
        "ascend",
        "iluvatar",
        "mthreads",
    ):
        raise RuntimeError("_fused_dropout float64 is unsupported on this backend")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("_fused_dropout generator must be a torch.Generator")
        if generator.device != self.device:
            raise RuntimeError(
                "_fused_dropout generator device does not match input device"
            )
    output = torch.empty_like(self)
    mask = torch.empty_like(self, dtype=torch.uint8)
    n = self.numel()
    if n == 0:
        return self.clone(), mask
    # torch_npu empty_like currently makes even dense permuted inputs contiguous.
    # The allocation still has exactly the dense storage extent we need.
    if (
        flag_gems.vendor_name == "ascend"
        and self.stride() != output.stride()
        and torch.ops.aten.is_non_overlapping_and_dense(self)
    ):
        output = output.as_strided(self.shape, self.stride())
        mask = mask.as_strided(self.shape, self.stride())
    with torch_device_fn.device(self.device):
        capturing = getattr(torch_device_fn, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("_fused_dropout does not support graph capture")
        if generator is None:
            generator = torch_device_fn.default_generators[self.device.index]
        seed, offset = _get_rng_bridge().reserve(generator, self.device, n)
        flat = self.stride() == output.stride()
        # Unused dense-layout metadata must not multiply the JIT cache.
        kernel = _fused_dropout_kernel
        if flag_gems.vendor_name == "ascend" and triton.__version__.split(".")[:2] == [
            "3",
            "2",
        ]:
            kernel = _fused_dropout_kernel_ascend_32
        kernel_options = {}
        if flag_gems.vendor_name == "iluvatar" and triton.__version__.startswith(
            "3.1."
        ):
            kernel_options["LEGACY_CONSTEXPR"] = True
        kernel[(triton.cdiv(n, 1024),)](
            self,
            output,
            mask,
            n,
            seed,
            offset,
            p,
            () if flat else tuple(self.shape),
            () if flat else self.stride(),
            () if flat else output.stride(),
            flat,
            BLOCK=256,
            enable_fp_fusion=False,
            **kernel_options,
        )
    return output, mask

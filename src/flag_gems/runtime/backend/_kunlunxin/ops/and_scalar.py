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

from .bitwise_and import bitwise_and_func_scalar, bitwise_and_scalar

# Use the generic op's logger name so the functional test's
# ``caplog.at_level("DEBUG", logger="flag_gems.ops.and_scalar")`` assertion on
# the "GEMS AND SCALAR" message still fires from this backend override.
logger = logging.getLogger("flag_gems.ops.and_scalar")


def and_scalar(self, other):
    """``__and__.Scalar`` for Kunlunxin.

    ``__and__.Scalar`` and ``bitwise_and.Scalar`` are the same ATen operation's
    two entry points. The generic fallback routes through the un-tuned
    ``pointwise_dynamic`` codegen, which is ~1000x slower than Torch on XPU for
    large tensors, so we reuse the XPU-tuned ``bitwise_and_scalar`` kernel.

    For a contiguous ``int16`` input we additionally pack two 16-bit lanes into
    one 32-bit word (same trick as ``bitwise_and_scalar_``): the compare then
    runs over half as many elements, roughly doubling throughput on the
    otherwise sub-par int16 path. The AND mask ``s | (s << 16)`` applies the
    scalar identically to both packed lanes, so the result is bit-exact.
    """
    logger.debug("GEMS_KUNLUNXIN AND_SCALAR")
    if (
        isinstance(self, torch.Tensor)
        and self.dtype == torch.int16
        and self.is_contiguous()
        and isinstance(other, int)
        and not isinstance(other, bool)
        and -0x8000 <= int(other) <= 0x7FFF
    ):
        nbytes = self.numel() * self.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            s = int(other) & 0xFFFF
            mask = s | (s << 16)
            try:
                in_view = self.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_and_scalar(self, other)
            out = torch.empty_strided(
                (in_view.numel(),), (1,), dtype=torch.int32, device=self.device
            )
            bitwise_and_func_scalar(in_view, mask, out0=out)
            return out.view(torch.int16).reshape(self.shape)
    return bitwise_and_scalar(self, other)

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

import math

import triton
import triton.language as tl

import flag_gems.ops._upsample_lanczos2d_aa_backward as _generic

# ``sinpi`` has no device implementation on this backend, so the filter uses the
# equivalent ``sin(pi * x)`` form; the argument is bounded by the support radius.


@triton.jit
def _lanczos_aa_filter(x):
    """Lanczos-3 filter used by PyTorch's antialiased implementation."""
    pix = math.pi * x
    sinc = tl.where(x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_three = tl.where(x == 0.0, 1.0, tl.sin(pix / 3.0) / (pix / 3.0))
    return tl.where(x < 3.0, sinc * sinc_three, 0.0)


def _should_use_fused_path(
    device_type, total_elems, NC, H_in, W_in, BLOCK_IW, MAX_OH, MAX_OW
):
    """Always take the two-pass gather on this backend.

    The fused gather keeps ``MAX_OH * MAX_OW`` copies of the body live in one
    program, which grows quickly with the output tile. The two-pass gather
    produces the same result with a predictable cost.
    """
    return False


def _install_generic_kernels():
    """Bind the repaired pieces into the module the generic entry points use."""
    _generic._lanczos_aa_filter = _lanczos_aa_filter
    _generic._should_use_fused_path = _should_use_fused_path


_install_generic_kernels()

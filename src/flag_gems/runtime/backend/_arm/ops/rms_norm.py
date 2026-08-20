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

"""
ARM CPU fused_add_rms_norm wrapper.

Wraps the _arm/fused/fused_add_rms_norm.py Triton kernel so it can be used
as a drop-in replacement for flag_gems.fused_add_rms_norm on ARM64 CPU.

Standalone rms_norm (without residual add) was removed: A/B measurement on
Qwen3-1.7B INT8 decode showed no measurable benefit over ATen's native
Qwen3RMSNorm.forward. See _arm/fused/fused_add_rms_norm.py for the note.
"""

from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
    fused_add_rms_norm as _arm_fused_add_rms_norm,
)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    ARM CPU drop-in for flag_gems.fused_add_rms_norm.

    In-place: residual = x + residual; x = rms_norm(residual) * weight.
    Returns (x, residual).
    """
    return _arm_fused_add_rms_norm(x, residual, normalized_shape, weight, eps)

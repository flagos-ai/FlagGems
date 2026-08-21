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

# Ascend-specific mHC operator implementations.
#
# mhc_post and mhc_pre use Ascend-optimised Triton kernels (num_stages=1,
# vectorised Sinkhorn).  mhc_bwd falls back to the reference PyTorch/Triton
# CG-based implementation.

from flag_gems.runtime.backend._ascend.fused.mhc.mhc_post import mhc_post, mhc_post_ref
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_post_backward import (
    mhc_post_backward,
    mhc_post_backward_ref,
)
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_pre_clamp_sinkhorn import (
    mhc_pre_clamp_sinkhorn,
)
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_pre_clamp_sinkhorn_backward import (
    mhc_pre_clamp_sinkhorn_backward,
    mhc_pre_clamp_sinkhorn_backward_ref,
)

__all__ = [
    "mhc_post",
    "mhc_post_ref",
    "mhc_post_backward",
    "mhc_post_backward_ref",
    "mhc_pre_clamp_sinkhorn",
    "mhc_pre_clamp_sinkhorn_backward",
    "mhc_pre_clamp_sinkhorn_backward_ref",
]

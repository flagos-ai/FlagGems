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

from flag_gems.fused.mhc.hc_head_fused_kernel import (
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
)
from flag_gems.fused.mhc.hc_split_sinkhorn import (
    hc_split_sinkhorn,
    mhc_split_sinkhorn_torch_ref,
)
from flag_gems.fused.mhc.mhc_bwd import mhc_bwd, mhc_bwd_ref, sinkhorn_forward
from flag_gems.fused.mhc.mhc_post import mhc_post
from flag_gems.fused.mhc.mhc_pre import mhc_pre

__all__ = [
    "hc_head_fused_kernel",
    "hc_head_fused_kernel_ref",
    "hc_split_sinkhorn",
    "mhc_bwd",
    "mhc_bwd_ref",
    "mhc_post",
    "mhc_post_ref",
    "mhc_pre",
    "mhc_pre_ref",
    "mhc_split_sinkhorn_torch_ref",
    "sinkhorn_forward",
]

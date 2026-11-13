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

"""gcu400 te_rmsnorm_bwd: fold the zero-centred gamma on the host.

``rmsnorm_bwd_dx_kernel`` applies ``w = w + 1.0`` inside a
``zero_centered_gamma`` constexpr branch.  On this backend that branch
miscompiles for several autotuned tile shapes (measured for BRS=4/8/16 with
BCS=256/512: the fp32 dx came back 1.35 off, matching neither the centred nor
the un-centred formula; BRS=2 or num_warps=2 were correct).  Since the
autotuner's best config is shared between both flag values, an fp32
``zero_centered_gamma=True`` call could silently run the broken variant.

Fold the +1 into the weight on the host and always ask the kernel for the
un-centred path.  dgamma does not depend on the weight, so the result is
unchanged; the fp32 dtype is preserved so the output dtype stays correct.
Half-precision inputs keep the generic path (their tests pass and the host
fold would change dgamma's dtype).
"""

import torch

from flag_gems.ops.te_rmsnorm import te_rmsnorm_bwd as _generic_te_rmsnorm_bwd

__all__ = ["te_rmsnorm_bwd"]


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    if zero_centered_gamma and x.dtype == torch.float32:
        gamma = gamma + 1.0
        zero_centered_gamma = False
    return _generic_te_rmsnorm_bwd(
        dz,
        x,
        rsigma,
        gamma,
        sm_margin=sm_margin,
        zero_centered_gamma=zero_centered_gamma,
    )


# Some callers (including tests) import ``te_rmsnorm_bwd`` from the shared
# module directly, which bypasses the backend registration.  Publish the
# wrapper there too; it is only imported on this backend.
import flag_gems.ops.te_rmsnorm as _common_te_rmsnorm  # noqa: E402

_common_te_rmsnorm.te_rmsnorm_bwd = te_rmsnorm_bwd

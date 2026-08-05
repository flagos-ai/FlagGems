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

from typing import Callable

import torch

from flag_gems.runtime.backend._ascend.ops.gather_ascend import gather
from flag_gems.runtime.backend._ascend.ops.gather_collapsed_uintdiv import (
    apply_prefix_narrows,
    can_collapse_axes,
    gather_collapsed,
)


def gather_auto(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    out: torch.Tensor,
    grid_fn,
    magic_map=None,
    use_collapsed=False,
    with_negative_index=False,
) -> Callable[[], None]:
    ok, narrows = can_collapse_axes(inp, index, dim)
    if ok and use_collapsed:
        inp = apply_prefix_narrows(inp, narrows)
        run_kernel = gather_collapsed(
            inp,
            dim,
            index,
            out,
            grid_fn=grid_fn,
            return_run_kernel=True,
            with_negative_index=with_negative_index,
        )

    else:

        def run_kernel():
            gather(
                inp,
                dim,
                index,
                out,
                grid_fn=grid_fn,
                magic_map=magic_map,
                with_negative_index=with_negative_index,
            )

    return run_kernel
